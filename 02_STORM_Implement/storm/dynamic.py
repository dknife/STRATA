"""
D-STORM: Dynamic STORM for evolving graphs.

Supports incremental updates to the distance matrix when edges
are added or removed, without full recomputation.

Classes:
    DynamicStorm - Maintains distance matrix under edge updates.
"""

import numpy as np
import scipy.sparse as sp


class DynamicStorm:
    """Dynamic STORM: incremental distance matrix maintenance.

    Maintains a distance matrix D and supports edge additions/deletions
    with localized updates instead of full recomputation.

    Args:
        A: Initial adjacency matrix (scipy.sparse CSR).
        D: Precomputed distance matrix (scipy.sparse CSR or dense).
            If None, computed from scratch using storm_apsp.
    """

    def __init__(self, A, D=None):
        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        self.A = A.astype(np.float32).tocsr()
        self.n = A.shape[0]

        if D is None:
            from storm.apsp import storm_apsp
            D = storm_apsp(A, verbose=False)

        # Store D as dense for O(1) element access during updates.
        # For very large graphs, consider sparse or compressed storage.
        if sp.issparse(D):
            self.D = D.toarray().astype(np.float32)
        else:
            self.D = np.asarray(D, dtype=np.float32)

        self._update_log = []

    def add_edge(self, u, v, weight=1.0):
        """Add edge (u, v) and incrementally update distance matrix.

        For each pair (i, j), check if the new path i->u->v->j
        is shorter than the existing D[i,j].

        Args:
            u: Source node of new edge.
            v: Target node of new edge.
            weight: Edge weight (default 1.0 for unweighted).

        Returns:
            n_updated: Number of node pairs whose distance was updated.
        """
        # Update adjacency
        self.A[u, v] = weight

        # Find all nodes that can reach u (column u of D)
        # and all nodes reachable from v (row v of D)
        dist_to_u = self.D[:, u]      # dist_to_u[i] = D[i, u]
        dist_from_v = self.D[v, :]    # dist_from_v[j] = D[v, j]

        # Include u and v themselves
        # Candidate new distance: D[i,u] + weight + D[v,j]
        new_dist = dist_to_u[:, np.newaxis] + weight + dist_from_v[np.newaxis, :]

        # Mask: only update where new path is shorter
        # D==0 means unreachable (except diagonal), so handle carefully
        currently_unreachable = (self.D == 0) & ~np.eye(self.n, dtype=bool)
        improvement = (new_dist < self.D) | (currently_unreachable & (new_dist > 0))

        # Don't update diagonal
        np.fill_diagonal(improvement, False)

        # Filter: source must be able to reach u, and v must reach target
        source_can_reach_u = (dist_to_u > 0) | (np.arange(self.n) == u)
        target_reachable_from_v = (dist_from_v > 0) | (np.arange(self.n) == v)
        valid = source_can_reach_u[:, np.newaxis] & target_reachable_from_v[np.newaxis, :]

        mask = improvement & valid
        n_updated = mask.sum()

        if n_updated > 0:
            self.D[mask] = new_dist[mask]

        self._update_log.append(('add', u, v, n_updated))
        return n_updated

    def delete_edge(self, u, v):
        """Delete edge (u, v) and update affected distances.

        Identifies node pairs whose shortest path used edge (u,v),
        then recomputes only those distances via localized BFS.

        Args:
            u: Source node of deleted edge.
            v: Target node of deleted edge.

        Returns:
            n_affected: Number of node pairs that needed recomputation.
        """
        # Remove from adjacency
        self.A[u, v] = 0
        self.A.eliminate_zeros()

        # Find affected pairs: those whose shortest path goes through (u,v)
        # D[i,j] == D[i,u] + 1 + D[v,j] means the path *could* use (u,v)
        dist_to_u = self.D[:, u]
        dist_from_v = self.D[v, :]

        via_uv = dist_to_u[:, np.newaxis] + 1 + dist_from_v[np.newaxis, :]
        affected = (self.D > 0) & (np.abs(self.D - via_uv) < 0.5)
        np.fill_diagonal(affected, False)

        n_affected = affected.sum()

        if n_affected > 0:
            # Recompute affected distances via BFS from affected sources
            affected_sources = np.where(affected.any(axis=1))[0]

            A_csr = self.A.tocsr()
            for src in affected_sources:
                affected_targets = np.where(affected[src])[0]
                distances = self._bfs_distances(A_csr, src, affected_targets)
                for tgt, dist in zip(affected_targets, distances):
                    self.D[src, tgt] = dist

        self._update_log.append(('del', u, v, n_affected))
        return n_affected

    def add_edges_batch(self, edges):
        """Add multiple edges in batch.

        Args:
            edges: List of (u, v) tuples or (u, v, weight) tuples.

        Returns:
            total_updated: Total number of distance updates.
        """
        total = 0
        for edge in edges:
            if len(edge) == 2:
                total += self.add_edge(edge[0], edge[1])
            else:
                total += self.add_edge(edge[0], edge[1], edge[2])
        return total

    def get_distance(self, i, j):
        """Query shortest path distance between nodes i and j."""
        return self.D[i, j]

    def get_distance_matrix(self):
        """Return current distance matrix."""
        return self.D.copy()

    def get_sparse_distance_matrix(self):
        """Return current distance matrix as sparse matrix."""
        return sp.csr_matrix(self.D)

    def _bfs_distances(self, A_csr, source, targets):
        """Compute BFS distances from source to specific targets.

        Args:
            A_csr: Adjacency matrix in CSR format.
            source: Source node.
            targets: Array of target node indices.

        Returns:
            distances: Array of distances (0 if unreachable).
        """
        n = A_csr.shape[0]
        dist = np.zeros(n, dtype=np.float32)
        visited = np.zeros(n, dtype=bool)
        visited[source] = True

        current_level = [source]
        step = 0
        target_set = set(targets)
        found = set()

        while current_level and len(found) < len(target_set):
            step += 1
            next_level = []
            for node in current_level:
                start, end = A_csr.indptr[node], A_csr.indptr[node + 1]
                neighbors = A_csr.indices[start:end]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        dist[nb] = step
                        next_level.append(nb)
                        if nb in target_set:
                            found.add(nb)
            current_level = next_level

        return dist[targets]

    @property
    def update_history(self):
        """Return log of all updates performed."""
        return self._update_log
