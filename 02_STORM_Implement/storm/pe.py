"""
STORM-PE: Positional Encoding for Graph Neural Networks
based on k-order reachability profiles.

STORM-PE encodes each node's structural position using the number of
nodes reachable at exactly k hops for k=1,...,K. This provides:
  - Deterministic (non-stochastic) structural encoding
  - Native support for directed and disconnected graphs
  - No sign ambiguity (unlike Laplacian PE)
  - Multi-scale structural information

Classes:
    StormPE               - PyTorch module for STORM positional encoding
    StormEnhancedGNN      - GNN with STORM-PE integration
Functions:
    compute_reachability_profile - Compute reachability profiles for all nodes
"""

import numpy as np
import scipy.sparse as sp

from storm.core import SparseStormIterator


def compute_reachability_profile(A, K=10, directed=True):
    """Compute k-order reachability profile for each node.

    For each node v, the profile is:
        r_v = [r_v^(1), r_v^(2), ..., r_v^(K)]
    where r_v^(k) = number of nodes reachable from v at exactly k hops.

    For directed graphs, computes both outgoing and incoming profiles:
        r_v^dir = [r_v^(1)_out, r_v^(1)_in, ..., r_v^(K)_out, r_v^(K)_in]

    Args:
        A: Adjacency matrix (scipy.sparse or np.ndarray).
        K: Maximum reachability order.
        directed: If True, compute bidirectional profiles (out + in).

    Returns:
        profiles: np.ndarray of shape (n, K) or (n, 2*K) if directed.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)
    n = A.shape[0]

    # Outgoing reachability: how many nodes can v reach at k hops
    out_profiles = np.zeros((n, K), dtype=np.float32)

    # First order: direct neighbors
    A_bool = A.copy()
    A_bool.data[:] = 1.0
    out_profiles[:, 0] = np.array(A_bool.sum(axis=1)).flatten()

    # Higher orders via STORM iterator
    iterator = SparseStormIterator(A, k=K)
    for Rk_star, power in iterator:
        if power - 1 < K:
            out_profiles[:, power - 1] = np.array(
                Rk_star.sum(axis=1)
            ).flatten()

    if not directed:
        return out_profiles

    # Incoming reachability: how many nodes can reach v at k hops
    # Compute on A^T
    A_T = A.T.tocsr()
    in_profiles = np.zeros((n, K), dtype=np.float32)

    A_T_bool = A_T.copy()
    A_T_bool.data[:] = 1.0
    in_profiles[:, 0] = np.array(A_T_bool.sum(axis=1)).flatten()

    iterator_T = SparseStormIterator(A_T, k=K)
    for Rk_star, power in iterator_T:
        if power - 1 < K:
            in_profiles[:, power - 1] = np.array(
                Rk_star.sum(axis=1)
            ).flatten()

    # Interleave: [out_1, in_1, out_2, in_2, ...]
    profiles = np.zeros((n, 2 * K), dtype=np.float32)
    profiles[:, 0::2] = out_profiles
    profiles[:, 1::2] = in_profiles

    return profiles


def compute_reachability_features(A, K=10, weights=None):
    """Compute weighted reachability feature matrix.

    P = sum_{k=1}^{K} w_k * R^(k)*

    This generalizes AORM Eq.19 with configurable weights.

    Args:
        A: Adjacency matrix.
        K: Maximum reachability order.
        weights: Weight for each order. Default: w_k = 1/k.

    Returns:
        P: Feature matrix as scipy.sparse.csr_matrix.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.astype(np.float32)

    if weights is None:
        weights = {k: 1.0 / k for k in range(1, K + 1)}

    # P starts with w_1 * R^(1)* = w_1 * H(A)
    A_bool = A.copy()
    A_bool.data[:] = 1.0
    P = A_bool.multiply(weights.get(1, 1.0))

    iterator = SparseStormIterator(A, k=K)
    for Rk_star, power in iterator:
        w = weights.get(power, 1.0 / power)
        P = P + Rk_star.multiply(w)

    return P


# === PyTorch modules (optional, requires torch + torch_geometric) ===

def _check_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


if _check_torch():
    import torch
    import torch.nn as nn

    class StormPE(nn.Module):
        """STORM Positional Encoding module.

        Transforms precomputed reachability profiles into learned
        positional embeddings via MLP.

        Args:
            max_k: Maximum reachability order K.
            hidden_dim: Output embedding dimension.
            directed: Whether input profiles are bidirectional.
            num_layers: Number of MLP layers.
        """

        def __init__(self, max_k, hidden_dim, directed=False, num_layers=2):
            super().__init__()
            input_dim = max_k * 2 if directed else max_k

            layers = []
            dims = [input_dim] + [hidden_dim] * num_layers
            for i in range(num_layers):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.LayerNorm(dims[i + 1]))
            self.mlp = nn.Sequential(*layers)

        def forward(self, reachability_profile):
            """
            Args:
                reachability_profile: Tensor of shape (n_nodes, input_dim).
            Returns:
                Positional embedding of shape (n_nodes, hidden_dim).
            """
            return self.mlp(reachability_profile)

    class StormEnhancedGNN(nn.Module):
        """GNN enhanced with STORM Positional Encoding.

        Combines a base GNN (GIN convolutions) with STORM-PE
        residual connections at each layer.

        Args:
            in_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            out_dim: Output dimension (e.g., number of classes).
            max_k: Maximum reachability order for PE.
            n_layers: Number of GNN layers.
            directed: Whether reachability profiles are bidirectional.
            task: 'node' for node classification, 'graph' for graph classification.
        """

        def __init__(self, in_dim, hidden_dim, out_dim, max_k=10,
                     n_layers=4, directed=False, task='node'):
            super().__init__()
            self.task = task
            self.pe = StormPE(max_k, hidden_dim, directed)
            self.input_proj = nn.Linear(in_dim, hidden_dim)

            # Use try/except for optional torch_geometric dependency
            try:
                from torch_geometric.nn import GINConv, global_mean_pool
                self.has_pyg = True
                self.pool = global_mean_pool

                self.convs = nn.ModuleList()
                self.norms = nn.ModuleList()
                for _ in range(n_layers):
                    mlp = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                    self.convs.append(GINConv(mlp))
                    self.norms.append(nn.LayerNorm(hidden_dim))

            except ImportError:
                self.has_pyg = False
                self.convs = nn.ModuleList()
                self.norms = nn.ModuleList()
                for _ in range(n_layers):
                    self.convs.append(nn.Linear(hidden_dim, hidden_dim))
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.classifier = nn.Linear(hidden_dim, out_dim)

        def forward(self, x, edge_index, reach_profile, batch=None):
            """
            Args:
                x: Node features (n_nodes, in_dim).
                edge_index: Edge index tensor (2, n_edges).
                reach_profile: Reachability profile (n_nodes, K or 2K).
                batch: Batch vector for graph classification.

            Returns:
                Output logits.
            """
            pe = self.pe(reach_profile)
            h = self.input_proj(x) + pe

            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                if self.has_pyg:
                    h_new = conv(h, edge_index)
                else:
                    h_new = conv(h)
                h = norm(h_new + pe)  # Residual PE connection
                h = torch.relu(h)

            if self.task == 'graph' and batch is not None and self.has_pyg:
                h = self.pool(h, batch)

            return self.classifier(h)
