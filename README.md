# D-STORM: Dynamic Sparse Topology-Aware Optimal Reachability Matrix

**Small World, Small Efforts** — Fast all-pairs shortest paths for graphs, from Python BFS to custom CUDA kernels.

D-STORM extends the [AORM framework](https://ieeexplore.ieee.org/document/9424548) (IEEE Access, 2021) with sparse matrix redesign, Cython fused pruning, CUDA GPU acceleration, and GraphBLAS baseline comparison. This repository contains 11 APSP implementations benchmarked across 6 graph topologies.

**Project Website:** [https://dknife.github.io/STORM](https://dknife.github.io/STORM)

## Key Results (Facebook, n=4,039)

All 11 methods produce identical distance matrices (verified element-wise).

### GPU Methods

| # | Method | ID | Time (s) | vs SciPy | Category |
|---|--------|----|----------|----------|----------|
| 1 | **GPU-PerSrc-BFS** | BG1 | **0.019** | **101x** | Baseline |
| 2 | **D-STORM-CUDA** (guard+CAS) | TG2 | 0.030 | 64x | D-STORM |
| 3 | D-STORM-cuBLAS | TG1 | 0.155 | 12x | D-STORM |

### CPU Methods

| # | Method | ID | Time (s) | vs SciPy | Category |
|---|--------|----|----------|----------|----------|
| 1 | **D-STORM-SpMM-Cython** | TC1 | **1.016** | **1.9x** | D-STORM |
| 2 | D-STORM-NumpyBLAS | TC2 | 1.723 | 1.1x | D-STORM |
| 3 | D-STORM-GraphBLAS | TC3 | 1.897 | 1.0x | D-STORM |
| 4 | SciPy (C BFS) | BC2 | 1.921 | 1.0x | Baseline |
| 5 | M-AORM | BC4 | 2.868 | 0.7x | Baseline |
| 6 | GB-bfs | BC5 | 4.065 | 0.5x | Baseline |
| 7 | I-AORM | BC3 | 7.489 | 0.3x | Baseline |
| 8 | NetworkX | BC1 | 14.213 | 0.1x | Baseline |

### Performance Tiers

```
Tier 1  BG1/TG2  (0.004–0.03s)  GPU per-source BFS / CUDA direct expand
Tier 2  TG1      (0.03–0.16s)   GPU cuBLAS dense matmul
Tier 3  TC1/BC2  (0.13–1.92s)   CPU SpMM+Cython / C BFS
Tier 4  TC2/TC3  (0.30–5.28s)   CPU dense BLAS / GraphBLAS
Tier 5  BC1–BC5  (0.40–14.2s)   Python BFS / edge-wise / GraphBLAS BFS
```

## D-STORM Contributions

### 1. CPU: Fastest matrix-algebraic APSP

D-STORM-SpMM-Cython (TC1) is the **fastest CPU method across all 6 graph topologies**, outperforming SciPy's native C BFS by up to 1.9x. The key is Cython fused pruning — collapsing three sparse operations (booleanize → prune → footprint update) into a single C pass over COO entries.

| Graph | TC1 (s) | SciPy (s) | Speedup |
|-------|---------|-----------|---------|
| Facebook (n=4,039) | 1.016 | 1.921 | 1.9x |
| BA-2000 | 0.284 | 0.312 | 1.1x |
| WS-2000 | 0.291 | 0.291 | 1.0x |
| Grid-45×45 (d=88) | 0.135 | 0.167 | 1.2x |

### 2. GPU: SpMM elimination via direct CSR expansion

D-STORM-CUDA (TG2) replaces cuSPARSE SpMM with a custom CUDA kernel that directly traverses CSR neighbors, eliminating three SpMM bottlenecks:

1. **Redundant products** — SpMM computes all matrix products including already-visited pairs
2. **Intermediate matrices** — SpMM output requires COO/CSR conversion overhead
3. **Separate prune step** — footprint check is fused into expansion (1-pass)

| Graph | GPU-Sparse (removed) | TG2 (s) | Improvement |
|-------|---------------------|---------|-------------|
| Facebook | 0.192 | 0.030 | 6.4x |
| Grid-45×45 | 0.236 | 0.013 | 18.2x |

### 3. Structural insight: D-STORM optimizes toward per-source BFS

Progressive removal of D-STORM's matrix-algebraic overhead converges to per-source BFS:

```
GPU-Sparse (cuSPARSE SpMM)     0.192s  ──  1.0x
  └─ Direct CSR expand          0.041s  ──  4.7x  (remove SpMM)
      └─ guard+CAS              0.030s  ──  6.4x  (reduce atomic contention)
          └─ Per-source BFS      0.019s  ── 10.1x  (remove all matrix overhead)
```

This reveals that for full APSP, the matrix-algebraic framework introduces indirection costs (shared footprint → atomics, per-hop kernel launch, frontier matrix management) that pure per-source BFS avoids entirely.

### 4. D-STORM's value beyond speed

While BG1 is fastest for full APSP, D-STORM provides capabilities that per-source BFS cannot:

- **k-hop constrained APSP** — exact hop shells at each distance level
- **Dynamic edge insertion** — O(n²) incremental update without full recomputation
- **Algebraic analysis** — matrix-based framework for theoretical convergence proofs

## Implementations (02_Implementations/)

| ID | Method | Kernel | Platform |
|----|--------|--------|----------|
| BC1 | NetworkX | Python BFS | CPU |
| BC2 | SciPy | C BFS | CPU |
| BC3 | I-AORM | edge-wise row sum | CPU |
| BC4 | M-AORM | dense BLAS matmul | CPU |
| BC5 | GB-bfs | GrB_vxm masked BFS | CPU |
| TC1 | D-STORM-SpMM-Cython | SciPy SpMM + Cython fused prune | CPU |
| TC2 | D-STORM-NumpyBLAS | NumPy BLAS matmul | CPU |
| TC3 | D-STORM-GraphBLAS | GrB_mxm + complement mask | CPU |
| BG1 | GPU-PerSrc-BFS | CUDA block-per-source BFS | GPU |
| TG1 | D-STORM-cuBLAS | cuBLAS dense matmul | GPU |
| TG2 | D-STORM-CUDA | CUDA CSR direct expand (guard+CAS) | GPU |

## Quick Start

### Run Full Benchmark (11 methods × 6 graphs)

```bash
cd 02_Implementations
pip install numpy scipy networkx cupy-cuda12x tqdm
python run_full_benchmark.py
```

TC3 (GraphBLAS) must run in a separate process due to `GrB_init` conflict with BC5:
```bash
pip install suitesparse-graphblas
python run_tc3_standalone.py
```

Results are saved to `full_benchmark_results.json`.

### Run Individual Methods

Each method in `02_Implementations/<ID>/apsp.py` exposes a `run_apsp(A_csr, k=-1, verbose=True)` function:

```python
import scipy.sparse as sp
from BC2_SciPy.apsp import run_apsp

A = sp.load_npz("graph.npz")  # or any scipy sparse matrix
D = run_apsp(A)                # returns int32 distance matrix
```

## Project Structure

```
02_Implementations/           # 11 APSP implementations
├── BC1_NetworkX/apsp.py
├── BC2_SciPy/apsp.py
├── BC3_IAORM/apsp.py
├── BC4_MAORM/apsp.py
├── BC5_GB_bfs/apsp.py
├── BG1_GPU_PerSrc_BFS/apsp.py
├── TC1_DSTORM_SpMM_Cython/apsp.py
├── TC2_DSTORM_NumpyBLAS/apsp.py
├── TC3_DSTORM_GraphBLAS/apsp.py
├── TG1_DSTORM_cuBLAS/apsp.py
├── TG2_DSTORM_CUDA/apsp.py
├── common/                   # Shared utilities
│   ├── cuda_env.py           # CUDA path auto-detection
│   └── loader.py             # Multi-format graph loader
├── run_full_benchmark.py     # Full benchmark script
└── run_tc3_standalone.py     # TC3 standalone (GraphBLAS init conflict)

03_Experiments/               # Benchmark reports
└── Comparison.tex            # 11×6 comparison report (Korean)

04_Datasets/                  # Graph datasets
├── real-world/               # Facebook social network (n=4,039)
├── simple/                   # Small test graphs
└── synthetic/                # Generated graphs

docs/                         # GitHub Pages website
```

## TG2 guard+CAS Kernel

D-STORM-CUDA (TG2) uses a custom CUDA kernel that directly expands frontier entries via CSR traversal, replacing cuSPARSE SpMM entirely. The footprint check uses a **guard+CAS** pattern:

```c
if (F[idx] == 0) {                      // non-atomic guard: skip visited cells
    if (atomicCAS(&F[idx], 0, 1) == 0) { // atomic CAS: race-free 0→1
        int pos = atomicAdd(out_count, 1);
        out_row[pos] = i;
        out_col[pos] = k;
    }
}
```

This achieves correctness (no duplicate frontier entries) with performance equal to non-atomic writes, and 38% faster than pure `atomicExch` on high-degree graphs.

## Benchmark Environment

- **OS:** Windows 11 Pro 10.0.26200
- **GPU:** NVIDIA GeForce RTX 5080 (16GB VRAM)
- **CUDA:** 12.9, CuPy 14.0.1
- **Python:** 3.14.3, NumPy 2.4.4, SciPy 1.17.1
- **GraphBLAS:** SuiteSparse 10.3.1 (CFFI)
- **Cython:** 3.2.4, MSVC 14.44 (VS Build Tools 2022)

## References

- S.-S. Kim, Y.-K. Kim, Y.-M. Kang, "AORM: Fast Incremental Arbitrary-Order Reachability Matrix Computation for Massive Graphs," *IEEE Access*, vol. 9, pp. 69539-69558, 2021.
- S.-S. Kim, Y.-M. Kang, Y.-K. Kim, "Sparsity-Aware Reachability Computation for Massive Graphs," *IEEE BigComp*, 2022.
