# D-STORM: Dynamic Sparse Topology-Aware Optimal Reachability Matrix

**Small World, Small Efforts** — Fast all-pairs shortest paths for graphs, from Python BFS to custom CUDA kernels.

D-STORM extends the [AORM framework](https://ieeexplore.ieee.org/document/9424548) (IEEE Access, 2021) with sparse matrix redesign, Cython fused pruning, CUDA GPU acceleration, and GraphBLAS baseline comparison. This repository contains 12 APSP implementations benchmarked across 6 graph topologies.

**Project Website:** [https://dknife.github.io/STORM](https://dknife.github.io/STORM)

## Key Results (Facebook, n=4,039)

All 12 methods produce identical distance matrices (verified element-wise).

### GPU Methods

| # | Method | ID | Time (s) | vs SciPy | Category |
|---|--------|----|----------|----------|----------|
| 1 | **GPU-PerSrc-BFS** | BG1 | **0.016** | **122x** | Baseline |
| 2 | **DAWN-SOVM** | BG2 | 0.019 | 100x | Baseline |
| 3 | **D-STORM-CUDA** (guard+CAS) | TG2 | 0.028 | 68x | D-STORM |
| 4 | **D-STORM-DAWN** (bitwise) | TG1 | 0.247 | 7.8x | D-STORM |

### CPU Methods

| # | Method | ID | Time (s) | vs SciPy | Category |
|---|--------|----|----------|----------|----------|
| 1 | **D-STORM-SpMM-Cython** | TC1 | **0.984** | **2.0x** | D-STORM |
| 2 | D-STORM-NumpyBLAS | TC2 | 1.697 | 1.1x | D-STORM |
| 3 | D-STORM-GraphBLAS | TC3 | 1.921 | 1.0x | D-STORM |
| 4 | SciPy (C BFS) | BC2 | 1.918 | 1.0x | Baseline |
| 5 | M-AORM | BC4 | 2.831 | 0.7x | Baseline |
| 6 | GB-bfs | BC5 | 4.071 | 0.5x | Baseline |
| 7 | I-AORM | BC3 | 8.693 | 0.2x | Baseline |
| 8 | NetworkX | BC1 | 14.073 | 0.1x | Baseline |

### Performance Tiers

```
Tier 1  BG1/BG2/TG2  (0.005–0.03s)  GPU BFS / CUDA direct expand / DAWN-SOVM
Tier 2  TG1          (0.01–0.25s)   GPU bitwise frontier-sharing D-STORM
Tier 3  TC1/BC2      (0.16–1.92s)   CPU SpMM+Cython / C BFS
Tier 4  TC2/TC3      (0.30–3.63s)   CPU dense BLAS / GraphBLAS
Tier 5  BC1–BC5      (0.39–14.1s)   Python BFS / edge-wise / GraphBLAS BFS
```

### Scalability (BA graphs, GPU methods)

| n | BG1 | BG2 | TG1 | TG2 |
|---:|----:|----:|----:|----:|
| 1,000 | 0.001 | 0.001 | 0.006 | 0.002 |
| 4,000 | 0.011 | 0.010 | 0.052 | 0.019 |
| 10,000 | 0.067 | 0.072 | 0.238 | 0.211 |
| 15,000 | 0.156 | 0.167 | 0.487 | 0.454 |
| **20,000** | **0.263** | **0.281** | **0.805** | **12.784** |

TG2 hits a memory cliff at n>15K (6.4GB dense buffers). TG1's bitwise approach (75% memory reduction) scales smoothly to n=20K and beyond.

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
| BG2 | DAWN-SOVM | CUDA frontier-driven BFS (DAWN) | GPU |
| TG1 | D-STORM-DAWN | Bitwise frontier-sharing (iBFS + DAWN) | GPU |
| TG2 | D-STORM-CUDA | CUDA CSR direct expand (guard+CAS) | GPU |

## D-STORM Contributions

### 1. CPU: Fastest matrix-algebraic APSP

D-STORM-SpMM-Cython (TC1) is the **fastest CPU method across all 6 graph topologies**, outperforming SciPy's native C BFS by up to 2.0x. The key is Cython fused pruning — collapsing three sparse operations (booleanize → prune → footprint update) into a single C pass over COO entries.

### 2. GPU: Two D-STORM CUDA strategies

**TG2 (D-STORM-CUDA)** — Fastest D-STORM GPU at small-to-medium scale. Replaces cuSPARSE SpMM with direct CSR expansion using guard+CAS footprint. 6–18x over previous GPU-Sparse. Limited to n<15K by dense n×n buffers.

**TG1 (D-STORM-DAWN)** — Scalable D-STORM GPU. Combines DAWN's frontier-driven expansion with iBFS's bitwise packing (32 sources per uint32 word). 75% memory reduction vs TG2, enabling n=20K+ operation. Frontier sharing: vertex j's neighbors traversed once for all 32 sources via bitwise OR.

| | TG2 (guard+CAS) | TG1 (bitwise) |
|---|---|---|
| n=4K | **0.019s** (faster) | 0.052s |
| n=20K | 12.8s (memory cliff) | **0.805s** (stable) |
| Memory | 16n² bytes | 4n² + batch buffers |
| Scalability limit | ~15K | **VRAM-limited only** |

### 3. Structural insight: D-STORM optimizes toward per-source BFS

Progressive removal of D-STORM's matrix-algebraic overhead converges to per-source BFS:

```
GPU-Sparse (cuSPARSE SpMM)     0.192s  ──  1.0x
  └─ Direct CSR expand (TG2)   0.028s  ──  6.9x  (remove SpMM + guard+CAS)
      └─ Per-source BFS (BG1)  0.016s  ── 12.0x  (remove all matrix overhead)
```

The remaining gap is the structural cost of D-STORM's matrix-algebraic framework: shared footprint requiring atomics, per-hop kernel launch, and batch management.

### 4. D-STORM's value beyond speed

While BG1/BG2 are fastest for full APSP, D-STORM provides unique capabilities:

- **Dynamic edge insertion** — O(n²) incremental update, 22–33x faster than full recomputation
- **Algebraic analysis** — matrix-based framework for theoretical convergence proofs
- **Hop shell structure** — explicit frontier matrices at each distance level

## Quick Start

### Run Full Benchmark (12 methods × 6 graphs)

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
02_Implementations/           # 12 APSP implementations
├── BC1_NetworkX/             # CPU baselines
├── BC2_SciPy/
├── BC3_IAORM/
├── BC4_MAORM/
├── BC5_GB_bfs/
├── TC1_DSTORM_SpMM_Cython/   # D-STORM CPU variants
├── TC2_DSTORM_NumpyBLAS/
├── TC3_DSTORM_GraphBLAS/
├── BG1_GPU_PerSrc_BFS/       # GPU baselines
├── BG2_DAWN/
├── TG1_DSTORM_DAWN/          # D-STORM GPU variants
├── TG2_DSTORM_CUDA/
├── common/                   # Shared utilities
├── run_full_benchmark.py     # Full benchmark script
└── run_tc3_standalone.py     # TC3 standalone

03_Experiments/               # Benchmark reports
04_Datasets/                  # Graph datasets
docs/                         # GitHub Pages website
```

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
- Y. Feng et al., "DAWN: Matrix Operation-Optimized Algorithm for Shortest Paths Problem on Unweighted Graphs," *ACM ICS*, 2024.
- H. Liu, H. H. Huang, "iBFS: Concurrent Breadth-First Search on GPUs," *ACM SIGMOD*, 2016.
