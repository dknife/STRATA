# STRATA: STratified Reachability And Topology Algebra

**Small World, Small Efforts** — Fast all-pairs shortest paths for graphs, from Python BFS to custom CUDA kernels.

STRATA extends the [AORM framework](https://ieeexplore.ieee.org/document/9424548) (IEEE Access, 2021) with sparse matrix redesign, Cython fused pruning, CUDA GPU acceleration, and GraphBLAS baseline comparison. This repository contains 12 APSP implementations benchmarked across 6 graph topologies.

**Project Website:** [https://dknife.github.io/STRATA](https://dknife.github.io/STRATA)

## Key Results (Facebook, n=4,039)

All 12 methods produce identical distance matrices (verified element-wise).

### GPU Methods

| # | Method | ID | Time (s) | vs SciPy | Category |
|---|--------|----|----------|----------|----------|
| 1 | **GPU-PerSrc-BFS** | BG1 | **0.016** | **122x** | Baseline |
| 2 | **DAWN-SOVM** | BG2 | 0.019 | 100x | Baseline |
| 3 | **STRATA-CUDA** (guard+CAS) | TG2 | 0.028 | 68x | STRATA |
| 4 | **STRATA-DAWNiBFS** (bitwise) | TG1 | 0.247 | 7.8x | STRATA |

### CPU Methods

| # | Method | ID | Time (s) | vs SciPy | Category |
|---|--------|----|----------|----------|----------|
| 1 | **STRATA-SpMM-Cython** | TC1 | **0.984** | **2.0x** | STRATA |
| 2 | STRATA-NumpyBLAS | TC2 | 1.697 | 1.1x | STRATA |
| 3 | STRATA-GraphBLAS | TC3 | 1.921 | 1.0x | STRATA |
| 4 | SciPy (C BFS) | BC2 | 1.918 | 1.0x | Baseline |
| 5 | M-AORM | BC4 | 2.831 | 0.7x | Baseline |
| 6 | GB-bfs | BC5 | 4.071 | 0.5x | Baseline |
| 7 | I-AORM | BC3 | 8.693 | 0.2x | Baseline |
| 8 | NetworkX | BC1 | 14.073 | 0.1x | Baseline |

### Performance Tiers

```
Tier 1  BG1/BG2/TG2  (0.005–0.03s)  GPU BFS / CUDA direct expand / DAWN-SOVM
Tier 2  TG1          (0.01–0.25s)   GPU bitwise frontier-sharing STRATA
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
| TC1 | STRATA-SpMM-Cython | SciPy SpMM + Cython fused prune | CPU |
| TC2 | STRATA-NumpyBLAS | NumPy BLAS matmul | CPU |
| TC3 | STRATA-GraphBLAS | GrB_mxm + complement mask | CPU |
| BG1 | GPU-PerSrc-BFS | CUDA block-per-source BFS | GPU |
| BG2 | DAWN-SOVM | CUDA frontier-driven BFS (DAWN) | GPU |
| TG1 | STRATA-DAWNiBFS | Bitwise frontier-sharing (iBFS + DAWN) | GPU |
| TG2 | STRATA-CUDA | CUDA CSR direct expand (guard+CAS) | GPU |

## Research Contributions

This project is not a claim that STRATA is the fastest APSP method. Rather, it is a **systematic investigation of the limits and merits of matrix-algebraic APSP**, conducted through 12 implementations on a common platform.

### 1. Matrix-algebraic APSP can beat native C BFS on CPU

STRATA-SpMM-Cython (TC1) outperforms SciPy's native C BFS across **all 6 graph topologies** (up to 2.0x). This is a non-obvious result: a matrix-algebraic approach with Python orchestration beats a hand-optimized C loop. The key is Cython fused pruning — collapsing three sparse operations (booleanize → prune → footprint update) into a single C pass over COO entries.

| Graph | TC1 (s) | SciPy (s) | Speedup |
|-------|---------|-----------|---------|
| Facebook (n=4,039) | 0.984 | 1.918 | 2.0x |
| Grid-45×45 (d=88) | 0.159 | 0.161 | 1.0x |

### 2. On GPU, matrix-algebraic APSP converges to per-source BFS

Progressive optimization of STRATA's GPU implementation reveals a structural convergence:

```
cuSPARSE SpMM (GPU-Sparse)     0.192s  ──  1.0x    matrix algebra
  └─ CSR direct expand (TG2)   0.028s  ──  6.9x    remove SpMM
      └─ Bitwise sharing (TG1) 0.247s  ──  0.8x    add iBFS packing
          └─ Per-source BFS     0.016s  ── 12.0x    remove all framework
```

Each optimization step removes matrix-algebraic indirection: SpMM → direct CSR traversal → frontier sharing → independent BFS. The **remaining 2.7x gap between TG1 and BG1** is the irreducible cost of STRATA's framework: shared footprint (atomicOr), per-hop synchronization, and batch management. This is a structural limit, not an engineering gap.

### 3. Systematic 12-method benchmark on common platform

All 12 methods — spanning Python BFS, C BFS, AORM (edge-wise and matmul), STRATA (3 CPU + 2 GPU variants), GraphBLAS (BFS and STRATA), GPU per-source BFS, and DAWN — are benchmarked under identical conditions (same hardware, same graphs, same correctness verification). This enables fair cross-paradigm comparison that is difficult to find in existing literature.

### 4. STRATA's unique capabilities remain unmatched

Per-source BFS is faster for full APSP, but STRATA provides capabilities that BFS cannot:

- **Dynamic edge insertion** — When edge (a,b) is added: D'(i,j) = min(D(i,j), D(i,a)+1+D(b,j)). O(n²) update vs full O(n·m) recomputation, yielding 22–33x speedup. No BFS-based method can do this without recomputing from scratch.
- **Algebraic convergence analysis** — The matrix framework enables theoretical proofs about iteration count, convergence rate, and relationship to graph diameter.

### 5. GPU STRATA scalability via bitwise frontier sharing

TG1 (STRATA-DAWNiBFS) demonstrates that combining iBFS's bitwise packing with DAWN's frontier-driven expansion reduces STRATA's memory from 16n² to ~4.25n² bytes, enabling operation at n=20K+ where TG2 fails. While slower than pure BFS (2.7x), this is the fastest known STRATA GPU implementation that maintains the algebraic framework at scale.

| | TG2 (guard+CAS) | TG1 (bitwise) | BG1 (pure BFS) |
|---|---|---|---|
| n=4K | **0.019s** | 0.052s | 0.011s |
| n=20K | 12.8s (OOM) | 0.805s | **0.263s** |
| Framework | STRATA | STRATA | None |
| Dynamic update | Yes | Yes | **No** |

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
├── TC1_STRATA_SpMM_Cython/   # STRATA CPU variants
├── TC2_STRATA_NumpyBLAS/
├── TC3_STRATA_GraphBLAS/
├── BG1_GPU_PerSrc_BFS/       # GPU baselines
├── BG2_DAWN/
├── TG1_STRATA_DAWNiBFS/          # STRATA GPU variants
├── TG2_STRATA_CUDA/
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
