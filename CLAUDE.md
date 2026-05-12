# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

STRATA (STratified Reachability And Topology Algebra) is a research framework for APSP on graphs. Hop-shells are stratified by distance; both σ-from-shell and level-based deletion operate on this stratification. 12 implementations (5 CPU baselines, 3 STRATA CPU, 2 GPU baselines, 2 STRATA GPU) benchmarked across 6 graph topologies + scaling analysis.

**GitHub:** https://github.com/dknife/STRATA
**Pages:** https://dknife.github.io/STRATA

## Directory Structure

- **02_Implementations/** — 12 APSP implementations (BC1-BC5, TC1-TC3, BG1-BG2, TG1-TG2) + common/ + benchmark scripts.
- **03_Experiments/** — Comparison.tex (12-method report, Korean, 12 pages).
- **04_Datasets/** — Facebook (real-world), simple, synthetic graphs.
- **05_Integrated/** — Extended paper (gitignored).
- **06_FinalPaper/** — ACM sigconf paper (gitignored).
- **Z_OldExperiments/** — Archived old implementations (gitignored).
- **docs/** — GitHub Pages website.

## Running

### Full Benchmark (12 methods x 6 graphs)
```bash
cd 02_Implementations
pip install numpy scipy networkx cupy-cuda12x tqdm
python run_full_benchmark.py

# TC3 (GraphBLAS) — separate process due to GrB_init conflict with BC5
pip install suitesparse-graphblas
python run_tc3_standalone.py
```

### Individual Method
```python
import sys; sys.path.insert(0, '02_Implementations')
from TC1_STRATA_SpMM_Cython.apsp import run_apsp
D = run_apsp(A_csr, k=-1, verbose=True)  # returns int32 distance matrix
```

## Implementations

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
| BG2 | DAWN-SOVM | CUDA frontier-driven BFS (DAWN ICS 2024) | GPU |
| TG1 | STRATA-DAWNiBFS | Bitwise frontier sharing (DAWN + iBFS SIGMOD 2016) | GPU |
| TG2 | STRATA-CUDA | CUDA CSR direct expand (guard+CAS) | GPU |

**Removed:** GPU-Fused, GPU-Sparse (cuSPARSE), STRATA-cuBLAS (O(n³)). Do not re-add.

## Key Dependencies

Python 3.14+, numpy, scipy, networkx, tqdm. Optional: cupy-cuda12x (GPU), suitesparse-graphblas (GraphBLAS).

## Key Conventions

- Distance matrix D: always dense int32. No sparse/float.
- STRATA naming: STRATA-{kernel}. External techniques credited in name (e.g., DAWNiBFS).
- Commits: no Co-Authored-By trailer.
- Reports: CPU results first, GPU second.
- Benchmark protocol: 3 runs min, GPU 1 warmup, all methods correctness-verified.

## Benchmarking Data

- `02_Implementations/full_benchmark_results.json` — 12 methods x 6 graphs.
- `02_Implementations/tc3_benchmark_results.json` — TC3 standalone results.
- `03_Experiments/Comparison.tex` — Full report with charts.
