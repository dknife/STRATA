# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

D-STORM (Dynamic Sparse Topology-aware Optimal Reachability Matrix) is a research framework for computing all-pairs shortest paths (APSP) on graphs. It extends the AORM framework (IEEE Access, 2021) with sparse matrix redesign, Cython fused pruning, CUDA GPU acceleration, and GraphBLAS baseline comparison.

**GitHub:** https://github.com/dknife/STORM
**Pages:** https://dknife.github.io/STORM

## Directory Structure

- **AORM-main/** — Original AORM implementation (2021). Dense numpy matrices. Reference only.
- **02_STORM_Implement/** — CPU D-STORM. Cython fused pruning kernel + NumPy fallback.
- **02_STORM_GPU_Implement/** — GPU D-STORM. cuBLAS (GPU-Dense) and cuSPARSE (GPU-Sparse).
- **03_Baseline_GraphBLAS/** — SuiteSparse:GraphBLAS APSP baseline via CFFI.
- **03_experiment_result/** — Benchmark scripts, JSON data, LaTeX reports (종합보고.tex).
- **03_a_experimental_result/** — Grid approximation visualizations (PNG).
- **05_Integrated/** — Extended paper (storm_extended.tex, IEEEtran format).
- **06_FinalPaper/** — ACM sigconf paper (main.tex, D-STORM submission draft).
- **docs/** — GitHub Pages website (index.html, howto.html, references.html).

## Running

### CPU D-STORM
```bash
cd 02_STORM_Implement
pip install -r requirements.txt

# Full APSP on edge-list graph
python main.py -m storm -i ../AORM-main/datasets/real-world/facebook_combined.txt -r

# Hop-constrained (k=5)
python main.py -m storm -i graph.txt -r -k 5

# Compare all CPU methods
python main.py -m compare -i graph.txt -r
```

### GPU D-STORM
```bash
cd 02_STORM_GPU_Implement
pip install -r requirements.txt

# GPU sparse (recommended for n >= 2000)
python main.py -m sparse -i graph.txt -r

# GPU dense (fastest for n < 2000)
python main.py -m dense -i graph.txt -r

# Compare GPU vs CPU
python main.py -m compare -i graph.txt -r
```

### GraphBLAS Baseline
```bash
cd 03_Baseline_GraphBLAS
pip install suitesparse-graphblas -r requirements.txt
python main.py -m compare -i graph.txt -r
```

### Full Benchmark (10 methods x 10 datasets)
```bash
cd 03_experiment_result
python run_full_benchmark.py
```

## Architecture

### CPU D-STORM (02_STORM_Implement/storm/)
- **core.py** — `SparseStormIterator`: scipy CSR SpMM + Cython fused pruning (or NumPy vectorized fallback). Dense Boolean footprint for O(1) lookup.
- **_storm_core.pyx** — Cython C-extension: single-pass COO scan for prune + footprint update.
- **apsp.py** — `storm_apsp()` (sparse), `storm_apsp_dense()` (BLAS matmul).
- **dynamic.py** — D-STORM edge insertion: D'(i,j) = min(D(i,j), D(i,a)+1+D(b,j)).
- **loader.py** — Multi-format graph loader (edgelist, gpickle, npz).

### GPU D-STORM (02_STORM_GPU_Implement/storm_gpu/)
- **cuda_env.py** — Auto-detects CUDA_PATH from pip-installed nvidia packages.
- **core.py** — `GpuSparseStormIterator` (cuSPARSE), `GpuDenseStormIterator` (cuBLAS).
- **apsp.py** — `gpu_storm_apsp()`, `gpu_storm_apsp_dense()`.

### GraphBLAS (03_Baseline_GraphBLAS/)
- **graphblas_apsp.py** — `graphblas_bfs_apsp()` (per-source GrB_vxm), `graphblas_frontier_apsp()` (GrB_mxm with LOR_LAND semiring + complement mask).

## Cython Build Notes

- macOS: `python setup.py build_ext --inplace`
- Windows: setuptools may fail with Korean/Unicode paths. Use `build_cython.bat` or manual `cl`+`link` build via vcvarsall.bat. See README.md for details.
- Without Cython: `SparseStormIterator` auto-falls back to NumPy vectorized path (~27% slower).

## Key Dependencies

Python 3.14+, numpy, scipy, networkx, tqdm. Optional: cupy-cuda12x (GPU), suitesparse-graphblas (GraphBLAS), cython (build).

## Benchmarking Data

- `03_experiment_result/full_benchmark_results.json` — 100 measurement points (10 methods x 10 datasets).
- `03_experiment_result/storm_results.json` — Per-experiment structured data.
- All timings: 3-run minimum, GPU with 1 warmup run.
