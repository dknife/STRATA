# STORM: Sparse Topology-Aware Optimal Reachability Matrix

**Small World, Small Efforts** — Fast incremental all-pairs shortest paths for small-world networks.

STORM extends the [AORM framework](https://ieeexplore.ieee.org/document/9424548) (IEEE Access, 2021) with sparse matrix redesign, a Cython C-extension for fused pruning, CUDA GPU acceleration, and GraphBLAS baseline comparison.

**Project Website:** [https://dknife.github.io/STORM](https://dknife.github.io/STORM)

## Key Results (Facebook, n=4,039)

| Method | Time (s) | vs SciPy (C) | vs NetworkX |
|--------|----------|-------------|-------------|
| **GPU-Dense** (cuBLAS) | **0.114** | **17.9x** | **128.8x** |
| **GPU-Sparse** (cuSPARSE) | 0.175 | 11.7x | 83.9x |
| **D-STORM-Sparse** (Cython) | 1.093 | **1.87x** | 13.4x |
| D-STORM-Dense | 1.816 | 1.13x | 8.1x |
| GraphBLAS-frontier | 1.997 | 1.02x | 7.3x |
| SciPy (C Dijkstra) | 2.045 | 1.0x | 7.2x |
| I-AORM (original) | 8.070 | 0.25x | 1.8x |
| NetworkX | 14.683 | 0.14x | 1.0x |

- **Hop-constrained (k=5):** 18.3x over NetworkX at 89% accuracy
- **Dynamic edge addition:** 22-33x over full recomputation
- **GPU speedup scales with size:** 14x (n=500) to 16.4x (n=5,000)

## Implementations

| Directory | Description | Kernel |
|-----------|-------------|--------|
| `02_STORM_Implement/` | CPU D-STORM with Cython fused pruning | scipy SpMM + Cython C |
| `02_STORM_GPU_Implement/` | GPU D-STORM via CUDA | cuBLAS / cuSPARSE (CuPy) |
| `03_Baseline_GraphBLAS/` | GraphBLAS APSP baseline | SuiteSparse:GraphBLAS (CFFI) |
| `03_experiment_result/` | Benchmark scripts, JSON data, LaTeX reports | -- |
| `05_Integrated/` | Extended paper with all results | -- |

## Build & Run

### CPU D-STORM (Cython)

```bash
cd 02_STORM_Implement
pip install -r requirements.txt
```

**Build Cython extension:**

On macOS/Linux:
```bash
python setup.py build_ext --inplace
```

On Windows (requires [VS Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)):
```bat
:: Option 1: setuptools (if no Unicode path issues)
python setup.py build_ext --inplace

:: Option 2: Manual build (if setuptools fails due to Korean/Unicode paths)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
python -m cython storm/_storm_core.pyx
cl /c /O2 /MD /I"%PYTHON_INCLUDE%" /I"%NUMPY_INCLUDE%" /Tc storm\_storm_core.c /Fo:storm\_storm_core.obj
link /DLL /LIBPATH:"%PYTHON_LIBS%" python3XX.lib storm\_storm_core.obj /OUT:storm\_storm_core.cpXXX-win_amd64.pyd
```

Find your paths with:
```bash
python -c "import sysconfig, numpy; print('INCLUDE:', sysconfig.get_path('include')); print('LIBS:', sysconfig.get_config_var('installed_base')+'/libs'); print('NUMPY:', numpy.get_include())"
```

**Without Cython:** D-STORM automatically falls back to a NumPy vectorized path. Performance is ~27% slower than Cython but still competitive with SciPy.

**Run:**
```bash
# Full APSP
python main.py -m storm -i graph.txt -r

# Hop-constrained (k=5)
python main.py -m storm -i graph.txt -r -k 5

# Compare all CPU methods
python main.py -m compare -i graph.txt -r
```

### GPU D-STORM (CUDA)

Requires NVIDIA GPU with CUDA 12.x.

```bash
cd 02_STORM_GPU_Implement
pip install -r requirements.txt
```

If CuPy cannot find CUDA, install the runtime packages:
```bash
pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
```

The `storm_gpu/cuda_env.py` module automatically detects `CUDA_PATH` from pip-installed nvidia packages.

**Run:**
```bash
# GPU sparse (cuSPARSE, recommended for n >= 2000)
python main.py -m sparse -i graph.txt -r

# GPU dense (cuBLAS, fastest for n < 2000)
python main.py -m dense -i graph.txt -r

# Compare GPU vs CPU vs SciPy vs NetworkX
python main.py -m compare -i graph.txt -r
```

### GraphBLAS Baseline

```bash
cd 03_Baseline_GraphBLAS
pip install suitesparse-graphblas
pip install -r requirements.txt

# Compare GraphBLAS vs D-STORM vs SciPy vs NetworkX
python main.py -m compare -i graph.txt -r
```

### Full Benchmark (10 methods x 10 datasets)

```bash
cd 03_experiment_result
python run_full_benchmark.py
```

Results are saved to `full_benchmark_results.json`.

## Project Structure

```
02_STORM_Implement/
├── main.py                  # CPU CLI entry point
├── setup.py                 # Cython build config
├── build_cython.bat         # Windows manual build script
└── storm/
    ├── core.py              # SparseStormIterator (Cython + NumPy fallback)
    ├── _storm_core.pyx      # Cython fused pruning kernel
    ├── apsp.py              # APSP functions
    ├── dynamic.py           # D-STORM edge insertion
    ├── pe.py                # Reachability profile encoding
    ├── girth.py             # Graph girth
    └── loader.py            # Multi-format graph loader

02_STORM_GPU_Implement/
├── main.py                  # GPU CLI entry point
└── storm_gpu/
    ├── cuda_env.py          # CUDA path auto-detection
    ├── core.py              # GpuSparseStormIterator, GpuDenseStormIterator
    ├── apsp.py              # GPU APSP functions
    └── loader.py            # Graph loader

03_Baseline_GraphBLAS/
├── main.py                  # GraphBLAS CLI entry point
├── graphblas_apsp.py        # BFS + frontier APSP via GrB_mxm / GrB_vxm
└── loader.py                # Graph loader

03_experiment_result/
├── run_full_benchmark.py    # Full 10x10 benchmark script
├── full_benchmark_results.json
├── storm_results.json
└── 종합보고.tex              # Comprehensive report (Korean)
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
