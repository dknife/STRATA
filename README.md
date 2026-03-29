# STORM: Sparse Topology-Aware Optimal Reachability Matrix

**Small World, Small Efforts** — Fast incremental all-pairs shortest paths for small-world networks.

STORM extends the [AORM framework](https://ieeexplore.ieee.org/document/9424548) with sparse matrix redesign, a Cython C-extension for fused pruning, and dynamic edge support.

## Key Results

| Metric | STORM | vs I-AORM | vs SciPy (C) | vs NetworkX |
|--------|-------|-----------|--------------|-------------|
| Facebook APSP | 1.16s | **3.0x** | **2.6x** | **16.5x** |
| Hop-constrained k=5 | 0.96s | 1.9x | — | **18.3x** |
| Dynamic edge add | 8.5ms | — | — | **33x** vs recomp |

## Installation

```bash
cd Implementation
pip install -r requirements.txt
python setup.py build_ext --inplace  # Compile Cython extension
```

## Usage

```bash
# Sparse STORM APSP
python main.py -m storm -i graph.txt -r

# Hop-constrained (k=5)
python main.py -m storm -i graph.txt -r -k 5

# Compare with NetworkX
python main.py -m compare -i graph.txt -r
```

## Project Structure

```
Implementation/
├── main.py                 # CLI entry point
├── setup.py                # Cython build configuration
├── requirements.txt        # Dependencies
└── storm/
    ├── core.py             # SparseStormIterator, DenseStormIterator
    ├── core_optimized.py   # Optimized iterator with dense boolean footprint
    ├── core_cython.py      # Cython-accelerated APSP wrapper
    ├── _storm_core.pyx     # Cython C-extension (fused pruning kernel)
    ├── apsp.py             # APSP algorithms
    ├── gpu.py              # GPU acceleration (CuPy)
    ├── pe.py               # STORM-PE positional encoding
    ├── dynamic.py          # D-STORM dynamic edge updates
    ├── girth.py            # Graph girth computation
    └── loader.py           # Multi-format graph loader
```

## References

- S.-S. Kim, Y.-K. Kim, Y.-M. Kang, "AORM: Fast Incremental Arbitrary-Order Reachability Matrix Computation for Massive Graphs," *IEEE Access*, vol. 9, 2021.
- S.-S. Kim, Y.-M. Kang, Y.-K. Kim, "Sparsity-Aware Reachability Computation for Massive Graphs," *IEEE BigComp*, 2022.
