"""
STORM: Scalable Topological Order Reachability Matrix

A scalable framework for arbitrary-order reachability computation
on massive graphs, with GPU acceleration and GNN integration.

Modules:
    core    - Sparse AORM iterators (CPU)
    gpu     - GPU-accelerated STORM via CuPy
    apsp    - All-pairs shortest path algorithms
    pe      - STORM-PE: Positional Encoding for GNNs
    dynamic - Dynamic STORM for evolving graphs
    loader  - Graph loading and preprocessing
"""

__version__ = "0.1.0"

from storm.core import SparseStormIterator, storm_reachability
from storm.apsp import storm_apsp, storm_apsp_constrained
from storm.loader import load_graph
