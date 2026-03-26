"""
Set BLAS threading env before NumPy initializes OpenBLAS.

Mitigates OpenBLAS warnings about OpenMP parallel regions (nested threading).
Uses setdefault so a pre-set environment wins.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
