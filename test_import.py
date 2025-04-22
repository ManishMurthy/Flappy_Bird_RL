# test_imports.py
print("Testing imports...")
import sys
print(f"Using Python at: {sys.executable}")

import numpy
print(f"NumPy version: {numpy.__version__}")

import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")

import torch
print(f"PyTorch version: {torch.__version__}")

print("All imports successful!")