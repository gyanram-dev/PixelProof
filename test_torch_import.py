import time
import sys
import os

# Fix for OMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Checking environment...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

try:
    import torch
    print(f"Successfully imported torch version: {torch.__version__}")
except Exception as e:
    print(f"Failed to import torch: {e}")

print("Done.")
