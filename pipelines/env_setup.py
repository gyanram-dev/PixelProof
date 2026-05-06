import os
import sys
import logging

def setup_environment():
    """
    Sets up the environment for PixelProof, specifically addressing common 
    Windows/PyTorch issues like OpenMP conflicts.
    """
    # 1. Address OpenMP conflict (libiomp5md.dll)
    # This is common on Windows when multiple libraries (like NumPy/MKL and Torch) 
    # bundle their own OpenMP runtimes.
    if sys.platform == "win32":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        # 2. Enable ANSI escape sequences for Windows console
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    # 3. Basic diagnostic info
    logging.info(f"Platform: {sys.platform}")
    logging.info(f"Python: {sys.version.split()[0]}")

def verify_ml_stack():
    """
    Verifies that the ML stack (torch, torchvision) is importable.
    """
    try:
        import torch
        import torchvision
        logging.info(f"Torch version: {torch.__version__}")
        logging.info(f"Torchvision version: {torchvision.__version__}")
        return True
    except ImportError as e:
        logging.error(f"ML Stack verification failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during ML stack verification: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_environment()
    if verify_ml_stack():
        print("Environment setup and verification successful.")
    else:
        print("Environment verification failed.")
        sys.exit(1)
