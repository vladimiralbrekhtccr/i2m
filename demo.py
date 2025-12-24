import os
import sys
import torch
import pandas as pd
import transformers
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def main():
    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Example: access env var
    # secret = os.getenv("MY_SECRET")

if __name__ == "__main__":
    main()
