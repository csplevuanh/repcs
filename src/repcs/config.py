import os
import torch

TOP_K = int(os.getenv("REPCS_TOP_K", 20))
KL_THRESHOLD = float(os.getenv("REPCS_KL_THRESHOLD", 0.05))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.getenv("REPCS_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
