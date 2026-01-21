import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict, Optional
from collections import defaultdict
import warnings
import pickle
import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


current_dir = Path(__file__).resolve().parent

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print(f"Adding to path: {current_dir}")

from train.trainer import run_once
from config.schema import TrainConfig



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = TrainConfig(task_name="LSTM", policy_name="modified_loss", seed=0)
    cfg.ensure_dirs()
    
    run_once(cfg, device)

if __name__ == "__main__":
    main()
