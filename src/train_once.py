# train_once.py
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict, Optional
from collections import defaultdict
import warnings
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


current_dir = Path(__file__).resolve().parent

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print(f"Adding to path: {current_dir}")


from config.schema import TrainConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cfg = TrainConfig(dataset="burgundy", task_name="LSTM", policy_name="modified_loss", seed=0)
    # cfg = TrainConfig(dataset="librispeech", task_name="LSTM_WORD", policy_name="baseline", seed=0)
    cfg = TrainConfig(dataset="librispeech", task_name="LAS_MOCHA_WORDS", policy_name="semantic", seed=0)
    cfg.ensure_dirs()
    

    if cfg.dataset == "librispeech":
        if cfg.task_name == "LSTM_WORD":
            from train.trainer_librispeech_words import run_once
            run_once(cfg, device)
        if cfg.task_name == "LSTM":
            from train.trainer_librispeech import run_once
            run_once(cfg, device)
        
        if cfg.task_name == "LAS_MOCHA_WORDS":
            cfg.batch_size = 4
            cfg.epoch = 50
            cfg.lstm_layers = 3
            cfg.lstm_hidden = 512
            cfg.las_enc_subsample = 2
            cfg.las_max_words = 200
            from train.trainer_librispeech_mocha_words import run_once
            run_once(cfg, device)
        
        if cfg.task_name == "RNNT":
            from train.trainer_rnnt import run_once
            run_once(cfg, device)

    elif cfg.dataset == "burgundy":
        from train.trainer_burgundy import run_once
        run_once(cfg, device)
    
if __name__ == "__main__":
    main()
