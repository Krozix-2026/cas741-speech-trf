# config/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class TrainConfig:
    task_name: str = "LSTM"
    policy_name: str = "modified_loss"
    
    seed: int = 0
    epoch: int  = 100
    
    runs_root: Path = Path("runs")
    
    ## ---- EarShot specific ----
    speaker: List[str] = field(default_factory=lambda: [
        "Agnes", "Allison", "Bruce", "Junior", "Princess", "Samantha",
        "Tom", "Victoria", "Alex", "Ava", "Fred", "Kathy", "Ralph",
        "Susan", "Vicki", "MALD",
    ])
    
    gammatone_dir: Path = Path("C:/Dataset/EARSHOT/EARSHOT-gammatone+/GAMMATONE_64_100/")
    neighbors: Path = Path("C:/Dataset/EARSHOT/EARSHOT-gammatone+/MALD-NEIGHBORS-1000.txt")
    
    # ---- LibriSpeech specific ----
    librispeech_root: Path = Path("C:\Dataset\LibriSpeech")
    train_subset: str = "train-clean-100"
    valid_subset: str = "dev-clean"
    test_subset: str = "test-clean"

    # ---- feature / model ----
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0

    lstm_hidden: int = 512
    lstm_layers: int = 3
    bidirectional: bool = True
    dropout: float = 0.1
    
    # ---- training ----
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    num_workers: int = 0
    use_amp: bool = True

    # ---- debug / speed ----
    limit_train_samples: Optional[int] = 2000
    limit_valid_samples: Optional[int] = 500
    limit_test_samples: Optional[int] = 500
    log_every: int = 20
    eval_every_epochs: int = 1

    def run_id(self) -> str:
        return f"{self.task_name}_{self.policy_name}_s{self.seed:03d}"
    
    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id()

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "ckpt"
    
    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"
    
    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)