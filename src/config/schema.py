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
    
    speaker: List[str] = field(default_factory=lambda: [
        "Agnes", "Allison", "Bruce", "Junior", "Princess", "Samantha",
        "Tom", "Victoria", "Alex", "Ava", "Fred", "Kathy", "Ralph",
        "Susan", "Vicki", "MALD",
    ])
    
    gammatone_dir: Path = Path("C:/Dataset/EARSHOT/EARSHOT-gammatone+/GAMMATONE_64_100/")
    neighbors: Path = Path("C:/Dataset/EARSHOT/EARSHOT-gammatone+/MALD-NEIGHBORS-1000.txt")
    
    runs_root: Path = Path("runs")

    def run_id(self) -> str:
        return f"{self.task_name}_{self.policy_name}_s{self.seed:03d}"
    
    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id()

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # self.trace_dir.mkdir(parents=True, exist_ok=True)
        # self.wm_dir.mkdir(parents=True, exist_ok=True)