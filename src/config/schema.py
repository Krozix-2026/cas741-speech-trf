# config/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainConfig:
    # ---- global ----
    dataset: str = "librispeech"   # "librispeech" | "burgundy"
    task_name: str = "LSTM"        # "LSTM" | "RNNT"
    policy_name: str = "modified_loss"

    seed: int = 0
    epoch: int = 100

    runs_root: Path = Path("runs")

    # ---- Burgundy / EARSHOT specific (optional until dataset decides) ----
    speaker: List[str] = field(default_factory=list)
    gammatone_dir: Optional[Path] = None
    neighbors: Optional[Path] = None

    # ---- LibriSpeech specific (optional until dataset decides) ----
    librispeech_root: Optional[Path] = None
    train_subset: Optional[str] = None
    valid_subset: Optional[str] = None
    test_subset: Optional[str] = None
    
    # ---- LibriSpeech word-aligned coch specific ----
    librispeech_manifest: Optional[Path] = Path(r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")
    coch_feat_dim: int = 64
    env_sr: int = 100
    tail_ms: int = 100
    word_vocab_topk: int = 20000

    # ---- LAS-Monotonic (semantic) params ----
    las_enc_subsample: int = 2         # causal time downsample by stacking frames
    las_attn_dim: int = 128
    las_dec_embed: int = 256
    las_dec_hidden: int = 512
    las_dec_layers: int = 2
    las_max_words: int = 200           # cap output words per utterance for speed/stability
    
    
    # ---- feature / model (shared defaults; can be overridden per task) ----
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0

    limit_train_samples: Optional[int] = None
    limit_valid_samples: Optional[int] = None
    limit_test_samples: Optional[int] = None

    dropout: float = 0.1

    # ---- LSTM params ----
    lstm_hidden: int = 512
    lstm_layers: int = 3
    bidirectional: bool = True

    # ---- RNNT params ----
    rnnt_enc_subsample: int = 4
    rnnt_pred_embed: int = 256
    rnnt_pred_hidden: int = 512
    rnnt_joint_dim: int = 512

    # ---- training ----
    batch_size: int = 32

    # ---- optim ----
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    num_workers: int = 0
    use_amp: bool = True

    # ---- debug / speed ----
    log_every: int = 20
    eval_every_epochs: int = 1

    def __post_init__(self) -> None:
        # normalize typos / casing
        ds = self.dataset.strip().lower()
        self.dataset = ds

        task = self.task_name.strip().upper()
        self.task_name = task

        # ---- dataset defaults ----
        if self.dataset == "burgundy":
            if not self.speaker:
                self.speaker = [
                    "Agnes", "Allison", "Bruce", "Junior", "Princess", "Samantha",
                    "Tom", "Victoria", "Alex", "Ava", "Fred", "Kathy", "Ralph",
                    "Susan", "Vicki", "MALD",
                ]
            if self.gammatone_dir is None:
                self.gammatone_dir = Path(r"C:\Dataset\EARSHOT\EARSHOT-gammatone+\GAMMATONE_64_100")
            if self.neighbors is None:
                self.neighbors = Path(r"C:\Dataset\EARSHOT\EARSHOT-gammatone+\MALD-NEIGHBORS-1000.txt")

            # 这里建议强校验：避免运行到一半才炸
            if not self.gammatone_dir.exists():
                raise FileNotFoundError(f"gammatone_dir not found: {self.gammatone_dir}")
            if not self.neighbors.exists():
                raise FileNotFoundError(f"neighbors not found: {self.neighbors}")

            self.epoch = 500
            
        elif self.dataset == "librispeech":
            if self.librispeech_root is None:
                self.librispeech_root = Path(r"C:\Dataset")
            if self.train_subset is None:
                self.train_subset = "train-clean-100"
            if self.valid_subset is None:
                self.valid_subset = "dev-clean"
            if self.test_subset is None:
                self.test_subset = "test-clean"
            if self.librispeech_manifest is None:
                self.librispeech_manifest = Path(r"C:\Dataset\manifest_librispeech_coch_align.jsonl")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset} (expected 'librispeech' or 'burgundy')")

        # ---- task defaults ----
        if self.task_name in ("LSTM", "LSTM_WORD"):
            # 你原来 LSTM 的 batch_size=32
            if self.batch_size is None:
                self.batch_size = 32
            # LSTM 的 bidirectional 默认 True 已经设了

        elif self.task_name == "LAS_MOCHA_WORDS":
            # Monotonic-attention LAS baseline (strictly causal encoder + monotonic attention)
            self.bidirectional = False
            # this model is heavier than frame-classification; safer default
            self.batch_size = min(self.batch_size, 8)
        
        
        elif self.task_name == "RNNT":
            # 你原来 RNNT 的默认设置
            self.lstm_hidden = 256
            self.lstm_layers = 1
            self.bidirectional = False
            self.batch_size = 1

        else:
            raise ValueError(f"Unknown task_name: {self.task_name} (expected 'LSTM' or 'RNNT')")

    def run_id(self) -> str:
        return f"{self.dataset}_{self.task_name}_{self.policy_name}_s{self.seed:03d}"

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
