# config/schema.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, List, Optional


try:
    from .paths_local import DATA_ROOT as _DATA_ROOT
    from .paths_local import LIBRISPEECH_ROOT as _LIBRISPEECH_ROOT
    from .paths_local import LIBRISPEECH_ALIGNMENT_ROOT as _LIBRISPEECH_ALIGNMENT_ROOT
    from .paths_local import LIBRISPEECH_GAMMATONE_ROOT as _LIBRISPEECH_GAMMATONE_ROOT
    from .paths_local import LIBRISPEECH_MANIFEST_ROOT as _LIBRISPEECH_MANIFEST_ROOT
    from .paths_local import LIBRISPEECH_MANIFEST_PHONE_ROOT as _LIBRISPEECH_MANIFEST_PHONE_ROOT
    from .paths_local import EARSHOT_ROOT as _EARSHOT_ROOT
except Exception:
    _DATA_ROOT = None
    _LIBRISPEECH_ROOT = None
    _LIBRISPEECH_ALIGNMENT_ROOT = None
    _LIBRISPEECH_GAMMATONE_ROOT = None
    _LIBRISPEECH_MANIFEST_ROOT = None
    _LIBRISPEECH_MANIFEST_PHONE_ROOT = None
    _EARSHOT_ROOT = None


# =========================================================
# Enums
# =========================================================
class Dataset(StrEnum):
    LIBRISPEECH = "librispeech"
    BURGUNDY = "burgundy"


class TaskName(StrEnum):
    LSTM = "LSTM"
    LSTM_PHONE = "LSTM_PHONE"
    LSTM_WORD = "LSTM_WORD"
    LSTM_WORD_SEM = "LSTM_WORD_SEM"
    LAS_MOCHA_WORDS = "LAS_MOCHA_WORDS"
    LAS_WORDS = "LAS_WORDS"
    RNNT = "RNNT"


class LabelType(StrEnum):
    ONEHOT = "onehot"
    SRV = "srv"


class PolicyName(StrEnum):
    BASELINE = "baseline"
    MODIFIED_LOSS = "modified_loss"
    SEMANTIC = "semantic"
    SEMANTIC_HIER = "semantic_hier"
    LAS_GLOBAL = "las_global"


# =========================================================
# Helpers
# =========================================================
def _to_path(v: Optional[Path | str]) -> Optional[Path]:
    if v is None:
        return None
    if isinstance(v, Path):
        return v
    return Path(v)


def _coerce_enum(value: Any, enum_cls: type[StrEnum], field_name: str, *, normalize: str) -> StrEnum:
    if isinstance(value, enum_cls):
        return value

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str or {enum_cls.__name__}, got {type(value).__name__}")

    s = value.strip()
    if normalize == "lower":
        s = s.lower()
    elif normalize == "upper":
        s = s.upper()

    try:
        return enum_cls(s)
    except ValueError as e:
        valid = [x.value for x in enum_cls]
        raise ValueError(f"Invalid {field_name}={value!r}. Expected one of {valid}") from e


def _serialize_value(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, StrEnum):
        return v.value
    if isinstance(v, list):
        return [_serialize_value(x) for x in v]
    return v


# =========================================================
# Config
# =========================================================
@dataclass
class TrainConfig:
    # -----------------------------
    # global
    # -----------------------------
    dataset: Dataset | str = Dataset.LIBRISPEECH
    task_name: TaskName | str = TaskName.LSTM
    policy_name: PolicyName | str = PolicyName.MODIFIED_LOSS

    seed: int = 0
    epoch: int = 100

    runs_root: Path | str = Path("runs")
    check_paths: bool = True

    # -----------------------------
    # Burgundy
    # -----------------------------
    speaker: List[str] = field(default_factory=list)
    gammatone_dir: Optional[Path | str] = None
    neighbors: Optional[Path | str] = None

    # -----------------------------
    # LibriSpeech specific
    # -----------------------------
    librispeech_root: Optional[Path | str] = None
    librispeech_manifest: Optional[Path | str] = _LIBRISPEECH_MANIFEST_ROOT

    train_subset: Optional[str] = None
    valid_subset: Optional[str] = None
    test_subset: Optional[str] = None

    # label style
    label_type: LabelType | str = LabelType.ONEHOT

    # -----------------------------
    # SRV params
    # -----------------------------
    srv_dim: int = 2048
    srv_k: int = 16
    srv_value: str = "pm1"         # "pm1" | "binary"
    srv_seed: int = 123
    srv_loss: str = "cosine"       # "cosine" | "mse"
    srv_eval_nn: bool = False
    srv_eval_max_frames: int = 4000

    # -----------------------------
    # Phoneme supervision
    # -----------------------------
    phone_label_region: str = "center"   # "all" | "center" | "tail"
    phone_tail_ms: int = 30
    phone_center_ms: int = 30

    # -----------------------------
    # LibriSpeech word-aligned coch specific
    # -----------------------------
    coch_feat_dim: int = 64
    env_sr: int = 100
    tail_ms: int = 100
    word_vocab_topk: int = 20000

    # -----------------------------
    # Hierarchical semantic head
    # -----------------------------
    semantic_alpha: float = 1.0
    semantic_beta: float = 0.5

    word_rep_dim: int = 256
    word_lstm_hidden: int = 256
    word_lstm_layers: int = 1

    rep_dropout: float = 0.1
    word_dropout_p: float = 0.25
    rep_noise_std: float = 0.0

    semantic_shift: int = 1
    semantic_detach_frame: bool = False

    # -----------------------------
    # LAS / attention params
    # -----------------------------
    las_enc_subsample: int = 2
    las_attn_dim: int = 128
    las_dec_embed: int = 256
    las_dec_hidden: int = 512
    las_dec_layers: int = 2
    las_max_words: int = 200

    # -----------------------------
    # feature / model
    # -----------------------------
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0

    limit_train_samples: Optional[int] = None
    limit_valid_samples: Optional[int] = None
    limit_test_samples: Optional[int] = None

    dropout: float = 0.1

    # LSTM params
    lstm_hidden: int = 512
    lstm_layers: int = 3
    bidirectional: bool = False

    # RNNT params
    rnnt_enc_subsample: int = 4
    rnnt_pred_embed: int = 256
    rnnt_pred_hidden: int = 512
    rnnt_joint_dim: int = 512

    # training
    batch_size: int = 32

    # optim
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    num_workers: int = 0
    use_amp: bool = True

    # debug / speed
    log_every: int = 20
    eval_every_epochs: int = 1

    # =====================================================
    # lifecycle
    # =====================================================
    def __post_init__(self) -> None:
        self._normalize_types()
        self._apply_dataset_defaults()
        self._apply_task_defaults()
        self.validate()

    # =====================================================
    # normalize
    # =====================================================
    def _normalize_types(self) -> None:
        self.dataset = _coerce_enum(self.dataset, Dataset, "dataset", normalize="lower")
        self.task_name = _coerce_enum(self.task_name, TaskName, "task_name", normalize="upper")
        self.label_type = _coerce_enum(self.label_type, LabelType, "label_type", normalize="lower")
        self.policy_name = _coerce_enum(self.policy_name, PolicyName, "policy_name", normalize="lower")

        self.runs_root = _to_path(self.runs_root) or Path("runs")
        self.gammatone_dir = _to_path(self.gammatone_dir)
        self.neighbors = _to_path(self.neighbors)
        self.librispeech_root = _to_path(self.librispeech_root)
        self.librispeech_manifest = _to_path(self.librispeech_manifest)

        self.srv_value = self.srv_value.strip().lower()
        self.srv_loss = self.srv_loss.strip().lower()
        self.phone_label_region = self.phone_label_region.strip().lower()

    # =====================================================
    # defaults
    # =====================================================
    def _apply_dataset_defaults(self) -> None:
        if self.dataset == Dataset.BURGUNDY:
            if not self.speaker:
                self.speaker = [
                    "Agnes", "Allison", "Bruce", "Junior", "Princess", "Samantha",
                    "Tom", "Victoria", "Alex", "Ava", "Fred", "Kathy", "Ralph",
                    "Susan", "Vicki", "MALD",
                ]

            if self.gammatone_dir is None:
                self.gammatone_dir = Path(
                    r"C:\Dataset\EARSHOT\EARSHOT-gammatone+\GAMMATONE_64_100"
                )
            if self.neighbors is None:
                self.neighbors = Path(
                    r"C:\Dataset\EARSHOT\EARSHOT-gammatone+\MALD-NEIGHBORS-1000.txt"
                )

            # 只在用户没有主动改 epoch 时，才替换成 Burgundy 默认
            if self.epoch == 100:
                self.epoch = 500

        elif self.dataset == Dataset.LIBRISPEECH:
            if self.librispeech_root is None:
                self.librispeech_root = _LIBRISPEECH_ROOT or Path(r"C:\Dataset")

            if self.librispeech_manifest is None:
                self.librispeech_manifest = _LIBRISPEECH_MANIFEST_ROOT

            if self.train_subset is None:
                self.train_subset = "train-clean-100"
            if self.valid_subset is None:
                self.valid_subset = "dev-clean"
            if self.test_subset is None:
                self.test_subset = "test-clean"

    def _apply_task_defaults(self) -> None:
        if self.task_name in {
            TaskName.LSTM,
            TaskName.LSTM_WORD,
            TaskName.LSTM_WORD_SEM,
            TaskName.LSTM_PHONE,
        }:
            if self.task_name == TaskName.LSTM_PHONE and _LIBRISPEECH_MANIFEST_PHONE_ROOT is not None:
                self.librispeech_manifest = _LIBRISPEECH_MANIFEST_PHONE_ROOT

        elif self.task_name == TaskName.LAS_MOCHA_WORDS:
            self.bidirectional = False
            self.batch_size = min(self.batch_size, 8)

        elif self.task_name == TaskName.LAS_WORDS:
            self.bidirectional = True
            self.batch_size = min(self.batch_size, 16)

        elif self.task_name == TaskName.RNNT:
            # 只在仍是共享默认值时才自动改
            if self.lstm_hidden == 512:
                self.lstm_hidden = 256
            if self.lstm_layers == 3:
                self.lstm_layers = 1
            self.bidirectional = False
            if self.batch_size == 32:
                self.batch_size = 1

    # =====================================================
    # validation
    # =====================================================
    def validate(self) -> None:
        # ---------- basic ranges ----------
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.epoch <= 0:
            raise ValueError("epoch must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")

        if self.srv_dim <= 0:
            raise ValueError("srv_dim must be > 0")
        if self.srv_k <= 0:
            raise ValueError("srv_k must be > 0")
        if self.srv_eval_max_frames <= 0:
            raise ValueError("srv_eval_max_frames must be > 0")

        if self.phone_tail_ms <= 0:
            raise ValueError("phone_tail_ms must be > 0")
        if self.phone_center_ms <= 0:
            raise ValueError("phone_center_ms must be > 0")
        if self.tail_ms <= 0:
            raise ValueError("tail_ms must be > 0")

        if self.semantic_alpha < 0:
            raise ValueError("semantic_alpha must be >= 0")
        if self.semantic_beta < 0:
            raise ValueError("semantic_beta must be >= 0")
        if self.semantic_shift < 1:
            raise ValueError("semantic_shift must be >= 1")

        if self.sample_rate <= 0 or self.n_mels <= 0:
            raise ValueError("sample_rate and n_mels must be > 0")
        if self.win_length_ms <= 0 or self.hop_length_ms <= 0:
            raise ValueError("win_length_ms and hop_length_ms must be > 0")

        if self.lstm_hidden <= 0 or self.lstm_layers <= 0:
            raise ValueError("lstm_hidden and lstm_layers must be > 0")

        # ---------- categorical ----------
        if self.srv_value not in {"pm1", "binary"}:
            raise ValueError("srv_value must be 'pm1' or 'binary'")
        if self.srv_loss not in {"cosine", "mse"}:
            raise ValueError("srv_loss must be 'cosine' or 'mse'")
        if self.phone_label_region not in {"all", "center", "tail"}:
            raise ValueError("phone_label_region must be one of {'all', 'center', 'tail'}")

        # ---------- combination constraints ----------
        if self.dataset == Dataset.BURGUNDY and self.task_name != TaskName.LSTM:
            raise ValueError("dataset='burgundy' currently only supports task_name='LSTM'")

        if self.task_name == TaskName.LSTM_PHONE and self.label_type != LabelType.SRV:
            raise ValueError("LSTM_PHONE currently requires label_type='srv'")

        if self.task_name == TaskName.LSTM_WORD_SEM and self.label_type != LabelType.SRV:
            raise ValueError("LSTM_WORD_SEM requires label_type='srv'")

        if self.task_name == TaskName.LSTM_WORD and self.label_type not in {LabelType.ONEHOT, LabelType.SRV}:
            raise ValueError("LSTM_WORD requires label_type in {'onehot', 'srv'}")

        # ---------- path checks ----------
        if self.check_paths:
            self._validate_paths()

    def _validate_paths(self) -> None:
        if self.dataset == Dataset.BURGUNDY:
            if self.gammatone_dir is None:
                raise ValueError("gammatone_dir is required for Burgundy")
            if self.neighbors is None:
                raise ValueError("neighbors is required for Burgundy")

            if not self.gammatone_dir.exists():
                raise FileNotFoundError(f"gammatone_dir not found: {self.gammatone_dir}")
            if not self.neighbors.exists():
                raise FileNotFoundError(f"neighbors not found: {self.neighbors}")

        elif self.dataset == Dataset.LIBRISPEECH:
            if self.librispeech_root is not None and not self.librispeech_root.exists():
                raise FileNotFoundError(f"librispeech_root not found: {self.librispeech_root}")

            if self.librispeech_manifest is not None and not self.librispeech_manifest.exists():
                raise FileNotFoundError(f"librispeech_manifest not found: {self.librispeech_manifest}")

    # =====================================================
    # convenience
    # =====================================================
    def run_id(self) -> str:
        return (
            f"{self.dataset.value}_"
            f"{self.task_name.value}_"
            f"{self.label_type.value}_"
            f"{self.policy_name.value}_"
            f"s{self.seed:03d}"
        )

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

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return {k: _serialize_value(v) for k, v in raw.items()}

    def save_json(self, path: Path | str) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)