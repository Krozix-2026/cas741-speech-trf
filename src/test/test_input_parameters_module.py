import json
from pathlib import Path

import pytest

from config.enums import Dataset as EnumDataset
from config.enums import TaskName as EnumTaskName
from config.enums import LabelType as EnumLabelType
from config.enums import PolicyName as EnumPolicyName

from config.paths_local import (
    DATA_ROOT,
    LIBRISPEECH_ROOT,
    LIBRISPEECH_ALIGNMENT_ROOT,
    LIBRISPEECH_GAMMATONE_ROOT,
    LIBRISPEECH_MANIFEST_ROOT,
    LIBRISPEECH_MANIFEST_PHONE_ROOT,
    EARSHOT_ROOT,
)

from config.presets import (
    PRESETS,
    get_preset,
    librispeech_lstm,
    librispeech_lstm_phone_srv,
    librispeech_lstm_word_onehot,
    librispeech_lstm_word_srv,
    librispeech_lstm_word_semantic_hier,
    librispeech_las_mocha_words,
    librispeech_las_words,
    burgundy_lstm,
)

from config.schema import (
    TrainConfig,
    Dataset,
    TaskName,
    LabelType,
    PolicyName,
)



def test_dataset_enum_values():
    assert EnumDataset.LIBRISPEECH == "librispeech"
    assert EnumDataset.BURGUNDY == "burgundy"


def test_taskname_enum_values():
    assert EnumTaskName.LSTM == "LSTM"
    assert EnumTaskName.LSTM_PHONE == "LSTM_PHONE"
    assert EnumTaskName.LSTM_WORD == "LSTM_WORD"
    assert EnumTaskName.LSTM_WORD_SEM == "LSTM_WORD_SEM"
    assert EnumTaskName.LAS_MOCHA_WORDS == "LAS_MOCHA_WORDS"
    assert EnumTaskName.LAS_WORDS == "LAS_WORDS"
    assert EnumTaskName.RNNT == "RNNT"


def test_labeltype_enum_values():
    assert EnumLabelType.ONEHOT == "onehot"
    assert EnumLabelType.SRV == "srv"


def test_policyname_enum_values():
    assert EnumPolicyName.BASELINE == "baseline"
    assert EnumPolicyName.MODIFIED_LOSS == "modified_loss"
    assert EnumPolicyName.SEMANTIC == "semantic"
    assert EnumPolicyName.SEMANTIC_HIER == "semantic_hier"
    assert EnumPolicyName.LAS_GLOBAL == "las_global"



def test_paths_local_are_path_objects():
    assert isinstance(DATA_ROOT, Path)
    assert isinstance(LIBRISPEECH_ROOT, Path)
    assert isinstance(LIBRISPEECH_ALIGNMENT_ROOT, Path)
    assert isinstance(LIBRISPEECH_GAMMATONE_ROOT, Path)
    assert isinstance(LIBRISPEECH_MANIFEST_ROOT, Path)
    assert isinstance(LIBRISPEECH_MANIFEST_PHONE_ROOT, Path)
    assert isinstance(EARSHOT_ROOT, Path)


def test_paths_local_expected_suffixes():
    assert LIBRISPEECH_ROOT.name == "LibriSpeech"
    assert LIBRISPEECH_ALIGNMENT_ROOT.name == "LibriSpeech"
    assert LIBRISPEECH_GAMMATONE_ROOT.name == "LibriSpeech_coch64_env100_f16"
    assert LIBRISPEECH_MANIFEST_ROOT.name == "manifest_librispeech_coch_align.jsonl"
    assert LIBRISPEECH_MANIFEST_PHONE_ROOT.name == "manifest_librispeech_coch_align_phonemes.jsonl"
    assert EARSHOT_ROOT.name == "EARSHOT"



def test_trainconfig_normalizes_string_inputs_without_path_check():
    cfg = TrainConfig(
        dataset="LIBRISPEECH",
        task_name="lstm",
        label_type="ONEHOT",
        policy_name="MODIFIED_LOSS",
        check_paths=False,
    )

    assert cfg.dataset == Dataset.LIBRISPEECH
    assert cfg.task_name == TaskName.LSTM
    assert cfg.label_type == LabelType.ONEHOT
    assert cfg.policy_name == PolicyName.MODIFIED_LOSS


def test_librispeech_defaults_applied():
    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        check_paths=False,
    )

    assert cfg.train_subset == "train-clean-100"
    assert cfg.valid_subset == "dev-clean"
    assert cfg.test_subset == "test-clean"
    assert cfg.librispeech_root is not None
    assert cfg.librispeech_manifest is not None


def test_burgundy_defaults_applied():
    cfg = TrainConfig(
        dataset=Dataset.BURGUNDY,
        task_name=TaskName.LSTM,
        check_paths=False,
    )

    assert len(cfg.speaker) > 0
    assert cfg.gammatone_dir is not None
    assert cfg.neighbors is not None
    assert cfg.epoch == 500


def test_lstm_phone_uses_phone_manifest_when_available(monkeypatch):
    import config.schema as schema_mod

    fake_phone_manifest = Path("fake_phone_manifest.jsonl")
    monkeypatch.setattr(schema_mod, "_LIBRISPEECH_MANIFEST_PHONE_ROOT", fake_phone_manifest)

    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM_PHONE,
        label_type=LabelType.SRV,
        check_paths=False,
    )

    assert cfg.librispeech_manifest == fake_phone_manifest


def test_las_mocha_words_task_defaults():
    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LAS_MOCHA_WORDS,
        batch_size=128,
        check_paths=False,
    )

    assert cfg.bidirectional is False
    assert cfg.batch_size == 8


def test_las_words_task_defaults():
    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LAS_WORDS,
        batch_size=64,
        bidirectional=False,
        check_paths=False,
    )

    assert cfg.bidirectional is True
    assert cfg.batch_size == 16


def test_rnnt_task_defaults():
    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.RNNT,
        batch_size=32,
        lstm_hidden=512,
        lstm_layers=3,
        check_paths=False,
    )

    assert cfg.lstm_hidden == 256
    assert cfg.lstm_layers == 1
    assert cfg.bidirectional is False
    assert cfg.batch_size == 1



@pytest.mark.parametrize(
    "kwargs, expected_msg",
    [
        ({"seed": -1}, "seed must be >= 0"),
        ({"epoch": 0}, "epoch must be > 0"),
        ({"batch_size": 0}, "batch_size must be > 0"),
        ({"lr": 0}, "lr must be > 0"),
        ({"weight_decay": -1}, "weight_decay must be >= 0"),
        ({"grad_clip": 0}, "grad_clip must be > 0"),
        ({"num_workers": -1}, "num_workers must be >= 0"),
        ({"srv_dim": 0}, "srv_dim must be > 0"),
        ({"srv_k": 0}, "srv_k must be > 0"),
        ({"srv_eval_max_frames": 0}, "srv_eval_max_frames must be > 0"),
        ({"phone_tail_ms": 0}, "phone_tail_ms must be > 0"),
        ({"phone_center_ms": 0}, "phone_center_ms must be > 0"),
        ({"tail_ms": 0}, "tail_ms must be > 0"),
        ({"semantic_alpha": -0.1}, "semantic_alpha must be >= 0"),
        ({"semantic_beta": -0.1}, "semantic_beta must be >= 0"),
        ({"semantic_shift": 0}, "semantic_shift must be >= 1"),
        ({"sample_rate": 0}, "sample_rate and n_mels must be > 0"),
        ({"win_length_ms": 0}, "win_length_ms and hop_length_ms must be > 0"),
        ({"lstm_hidden": 0}, "lstm_hidden and lstm_layers must be > 0"),
    ],
)
def test_validate_numeric_ranges(kwargs, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        TrainConfig(check_paths=False, **kwargs)


@pytest.mark.parametrize(
    "kwargs, expected_msg",
    [
        ({"srv_value": "abc"}, "srv_value must be 'pm1' or 'binary'"),
        ({"srv_loss": "abc"}, "srv_loss must be 'cosine' or 'mse'"),
        ({"phone_label_region": "abc"}, "phone_label_region must be one of"),
    ],
)
def test_validate_categorical_values(kwargs, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        TrainConfig(check_paths=False, **kwargs)



def test_burgundy_only_supports_lstm():
    with pytest.raises(ValueError, match="dataset='burgundy' currently only supports task_name='LSTM'"):
        TrainConfig(
            dataset=Dataset.BURGUNDY,
            task_name=TaskName.LAS_WORDS,
            check_paths=False,
        )


def test_lstm_phone_requires_srv():
    with pytest.raises(ValueError, match="LSTM_PHONE currently requires label_type='srv'"):
        TrainConfig(
            dataset=Dataset.LIBRISPEECH,
            task_name=TaskName.LSTM_PHONE,
            label_type=LabelType.ONEHOT,
            check_paths=False,
        )


def test_lstm_word_sem_requires_srv():
    with pytest.raises(ValueError, match="LSTM_WORD_SEM requires label_type='srv'"):
        TrainConfig(
            dataset=Dataset.LIBRISPEECH,
            task_name=TaskName.LSTM_WORD_SEM,
            label_type=LabelType.ONEHOT,
            check_paths=False,
        )



def test_librispeech_missing_root_raises():
    with pytest.raises(FileNotFoundError, match="librispeech_root not found"):
        TrainConfig(
            dataset=Dataset.LIBRISPEECH,
            librispeech_root=Path("this_path_should_not_exist_12345"),
            librispeech_manifest=None,
            check_paths=True,
        )


def test_librispeech_missing_manifest_raises(tmp_path):
    fake_root = tmp_path / "LibriSpeech"
    fake_root.mkdir()

    with pytest.raises(FileNotFoundError, match="librispeech_manifest not found"):
        TrainConfig(
            dataset=Dataset.LIBRISPEECH,
            librispeech_root=fake_root,
            librispeech_manifest=tmp_path / "missing_manifest.jsonl",
            check_paths=True,
        )


def test_librispeech_existing_paths_pass(tmp_path):
    fake_root = tmp_path / "LibriSpeech"
    fake_root.mkdir()

    fake_manifest = tmp_path / "manifest.jsonl"
    fake_manifest.write_text("", encoding="utf-8")

    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        librispeech_root=fake_root,
        librispeech_manifest=fake_manifest,
        check_paths=True,
    )

    assert cfg.librispeech_root == fake_root
    assert cfg.librispeech_manifest == fake_manifest


def test_burgundy_missing_paths_raise():
    with pytest.raises(FileNotFoundError, match="gammatone_dir not found"):
        TrainConfig(
            dataset=Dataset.BURGUNDY,
            task_name=TaskName.LSTM,
            gammatone_dir=Path("missing_gammatone_dir_12345"),
            neighbors=Path("missing_neighbors_12345.txt"),
            check_paths=True,
        )



def test_presets_dict_contains_expected_keys():
    expected = {
        "ls_lstm",
        "ls_lstm_phone_srv",
        "ls_lstm_word_onehot",
        "ls_lstm_word_srv",
        "ls_lstm_word_semantic_hier",
        "ls_las_mocha_words",
        "ls_las_words",
        "burgundy_lstm",
    }
    assert set(PRESETS.keys()) == expected


@pytest.mark.parametrize(
    "preset_name, expected_dataset, expected_task",
    [
        ("ls_lstm", Dataset.LIBRISPEECH, TaskName.LSTM),
        ("ls_lstm_phone_srv", Dataset.LIBRISPEECH, TaskName.LSTM_PHONE),
        ("ls_lstm_word_onehot", Dataset.LIBRISPEECH, TaskName.LSTM_WORD),
        ("ls_lstm_word_srv", Dataset.LIBRISPEECH, TaskName.LSTM_WORD),
        ("ls_lstm_word_semantic_hier", Dataset.LIBRISPEECH, TaskName.LSTM_WORD_SEM),
        ("ls_las_mocha_words", Dataset.LIBRISPEECH, TaskName.LAS_MOCHA_WORDS),
        ("ls_las_words", Dataset.LIBRISPEECH, TaskName.LAS_WORDS),
        ("burgundy_lstm", Dataset.BURGUNDY, TaskName.LSTM),
    ],
)
def test_get_preset_returns_expected_basic_fields(preset_name, expected_dataset, expected_task):
    cfg = get_preset(preset_name, seed=7)

    assert isinstance(cfg, TrainConfig)
    assert cfg.dataset == expected_dataset
    assert cfg.task_name == expected_task
    assert cfg.seed == 7


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError, match="Unknown preset"):
        get_preset("unknown_preset_name", seed=0)


def test_librispeech_lstm_phone_srv_preset_fields():
    cfg = librispeech_lstm_phone_srv(seed=1)
    assert cfg.dataset == Dataset.LIBRISPEECH
    assert cfg.task_name == TaskName.LSTM_PHONE
    assert cfg.label_type == LabelType.SRV
    assert cfg.policy_name == PolicyName.MODIFIED_LOSS
    assert cfg.srv_dim == 2048
    assert cfg.srv_k == 16
    assert cfg.srv_loss == "cosine"
    assert cfg.srv_value == "pm1"
    assert cfg.phone_label_region == "center"
    assert cfg.phone_center_ms == 30


def test_librispeech_lstm_word_onehot_preset_fields():
    cfg = librispeech_lstm_word_onehot(seed=2)
    assert cfg.task_name == TaskName.LSTM_WORD
    assert cfg.label_type == LabelType.ONEHOT
    assert cfg.policy_name == PolicyName.BASELINE


def test_librispeech_lstm_word_srv_preset_fields():
    cfg = librispeech_lstm_word_srv(seed=3)
    assert cfg.task_name == TaskName.LSTM_WORD
    assert cfg.label_type == LabelType.SRV
    assert cfg.policy_name == PolicyName.BASELINE
    assert cfg.srv_dim == 2048


def test_librispeech_lstm_word_semantic_hier_preset_fields():
    cfg = librispeech_lstm_word_semantic_hier(seed=4)
    assert cfg.task_name == TaskName.LSTM_WORD_SEM
    assert cfg.label_type == LabelType.SRV
    assert cfg.policy_name == PolicyName.SEMANTIC_HIER
    assert cfg.semantic_alpha == 1.0
    assert cfg.semantic_beta == 0.5
    assert cfg.word_rep_dim == 256
    assert cfg.word_lstm_hidden == 256
    assert cfg.word_lstm_layers == 1
    assert cfg.semantic_shift == 1
    assert cfg.semantic_detach_frame is False


def test_librispeech_las_mocha_words_preset_fields():
    cfg = librispeech_las_mocha_words(seed=5)
    assert cfg.task_name == TaskName.LAS_MOCHA_WORDS
    assert cfg.batch_size == 8   # preset给128，但task默认会cap到8
    assert cfg.epoch == 200
    assert cfg.lstm_layers == 1
    assert cfg.lstm_hidden == 512
    assert cfg.las_enc_subsample == 2
    assert cfg.las_max_words == 200


def test_librispeech_las_words_preset_fields():
    cfg = librispeech_las_words(seed=6)
    assert cfg.task_name == TaskName.LAS_WORDS
    assert cfg.policy_name == PolicyName.LAS_GLOBAL
    assert cfg.batch_size == 16   # preset给64，但task默认会cap到16
    assert cfg.epoch == 200
    assert cfg.lstm_layers == 2
    assert cfg.lstm_hidden == 256
    assert cfg.bidirectional is True
    assert cfg.las_enc_subsample == 2
    assert cfg.las_max_words == 200


def test_burgundy_lstm_preset_fields():
    cfg = burgundy_lstm(seed=7)
    assert cfg.dataset == Dataset.BURGUNDY
    assert cfg.task_name == TaskName.LSTM
    assert cfg.policy_name == PolicyName.MODIFIED_LOSS
    assert cfg.seed == 7



def test_run_id_run_dir_and_subdirs(tmp_path):
    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM,
        label_type=LabelType.ONEHOT,
        policy_name=PolicyName.MODIFIED_LOSS,
        runs_root=tmp_path,
        check_paths=False,
    )

    expected_run_id = "librispeech_LSTM_onehot_modified_loss_s000"
    assert cfg.run_id() == expected_run_id
    assert cfg.run_dir == tmp_path / expected_run_id
    assert cfg.ckpt_dir == cfg.run_dir / "ckpt"
    assert cfg.log_dir == cfg.run_dir / "logs"


def test_ensure_dirs_creates_directories(tmp_path):
    cfg = TrainConfig(runs_root=tmp_path, check_paths=False)
    cfg.ensure_dirs()

    assert cfg.run_dir.exists()
    assert cfg.ckpt_dir.exists()
    assert cfg.log_dir.exists()


def test_to_dict_serializes_path_and_enum(tmp_path):
    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM,
        policy_name=PolicyName.MODIFIED_LOSS,
        runs_root=tmp_path,
        check_paths=False,
    )

    d = cfg.to_dict()
    assert isinstance(d["dataset"], str)
    assert d["dataset"] == "librispeech"
    assert isinstance(d["runs_root"], str)


def test_save_json_writes_valid_json(tmp_path):
    cfg = TrainConfig(
        dataset=Dataset.LIBRISPEECH,
        task_name=TaskName.LSTM,
        check_paths=False,
    )

    out = tmp_path / "config.json"
    cfg.save_json(out)

    assert out.exists()

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"] == "librispeech"
    assert data["task_name"] == "LSTM"