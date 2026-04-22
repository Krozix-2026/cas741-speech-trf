import json
from pathlib import Path

import numpy as np
import pytest
import torch

from speech_dataset.librispeech_aligned_words import (
    IGNORE,
    WordVocab,
    build_word_vocab_from_manifest,
    LibriSpeechAlignedWords,
)
from speech_dataset.collate_aligned_words import collate_aligned_words


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_valid_coch(path: Path, T: int):
    arr = np.random.randn(64, T).astype(np.float32)
    np.save(path, arr)
    return arr


def _save_invalid_coch(path: Path):
    arr = np.random.randn(63, 10).astype(np.float32) # wrong first dim
    np.save(path, arr)
    return arr


@pytest.fixture
def fake_manifest_and_files(tmp_path: Path):
    coch1 = tmp_path / "utt1.npy"
    coch2 = tmp_path / "utt2.npy"
    coch3 = tmp_path / "utt3.npy"

    _save_valid_coch(coch1, T=20)
    _save_valid_coch(coch2, T=12)
    _save_valid_coch(coch3, T=8)

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "utt_id": "utt-train-1",
            "subset": "train",
            "coch_path": str(coch1),
            "words": [
                ["hello", 2, 6],
                ["world", 8, 12],
            ],
        },
        {
            "utt_id": "utt-train-2",
            "subset": "train",
            "coch_path": str(coch2),
            "words": [
                ["hello", 0, 3],
                ["test", 7, 12],
            ],
        },
        {
            "utt_id": "utt-dev-1",
            "subset": "dev",
            "coch_path": str(coch3),
            "words": [
                ["devword", 1, 4],
            ],
        },
    ]
    _write_jsonl(manifest, rows)
    return manifest, rows


def test_build_word_vocab_from_manifest(fake_manifest_and_files):
    manifest, _ = fake_manifest_and_files

    vocab = build_word_vocab_from_manifest(manifest, topk=10)

    assert isinstance(vocab, WordVocab)
    assert vocab.unk_id == 0
    assert vocab.itos[0] == "<unk>"
    assert "hello" in vocab.stoi
    assert "world" in vocab.stoi
    assert vocab.size >= 4


def test_build_word_vocab_manifest_not_found(tmp_path: Path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        build_word_vocab_from_manifest(missing)


def test_dataset_filters_subset(fake_manifest_and_files):
    manifest, _ = fake_manifest_and_files
    vocab = build_word_vocab_from_manifest(manifest)

    ds_train = LibriSpeechAlignedWords(manifest, subset="train", vocab=vocab)
    ds_dev = LibriSpeechAlignedWords(manifest, subset="dev", vocab=vocab)

    assert len(ds_train) == 2
    assert len(ds_dev) == 1


def test_dataset_limit_samples(fake_manifest_and_files):
    manifest, _ = fake_manifest_and_files
    vocab = build_word_vocab_from_manifest(manifest)

    ds = LibriSpeechAlignedWords(
        manifest_path=manifest,
        subset="train",
        vocab=vocab,
        limit_samples=1,
    )
    assert len(ds) == 1


def test_dataset_getitem_returns_expected_fields(fake_manifest_and_files):
    manifest, _ = fake_manifest_and_files
    vocab = build_word_vocab_from_manifest(manifest)

    ds = LibriSpeechAlignedWords(
        manifest_path=manifest,
        subset="train",
        vocab=vocab,
        tail_frames=3,
    )
    item = ds[0]

    assert set(item.keys()) == {
        "feat", "feat_len", "target",
        "word_ids", "word_starts", "word_ends", "utt_id"
    }

    assert isinstance(item["feat"], torch.Tensor)
    assert isinstance(item["target"], torch.Tensor)
    assert item["feat"].dtype == torch.float32
    assert item["feat"].shape[1] == 64
    assert item["feat"].shape[0] == item["feat_len"]
    assert item["target"].shape == (item["feat_len"],)

    # word sequence info
    assert item["word_ids"].ndim == 1
    assert item["word_starts"].ndim == 1
    assert item["word_ends"].ndim == 1
    assert len(item["word_ids"]) == len(item["word_starts"]) == len(item["word_ends"])
    assert item["utt_id"] == "utt-train-1"


def test_target_supervision_marks_tail_frames(fake_manifest_and_files):
    manifest, _ = fake_manifest_and_files
    vocab = build_word_vocab_from_manifest(manifest)

    ds = LibriSpeechAlignedWords(
        manifest_path=manifest,
        subset="train",
        vocab=vocab,
        tail_frames=2,
    )
    item = ds[0]

    # word span [2,6) => supervise [4,6)
    hello_id = vocab.stoi["hello"]   
    # word span [8,12) => supervise [10,12)
    world_id = vocab.stoi["world"]   

    target = item["target"].tolist()

    assert target[4] == hello_id
    assert target[5] == hello_id
    assert target[10] == world_id
    assert target[11] == world_id

    # a frame not covered by supervision should stay IGNORE
    assert target[0] == IGNORE


def test_out_of_range_word_boundaries_are_clipped_or_skipped(tmp_path: Path):
    coch = tmp_path / "utt.npy"
    _save_valid_coch(coch, T=10)

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "utt_id": "utt-1",
            "subset": "train",
            "coch_path": str(coch),
            "words": [
                ["negstart", -3, 2],# should be clipped to [0,2)
                ["toolong", 8, 20],# should be clipped to [8,10)
                ["invalid1", 5, 5],# skipped
                ["invalid2", 11, 15],# skipped
                ["invalid3", -2, 0],# skipped due to ef <= 0
            ],
        }
    ]
    _write_jsonl(manifest, rows)

    vocab = build_word_vocab_from_manifest(manifest)
    ds = LibriSpeechAlignedWords(manifest, subset="train", vocab=vocab, tail_frames=2)
    item = ds[0]

    assert item["word_ids"].numel() == 2
    assert item["word_starts"].tolist() == [0, 8]
    assert item["word_ends"].tolist() == [2, 10]


def test_invalid_coch_shape_raises_value_error(tmp_path: Path):
    bad_coch = tmp_path / "bad.npy"
    _save_invalid_coch(bad_coch)

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "utt_id": "utt-bad",
            "subset": "train",
            "coch_path": str(bad_coch),
            "words": [["bad", 0, 3]],
        }
    ]
    _write_jsonl(manifest, rows)

    vocab = build_word_vocab_from_manifest(manifest)
    ds = LibriSpeechAlignedWords(manifest, subset="train", vocab=vocab)

    with pytest.raises(ValueError, match="Unexpected coch shape"):
        _ = ds[0]


def test_missing_coch_file_raises_file_not_found(tmp_path: Path):
    missing_coch = tmp_path / "missing.npy"

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "utt_id": "utt-missing",
            "subset": "train",
            "coch_path": str(missing_coch),
            "words": [["x", 0, 1]],
        }
    ]
    _write_jsonl(manifest, rows)

    vocab = build_word_vocab_from_manifest(manifest)
    ds = LibriSpeechAlignedWords(manifest, subset="train", vocab=vocab)

    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_collate_aligned_words_sorts_and_pads(fake_manifest_and_files):
    manifest, _ = fake_manifest_and_files
    vocab = build_word_vocab_from_manifest(manifest)

    ds = LibriSpeechAlignedWords(manifest, subset="train", vocab=vocab)

    item0 = ds[0]# T=20
    item1 = ds[1]# T=12

    batch = collate_aligned_words([item1, item0])#intentionally reversed

    assert batch.feats.shape == (2, 20, 64)
    assert batch.targets.shape == (2, 20)
    assert batch.feat_lens.tolist() == [20, 12] #sorted descending
    assert batch.utt_ids == ["utt-train-1", "utt-train-2"]

    # padded tail of shorter sample should remain IGNORE for targets
    assert torch.all(batch.targets[1, 12:] == IGNORE)

    # word padding
    assert batch.word_ids.shape[0] == 2
    assert batch.word_starts.shape == batch.word_ids.shape
    assert batch.word_ends.shape == batch.word_ids.shape
    assert batch.word_lens.tolist() == [2, 2]