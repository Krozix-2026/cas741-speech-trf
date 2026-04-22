import json
import os
from pathlib import Path

import numpy as np
import pytest

from speech_dataset.librispeech_aligned_words import (
    build_word_vocab_from_manifest,
    LibriSpeechAlignedWords,
)

REAL_MANIFEST = os.getenv("LIBRISPEECH_MANIFEST", "")


pytestmark = pytest.mark.integration


def _read_first_n_jsonl(path: Path, n: int = 5):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


@pytest.mark.skipif(not REAL_MANIFEST, reason="Set LIBRISPEECH_MANIFEST to run integration tests.")
def test_real_manifest_exists():
    manifest = Path(REAL_MANIFEST)
    assert manifest.exists()
    assert manifest.is_file()
    assert manifest.stat().st_size > 0


@pytest.mark.skipif(not REAL_MANIFEST, reason="Set LIBRISPEECH_MANIFEST to run integration tests.")
def test_real_manifest_not_empty():
    manifest = Path(REAL_MANIFEST)
    rows = _read_first_n_jsonl(manifest, n=3)
    assert len(rows) > 0


@pytest.mark.skipif(not REAL_MANIFEST, reason="Set LIBRISPEECH_MANIFEST to run integration tests.")
def test_real_manifest_entries_have_required_keys():
    manifest = Path(REAL_MANIFEST)
    rows = _read_first_n_jsonl(manifest, n=5)

    required = {"utt_id", "subset", "coch_path", "words"}
    for row in rows:
        assert required.issubset(row.keys())
        assert isinstance(row["words"], list)


@pytest.mark.skipif(not REAL_MANIFEST, reason="Set LIBRISPEECH_MANIFEST to run integration tests.")
def test_real_coch_files_exist_and_are_readable():
    manifest = Path(REAL_MANIFEST)
    rows = _read_first_n_jsonl(manifest, n=5)

    for row in rows:
        coch_path = Path(row["coch_path"])
        assert coch_path.exists(), f"Missing coch file: {coch_path}"
        assert coch_path.suffix == ".npy", f"Unexpected suffix: {coch_path.suffix}"

        arr = np.load(coch_path)
        assert arr.ndim == 2
        assert arr.shape[0] == 64
        assert arr.shape[1] > 0


@pytest.mark.skipif(not REAL_MANIFEST, reason="Set LIBRISPEECH_MANIFEST to run integration tests.")
def test_real_dataset_can_be_constructed_and_is_not_empty():
    manifest = Path(REAL_MANIFEST)
    vocab = build_word_vocab_from_manifest(manifest, topk=1000)

    ds = LibriSpeechAlignedWords(
        manifest_path=manifest,
        subset="train",
        vocab=vocab,
        limit_samples=5,
    )
    assert len(ds) > 0


@pytest.mark.skipif(not REAL_MANIFEST, reason="Set LIBRISPEECH_MANIFEST to run integration tests.")
def test_real_dataset_first_item_can_be_loaded():
    manifest = Path(REAL_MANIFEST)
    vocab = build_word_vocab_from_manifest(manifest, topk=1000)

    ds = LibriSpeechAlignedWords(
        manifest_path=manifest,
        subset="train",
        vocab=vocab,
        limit_samples=5,
    )
    item = ds[0]

    assert item["feat"].ndim == 2
    assert item["feat"].shape[1] == 64
    assert item["feat_len"] == item["feat"].shape[0]
    assert item["target"].shape[0] == item["feat_len"]