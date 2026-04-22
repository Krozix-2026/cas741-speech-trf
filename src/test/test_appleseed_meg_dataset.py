import os
from pathlib import Path
from collections import Counter

import pytest



# DATASET_ROOT = Path(r"C:\Dataset\Appleseed_BIDS_new")
DATASET_ROOT = Path(os.getenv("APPLESEED_BIDS_ROOT", r"C:\Dataset\Appleseed_BIDS_new"))

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 13)]
EMPTYROOM_DIR = DATASET_ROOT / "sub-emptyroom" / "ses-20191211" / "meg"


def find_meg_dirs(subject_dir: Path):

    return [p for p in subject_dir.rglob("meg") if p.is_dir()]


def find_fif_files_in_meg(subject_dir: Path):
    fif_files = []
    meg_dirs = find_meg_dirs(subject_dir)
    for meg_dir in meg_dirs:
        fif_files.extend(sorted(meg_dir.glob("*.fif")))
    return fif_files


def test_appleseed_dataset_root_exists():
  
    assert DATASET_ROOT.exists(), f"Dataset root does not exist: {DATASET_ROOT}"
    assert DATASET_ROOT.is_dir(), f"Dataset root is not a directory: {DATASET_ROOT}"


@pytest.mark.parametrize("subject", SUBJECTS)
def test_subject_directories_exist(subject):
    subject_dir = DATASET_ROOT / subject
    assert subject_dir.exists(), f"Missing subject directory: {subject_dir}"
    assert subject_dir.is_dir(), f"Subject path is not a directory: {subject_dir}"


@pytest.mark.parametrize("subject", SUBJECTS)
def test_each_subject_has_meg_folder(subject):

    subject_dir = DATASET_ROOT / subject
    meg_dirs = find_meg_dirs(subject_dir)

    assert len(meg_dirs) > 0, f"No meg directory found under {subject_dir}"


@pytest.mark.parametrize("subject", SUBJECTS)
def test_each_subject_has_fif_files(subject):

    subject_dir = DATASET_ROOT / subject
    fif_files = find_fif_files_in_meg(subject_dir)

    assert len(fif_files) > 0, f"No .fif files found under meg folders of {subject_dir}"

    for f in fif_files:
        assert f.exists(), f"Missing fif file: {f}"
        assert f.is_file(), f"Path is not a file: {f}"
        assert f.suffix == ".fif", f"Unexpected file suffix: {f}"


def test_fif_file_counts_are_uniform_across_subjects():

    counts = {}
    for subject in SUBJECTS:
        subject_dir = DATASET_ROOT / subject
        fif_files = find_fif_files_in_meg(subject_dir)
        counts[subject] = len(fif_files)

    unique_counts = set(counts.values())

    assert len(unique_counts) == 1, (
        "The number of .fif files is not uniform across subjects.\n"
        + "\n".join([f"{k}: {v}" for k, v in counts.items()])
    )


def test_emptyroom_directory_exists():
    assert EMPTYROOM_DIR.exists(), f"Empty-room directory does not exist: {EMPTYROOM_DIR}"
    assert EMPTYROOM_DIR.is_dir(), f"Empty-room path is not a directory: {EMPTYROOM_DIR}"


def test_emptyroom_has_fif_files():

    fif_files = sorted(EMPTYROOM_DIR.glob("*.fif"))

    assert len(fif_files) > 0, f"No .fif files found in empty-room directory: {EMPTYROOM_DIR}"

    for f in fif_files:
        assert f.exists(), f"Missing empty-room fif file: {f}"
        assert f.is_file(), f"Empty-room path is not a file: {f}"
        assert f.suffix == ".fif", f"Unexpected empty-room file suffix: {f}"


def test_report_fif_counts_for_debug():

    counts = {}
    for subject in SUBJECTS:
        subject_dir = DATASET_ROOT / subject
        counts[subject] = len(find_fif_files_in_meg(subject_dir))

    print("\nFIF counts by subject:")
    for k, v in counts.items():
        print(f"{k}: {v}")

    # print
    assert True