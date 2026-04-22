from pathlib import Path

DATA_ROOT = Path(r"C:\Dataset")

LIBRISPEECH_ROOT = DATA_ROOT / "LibriSpeech"
LIBRISPEECH_ALIGNMENT_ROOT = DATA_ROOT / "LibriSpeech-Alignments" / "LibriSpeech"
LIBRISPEECH_GAMMATONE_ROOT = DATA_ROOT / "LibriSpeech_coch64_env100_f16"
LIBRISPEECH_MANIFEST_ROOT = DATA_ROOT / "LibriSpeech" / "manifest_librispeech_coch_align.jsonl"
LIBRISPEECH_MANIFEST_PHONE_ROOT = DATA_ROOT / "LibriSpeech" / "manifest_librispeech_coch_align_phonemes.jsonl"

EARSHOT_ROOT = DATA_ROOT / "EARSHOT"
