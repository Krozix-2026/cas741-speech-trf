import json
import os
from pathlib import Path

import pytest

REAL_MANIFEST = os.getenv("LIBRISPEECH_MANIFEST", "")


@pytest.mark.skipif(not REAL_MANIFEST, reason="Set LIBRISPEECH_MANIFEST to run integration tests.")
def test_audio_files_have_valid_format_if_audio_path_present():
    sf = pytest.importorskip("soundfile")

    manifest = Path(REAL_MANIFEST)
    checked = 0

    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            audio_path = row.get("audio_path") or row.get("wav_path") or row.get("flac_path")
            if not audio_path:
                continue

            p = Path(audio_path)
            assert p.exists(), f"Missing audio file: {p}"
            assert p.suffix.lower() in {".wav", ".flac"}, f"Unexpected suffix: {p.suffix}"

            with sf.SoundFile(str(p)) as audio_file:
                assert audio_file.frames > 0
                assert audio_file.samplerate > 0
                assert audio_file.channels >= 1

            checked += 1
            if checked >= 5:
                break

    if checked == 0:
        pytest.skip("No raw audio path field found in manifest.")