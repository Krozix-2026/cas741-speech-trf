# make_librispeech_phoneme_manifest.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from datasets import load_dataset


def strip_stress(phone: str) -> str:
    # ARPAbet stress is usually a trailing digit: AH0/AH1/AH2, ER0 ...
    return re.sub(r"\d$", "", phone)


def subset_to_hf_split(subset: str) -> str:
    # your manifest uses: train-clean-100 / dev-clean / test-clean
    # HF uses: train_clean_100 / dev_clean / test_clean
    return subset.replace("-", "_")


def load_alignment_maps() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Return:
      maps[split][utt_id] = list_of_phoneme_dicts (each has phoneme/start/end in seconds)
    """
    maps: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for split in ["train_clean_100", "dev_clean", "test_clean"]:
        ds = load_dataset("gilkeyio/librispeech-alignments", split=split)
        m = {}
        for ex in ds:
            m[ex["id"]] = ex["phonemes"]
        maps[split] = m
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")
    ap.add_argument("--out_manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align_phonemes.jsonl")
    ap.add_argument("--env_sr", type=int, default=100, help="frames per second for coch features")
    ap.add_argument("--strip_stress", action="store_true", help="collapse AH0/AH1 -> AH")
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    maps = load_alignment_maps()

    n_total = 0
    n_ok = 0
    n_miss = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            n_total += 1

            subset = obj.get("subset", "")
            hf_split = subset_to_hf_split(subset)
            utt_id = obj.get("utt_id", obj.get("id", ""))

            # Only process the splits we care about
            if hf_split not in maps:
                obj["phonemes"] = []
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            ph_list = maps[hf_split].get(utt_id)
            if ph_list is None:
                # alignment missing
                n_miss += 1
                obj["phonemes"] = []
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # Determine T (frames) without loading .npy if possible
            T = obj.get("T", None)
            if T is None:
                coch = np.load(obj["coch_path"])
                # coch is (64, T)
                T = int(coch.shape[1])
            else:
                T = int(T)

            phonemes_frames: List[List[Any]] = []
            for ph in ph_list:
                p = ph["phoneme"]
                if args.strip_stress:
                    p = strip_stress(p)

                sf = int(np.floor(float(ph["start"]) * args.env_sr))
                ef = int(np.ceil(float(ph["end"]) * args.env_sr))

                # clamp & sanity
                if ef <= 0 or sf >= T:
                    continue
                sf = max(sf, 0)
                ef = min(ef, T)
                if ef <= sf:
                    continue

                phonemes_frames.append([p, sf, ef])

            obj["phonemes"] = phonemes_frames
            n_ok += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote: {out_path}")
    print(f"total={n_total} ok={n_ok} miss_alignment={n_miss}")


if __name__ == "__main__":
    main()