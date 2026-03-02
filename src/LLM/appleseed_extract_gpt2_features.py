# appleseed_extract_gpt2_features.py
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


@dataclass
class WordEvent:
    segment: str
    word_index: int
    word: str
    t_on: float
    t_off: float


_TEXT_RE = re.compile(r'text\s*=\s*"(.*)"\s*$')
_XMIN_RE = re.compile(r'xmin\s*=\s*([0-9.]+)\s*$')
_XMAX_RE = re.compile(r'xmax\s*=\s*([0-9.]+)\s*$')
_TIER_NAME_RE = re.compile(r'name\s*=\s*"(.*)"\s*$')


def _unescape_praat_string(s: str) -> str:
    # Praat TextGrid uses double quotes; inside, quotes are usually escaped as "" (rare in words).
    # We'll handle the most common case.
    return s.replace('""', '"')


def parse_words_from_textgrid(textgrid_path: Path) -> List[Tuple[str, float, float]]:
    """
    Return list of (word, t_on, t_off) from the IntervalTier named "words".
    Robust: parse each interval block without spilling into the next one.
    """
    lines = textgrid_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    in_item = False
    tier_class_interval = False
    tier_is_words = False

    results: List[Tuple[str, float, float]] = []

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        # start of item
        if s.startswith("item [") and s.endswith("]:"):
            in_item = True
            tier_class_interval = False
            tier_is_words = False
            i += 1
            continue

        if in_item:
            if s.startswith("class") and '"IntervalTier"' in s:
                tier_class_interval = True

            m = _TIER_NAME_RE.match(s)
            if m:
                name = _unescape_praat_string(m.group(1))
                tier_is_words = tier_class_interval and (name == "words")

            # parse only within words tier
            if tier_is_words and s.startswith("intervals [") and s.endswith("]:"):
                t_on = None
                t_off = None
                text = None

                j = i + 1
                # read until next interval or next item
                while j < len(lines):
                    s2 = lines[j].strip()
                    if (s2.startswith("intervals [") and s2.endswith("]:")) or (s2.startswith("item [") and s2.endswith("]:")):
                        break

                    mx = _XMIN_RE.match(s2)
                    if mx:
                        t_on = float(mx.group(1))

                    mx = _XMAX_RE.match(s2)
                    if mx:
                        t_off = float(mx.group(1))

                    mt = _TEXT_RE.match(s2)
                    if mt:
                        text = _unescape_praat_string(mt.group(1))

                    j += 1

                if text is not None and t_on is not None and t_off is not None:
                    results.append((text, t_on, t_off))

                # jump to end of this interval block
                i = j - 1

        i += 1

    if not results:
        raise RuntimeError(f'No word intervals found in tier "words" for: {textgrid_path}')
    return results


def clean_word(w: str, lowercase: bool, skip_labels: set) -> Optional[str]:
    w2 = w.strip()
    if lowercase:
        w2 = w2.lower()
    if w2 == "":
        return None
    if w2 in skip_labels:
        return None
    return w2


# -----------------------------
# GPT-2 feature extraction
# -----------------------------

def estimate_token_len(tok: GPT2TokenizerFast, word: str, is_first: bool) -> int:
    # GPT-2 BPE is whitespace-sensitive.
    # Approximation: prefix a space for non-first words.
    s = word if is_first else (" " + word)
    return len(tok.encode(s, add_special_tokens=False))


def chunk_words_by_ctx(tok, words, max_ctx: int, overlap_words: int):
    """
    Exact chunking using tokenizer measurement.
    Returns list of (start_word, end_word_exclusive, output_start_word).
    Guarantees each chunk token length <= max_ctx.
    """
    n = len(words)
    chunks = []
    start = 0
    first = True

    overlap_words = max(0, min(overlap_words, 400))

    while start < n:
        # binary search the largest end such that token_len <= max_ctx
        lo = start + 1
        hi = n
        best = None

        while lo <= hi:
            mid = (lo + hi) // 2
            enc = tok(
                words[start:mid],
                is_split_into_words=True,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            token_len = len(enc["input_ids"])
            if token_len <= max_ctx:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best is None:
            best = min(start + 1, n)

        end = best

        if first:
            output_start = start
            first = False
        else:
            output_start = min(start + overlap_words, end)

        chunks.append((start, end, output_start))

        if end >= n:
            break

        next_start = max(end - overlap_words, 0)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


@torch.no_grad()
def gpt2_features_for_segment(
    tok: GPT2TokenizerFast,
    model: GPT2LMHeadModel,
    words: List[str],
    layers: List[int],
    max_ctx: int,
    overlap_words: int,
    device: str,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Returns:
      surprisal: (N_words,) float32 (NaN for words we couldn't compute)
      H: dict[layer -> (N_words, hidden)] float16
    """
    n = len(words)
    surprisal = np.full((n,), np.nan, dtype=np.float32)

    hidden_size = model.config.n_embd
    H: Dict[int, np.ndarray] = {
        l: np.zeros((n, hidden_size), dtype=np.float16) for l in layers
    }

    chunks = chunk_words_by_ctx(tok, words, max_ctx=max_ctx, overlap_words=overlap_words)

    for (w_start, w_end, out_start) in chunks:
        chunk_words = words[w_start:w_end]

        enc = tok(
            chunk_words,
            is_split_into_words=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple length = n_layer+1 (emb + each layer)
        logits = out.logits  # (1, T_tokens, vocab)

        # token -> word id (within chunk)
        word_ids = enc.word_ids()
        if word_ids is None:
            raise RuntimeError("Tokenizer did not return word_ids(); ensure you use GPT2TokenizerFast.")

        # group token indices by word
        token_groups: List[List[int]] = [[] for _ in range(len(chunk_words))]
        for ti, wi in enumerate(word_ids):
            if wi is not None and 0 <= wi < len(chunk_words):
                token_groups[wi].append(ti)

        # surprisal per token (logits[t] predicts token_{t+1})
        logp = torch.log_softmax(logits[0, :-1], dim=-1)  # (T-1, vocab)
        ids = input_ids[0]  # (T,)

        # Fill outputs for words in [out_start, w_end)
        for local_wi in range(out_start - w_start, w_end - w_start):
            global_wi = w_start + local_wi
            if global_wi < 0 or global_wi >= n:
                continue

            toks = token_groups[local_wi]
            if not toks:
                continue

            # word surprisal = sum over its tokens, skipping token position 0 (no previous context within chunk)
            s = 0.0
            valid = 0
            for ti in toks:
                if ti == 0:
                    continue
                tok_id = int(ids[ti].item())
                s += float(-logp[ti - 1, tok_id].item())
                valid += 1
            if valid > 0:
                surprisal[global_wi] = np.float32(s)

            # hidden states: mean over tokens for each layer
            for l in layers:
                hs = hidden_states[l][0]  # (T_tokens, hidden)
                vec = hs[toks].mean(dim=0).detach().to("cpu").numpy()
                H[l][global_wi] = vec.astype(np.float16)

    return surprisal, H


# -----------------------------
# IO helpers
# -----------------------------

def save_events_csv(events: List[WordEvent], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment", "word_index", "word", "t_on", "t_off"])
        for e in events:
            w.writerow([e.segment, e.word_index, e.word, f"{e.t_on:.6f}", f"{e.t_off:.6f}"])


def save_events_jsonl(events: List[WordEvent], out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps({
                "segment": e.segment,
                "word_index": e.word_index,
                "word": e.word,
                "t_on": e.t_on,
                "t_off": e.t_off
            }, ensure_ascii=False) + "\n")


def try_import_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--textgrid_dir", type=str, default=r"C:\linux_project\CAS741\cas741-speech-trf\src\appleseed_eelbrain\Appleseed-main\stimuli\text", help="Directory containing segment*.TextGrid")
    ap.add_argument("--out_dir", type=str, default=r"C:\linux_project\CAS741\cas741-speech-trf\src\LLM\Appleseed_LLM_alignment", help="Output directory")
    ap.add_argument("--model_name", type=str, default="gpt2", help="HF model name (e.g., gpt2, gpt2-medium)")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 6, 12], help="Hidden state layers to save")
    ap.add_argument("--max_ctx", type=int, default=1000, help="Max context tokens for chunking")
    ap.add_argument("--overlap_words", type=int, default=80, help="Overlap words between chunks")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase words from TextGrid")
    ap.add_argument("--skip_labels", type=str, nargs="*", default=["sil", "sp", "silence", "<sil>", "<sp>"],
                    help="Labels to skip in words tier")
    ap.add_argument("--device", type=str, default="cuda", help="auto|cuda|cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Model dtype on GPU")
    args = ap.parse_args()

    textgrid_dir = Path(args.textgrid_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tg_files = sorted(textgrid_dir.glob("segment*.TextGrid"))
    if not tg_files:
        raise FileNotFoundError(f"No files matched segment*.TextGrid in: {textgrid_dir}")

    skip_labels = set([s.lower() for s in args.skip_labels])

    # Load model
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tok = GPT2TokenizerFast.from_pretrained(args.model_name)

    if device == "cuda" and args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = GPT2LMHeadModel.from_pretrained(args.model_name, torch_dtype=torch_dtype).to(device)
    model.eval()

    # Prepare global events + segment-wise extraction
    all_events: List[WordEvent] = []

    tqdm = try_import_tqdm()
    iterator = tqdm(tg_files, desc="Segments") if tqdm else tg_files

    for tg_path in iterator:
        segment_id = tg_path.stem  # "segment 1" or "segment 11b"
        raw = parse_words_from_textgrid(tg_path)
        # print("raw:", raw)

        # Clean + build events
        words: List[str] = []
        t_on: List[float] = []
        t_off: List[float] = []

        for (w, a, b) in raw:
            w2 = clean_word(w, lowercase=args.lowercase, skip_labels=skip_labels)
            # print("w2:", w2)
            if w2 is None:
                continue
            words.append(w2)
            t_on.append(float(a))
            t_off.append(float(b))

        if len(words) == 0:
            continue

        for i, (w, a, b) in enumerate(zip(words, t_on, t_off)):
            all_events.append(WordEvent(segment=segment_id, word_index=i, word=w, t_on=a, t_off=b))

        # if segment_id == "segment 1":
        #     print("[DEBUG] RAW first 15:", [x[0] for x in raw[:15]])
        #     print("[DEBUG] CLEAN first 15:", words[:15])
        #     print("[DEBUG] skip_labels:", sorted(list(skip_labels))[:20], "...")
        # print("words:", words)
        # print("words:", len(words))
        enc = tok(words, is_split_into_words=True, add_special_tokens=False)
        token_len = len(enc["input_ids"])
        print(segment_id, "words =", len(words), "tokens =", token_len, "ratio =", token_len/len(words))

        
        # GPT-2 features
        surprisal, H = gpt2_features_for_segment(
            tok=tok,
            model=model,
            words=words,
            layers=args.layers,
            max_ctx=args.max_ctx,
            overlap_words=args.overlap_words,
            device=device,
        )
        surprisal[0] = np.nan
        # print("surprisal:", surprisal)
        # print("H:", H)
        
        enc = tok(words, is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        tokens = tok.convert_ids_to_tokens(ids)
        word_ids = enc.word_ids()

        # group token indices by word id
        groups = [[] for _ in range(len(words))]
        for ti, wi in enumerate(word_ids):
            if wi is not None:
                groups[wi].append(ti)

        # print first 25 words with their tokens + surprisal
        for i in range(min(25, len(words))):
            tis = groups[i]
            toks = [tokens[t] for t in tis]
            print(f"{i:02d}  {words[i]:<12}  n_tok={len(tis):<2}  surprisal={surprisal[i]}  toks={toks}")
        

        # Save per-segment npz
        seg_out = out_dir / "segments"
        seg_out.mkdir(parents=True, exist_ok=True)
        npz_path = seg_out / f"{segment_id}.gpt2_features.npz"

        payload = {
            "segment": np.array([segment_id]),
            "words": np.array(words, dtype=object),
            "t_on": np.array(t_on, dtype=np.float32),
            "t_off": np.array(t_off, dtype=np.float32),
            "surprisal": surprisal.astype(np.float32),
            "model_name": np.array([args.model_name]),
            "layers": np.array(args.layers, dtype=np.int32),
            "max_ctx": np.array([args.max_ctx], dtype=np.int32),
            "overlap_words": np.array([args.overlap_words], dtype=np.int32),
        }
        for l, arr in H.items():
            payload[f"hidden_L{l}"] = arr  # float16

        np.savez_compressed(npz_path, **payload)

    # Save global events table
    save_events_csv(all_events, out_dir / "appleseed_word_events.csv")
    save_events_jsonl(all_events, out_dir / "appleseed_word_events.jsonl")

    print(f"[OK] Parsed TextGrids: {len(tg_files)}")
    print(f"[OK] Total words kept: {len(all_events)}")
    print(f"[OK] Wrote: {out_dir / 'appleseed_word_events.csv'}")
    print(f"[OK] Wrote per-segment features: {out_dir / 'segments'}")


if __name__ == "__main__":
    main()
