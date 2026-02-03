from __future__ import annotations
from pathlib import Path
import re, json, gzip


ALIGN_ROOT = Path(r"C:\Dataset\LibriSpeech-Alignments\LibriSpeech")
OUT_DIR = Path(r"C:\Dataset\LibriSpeech-Alignments\processed") 
UNALIGNED = Path(r"C:\Dataset\LibriSpeech-Alignments\unaligned.txt")

SUBSETS = ["train-clean-100", "dev-clean", "test-clean"]

# ====== Token cleaning ======
_strip_edge = re.compile(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$") # 去掉两端标点，但保留内部 apostrophe

def clean_token(tok: str) -> str:
    tok = tok.strip()
    if not tok:
        return ""
    tok = _strip_edge.sub("", tok)
    return tok.lower()

def parse_alignment_line(line: str):
    """
    Parse one line:
      utt_id "tok1,tok2,,tokN" "t1,t2,...,tN"
    Return: utt_id, tokens(list), times(list[float])
    """
    line = line.strip()
    if not line:
        return None

    # 找到 4 个引号位置
    q = [i for i, ch in enumerate(line) if ch == '"']
    if len(q) < 4:
        return None

    utt_id = line[:q[0]].strip()
    tokens_str = line[q[0] + 1 : q[1]]
    times_str  = line[q[2] + 1 : q[3]]

    tokens = tokens_str.split(",")
    times  = [float(x) for x in times_str.split(",") if x.strip()]

    # 容错：对齐长度不一致时做最保守修正
    if len(times) == len(tokens) - 1 and tokens and tokens[-1] == "":
        tokens = tokens[:-1]
    if len(times) != len(tokens):
        # 放弃这行，避免悄悄错位
        return None

    return utt_id, tokens, times

def build_word_segments(tokens, times):
    """
    Assume times[i] is END time of token i.
    start(i)=times[i-1] (or 0), end(i)=times[i]
    Skip empty/punct tokens after cleaning.
    """
    segs = []
    prev_end = 0.0
    for tok, end in zip(tokens, times):
        start = prev_end
        prev_end = end
        w = clean_token(tok)
        if not w:
            continue
        # 过滤极短/异常片段（可选）
        if end <= start:
            continue
        segs.append((w, start, end))
    return segs

def load_unaligned(path: Path) -> set[str]:
    bad = set()
    if not path.exists():
        return bad
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        # 常见格式：utt_id 或带其它列；取第一个字段
        bad.add(s.split()[0])
    return bad

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bad_utts = load_unaligned(UNALIGNED)
    print(f"Loaded unaligned utts: {len(bad_utts)}")

    for subset in SUBSETS:
        subset_dir = ALIGN_ROOT / subset
        if not subset_dir.exists():
            print(f"[Skip] subset not found: {subset_dir}")
            continue

        out_path = OUT_DIR / f"{subset}.word_align.jsonl.gz"
        n_lines = 0
        n_ok = 0
        n_skipped = 0

        with gzip.open(out_path, "wt", encoding="utf-8") as f_out:
            for aln_file in subset_dir.rglob("*.alignment.txt"):
                for line in aln_file.read_text(encoding="utf-8").splitlines():
                    n_lines += 1
                    parsed = parse_alignment_line(line)
                    if parsed is None:
                        n_skipped += 1
                        continue
                    utt_id, tokens, times = parsed
                    if utt_id in bad_utts:
                        continue

                    segs = build_word_segments(tokens, times)
                    if not segs:
                        n_skipped += 1
                        continue

                    # 写 jsonl：一行一个 utterance
                    obj = {"utt_id": utt_id, "words": segs}  # words: [ [w, start, end], ... ]
                    f_out.write(json.dumps(obj) + "\n")
                    n_ok += 1

        print(f"[{subset}] wrote: {out_path}")
        print(f"  lines={n_lines}, ok_utts={n_ok}, skipped_lines={n_skipped}")

if __name__ == "__main__":
    main()