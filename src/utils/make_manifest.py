from pathlib import Path
import json, re
import numpy as np

ENV_SR = 100
TAIL_MS = 100
K = int(round((TAIL_MS/1000) * ENV_SR))  # 10 frames at 100Hz

COCH_ROOT = Path(r"C:\Dataset\LibriSpeech_coch64_env100_f16")
ALIGN_ROOT = Path(r"C:\Dataset\Librispeech-Alignments\LibriSpeech")
OUT_MANIFEST = Path(r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")

SUBSETS = ["train-clean-100", "dev-clean", "test-clean"]

line_re = re.compile(r'^(\S+)\s+"([^"]*)"\s+"([^"]*)"\s*$')

def clean_word(tok: str) -> str:
    tok = tok.strip()
    if not tok:
        return ""
    # 去掉两端标点，保留内部 '
    tok = re.sub(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$", "", tok)
    return tok.lower()

def parse_alignment_line(line: str):
    m = line_re.match(line.strip())
    if not m:
        return None
    utt_id, token_csv, time_csv = m.group(1), m.group(2), m.group(3)
    # print("utt_id:", utt_id)
    # print("token_csv:", token_csv)
    # print("time_csv:", time_csv)
    tokens = token_csv.split(",")
    times = [float(x) for x in time_csv.split(",") if x.strip() != ""]

    if len(tokens) != len(times):
        return None  # 先保守跳过，避免静默错位

    # 这里采用：times[i] 是 token i 的“结束时间”
    # start = prev_end, end = times[i]
    segs = []
    prev = 0.0
    for tok, end in zip(tokens, times):
        start = prev
        prev = end
        w = clean_word(tok)
        if not w:
            continue
        if end <= start:
            continue
        segs.append((w, start, end))
    return utt_id, segs

def sec_to_frame(t: float) -> int:
    return int(round(t * ENV_SR))

def find_coch_path(subset: str, utt_id: str) -> Path | None:
    # 镜像结构里 utt_id.npy 在某个 speaker/chapter 目录下，rglob 搜索最稳（可缓存优化）
    hits = list((COCH_ROOT / subset).rglob(f"{utt_id}.npy"))
    return hits[0] if hits else None

def main():
    n = 0
    with OUT_MANIFEST.open("w", encoding="utf-8") as f_out:
        for subset in SUBSETS:
            for aln_file in (ALIGN_ROOT / subset).rglob("*.alignment.txt"):
                for line in aln_file.read_text(encoding="utf-8").splitlines():
                    parsed = parse_alignment_line(line)
                    # print("parsed:", parsed)
                    if parsed is None:
                        continue
                    utt_id, segs = parsed

                    coch_path = find_coch_path(subset, utt_id)
                    # print("coch_path:", coch_path)
                    if coch_path is None:
                        continue

                    coch = np.load(coch_path, mmap_mode="r")  # (F, T)
                    print("coch:", coch.shape)#(64, 200)
                    T = coch.shape[-1]

                    # 映射到帧并裁剪到 [0, T]
                    words_f = []
                    for w, s, e in segs:
                        sf = max(sec_to_frame(s), 0)
                        ef = min(sec_to_frame(e), T)
                        if ef > sf:
                            words_f.append((w, sf, ef))

                    if not words_f:
                        continue

                    obj = {
                        "subset": subset,
                        "utt_id": utt_id,
                        "coch_path": str(coch_path),
                        "T": int(T),
                        "words": words_f,
                    }
                    f_out.write(json.dumps(obj) + "\n")
                    n += 1
                    if n % 2000 == 0:
                        print("written", n)

    print("Done. total utterances:", n)
    print("Manifest:", OUT_MANIFEST)

if __name__ == "__main__":
    main()