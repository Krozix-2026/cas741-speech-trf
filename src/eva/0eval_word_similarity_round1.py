# eva\0eval_word_similarity_round1.py
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from network.lstm_frame_srv import LSTMFrameSRV


IGNORE = -100


def levenshtein(a, b):
    """edit distance for list/str"""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(
                dp[j] + 1,      # delete
                dp[j - 1] + 1,  # insert
                prev + cost     # subst
            )
            prev = cur
    return dp[-1]


def try_get_pronunciation_backend():
    """Prefer CMUdict via 'pronouncing'. If not available, return None."""
    try:
        import pronouncing  # pip install pronouncing
        return ("pronouncing", pronouncing)
    except Exception:
        return (None, None)


def get_phones(word, backend):
    name, lib = backend
    if name == "pronouncing":
        ps = lib.phones_for_word(word.lower())
        if not ps:
            return None
        return ps[0].split()
    return None


def infer_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    proj_w = sd["proj.weight"]  # (out_dim, hidden)
    out_dim = proj_w.shape[0]
    hidden = proj_w.shape[1]

    wih0 = sd["lstm.weight_ih_l0"]  # (4*hidden, in_dim)
    in_dim = wih0.shape[1]

    layer_keys = [k for k in sd.keys() if k.startswith("lstm.weight_ih_l")]
    layers = len(layer_keys)

    model = LSTMFrameSRV(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden=hidden,
        layers=layers,
        dropout=0.0,
        bidirectional=False,
    )

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    print(f"[MODEL] inferred in_dim={in_dim}, hidden={hidden}, layers={layers}, out_dim={out_dim}")
    return model


def pass1_count_words(manifest_path: Path, subset: str):
    cnt = Counter()
    n_utt = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("subset") != subset:
                continue
            n_utt += 1
            for w, sf, ef in obj["words"]:
                cnt[w] += 1
    print(f"[PASS1] subset={subset} utt={n_utt} unique_words={len(cnt)}")
    return cnt


def select_vocab(counter: Counter, min_count: int, topN: int):
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:topN]
    vocab = {w for w, _ in items}
    print(f"[VOCAB] min_count={min_count} -> kept={len(vocab)} (topN={topN})")
    return vocab, dict(items)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


@torch.no_grad()
def extract_embeddings(
    model,
    manifest_path: Path,
    subset: str,
    vocab: set,
    max_tokens_per_word: int,
    tail_frames: int,
    mid_margin_frames: int,
    device: str,
):
    """
    Stream through manifest, run model per utterance, aggregate type embeddings:
      - tail_mean: mean(h[max(ef-tail, sf):ef])
      - last_frame: h[ef-1]
      - delta: h[ef-1] - h[sf]
      - mid_mean: mean(h[sf+margin : ef-margin]) with fallback if too short
    """
    sum_tail = {}
    sum_last = {}
    sum_delta = {}
    sum_mid = {}
    cnt_word = defaultdict(int)

    n_utt = 0
    n_tok_used = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("subset") != subset:
                continue

            coch_path = Path(obj["coch_path"])
            if not coch_path.exists():
                raise FileNotFoundError(f"coch_path not found: {coch_path}")

            coch = np.load(coch_path)  # (64, T)
            if coch.ndim != 2 or coch.shape[0] != 64:
                raise ValueError(f"Unexpected coch shape {coch.shape} for {coch_path}")

            feats = torch.from_numpy(coch.astype("float32")).T.contiguous()  # (T,64)
            T = feats.shape[0]

            x = feats.unsqueeze(0).to(device)
            x_lens = torch.tensor([T], dtype=torch.long, device=device)

            _, h, _ = model.forward_with_hidden(x, x_lens)  # h: (1,T,H)
            h = h[0, :T].detach().cpu()  # (T,H)

            for w, sf, ef in obj["words"]:
                if w not in vocab:
                    continue
                if cnt_word[w] >= max_tokens_per_word:
                    continue

                sf = int(sf); ef = int(ef)
                if ef <= 0 or sf >= T:
                    continue
                sf = max(sf, 0)
                ef = min(ef, T)
                if ef <= sf:
                    continue

                # -------- 1) tail_mean --------
                i0 = max(ef - tail_frames, sf)
                tail_vec = h[i0:ef].mean(dim=0)

                # -------- 2) last_frame --------
                last_vec = h[ef - 1]

                # -------- 3) delta (cumulative) --------
                # 注意：delta更敏感于sf定位误差，但可以刻画“这个词期间状态变化”
                delta_vec = h[ef - 1] - h[sf]

                # -------- 4) mid_mean (avoid boundary transition) --------
                m = int(mid_margin_frames)
                a = sf + m
                b = ef - m
                if b > a:
                    mid_vec = h[a:b].mean(dim=0)
                else:
                    # 词太短，退化成整段平均（至少别空）
                    mid_vec = h[sf:ef].mean(dim=0)

                # normalize token vectors (cosine-friendly)
                tail_vec = _normalize(tail_vec)
                last_vec = _normalize(last_vec)
                delta_vec = _normalize(delta_vec)
                mid_vec  = _normalize(mid_vec)

                if w not in sum_tail:
                    sum_tail[w] = tail_vec.clone()
                    sum_last[w] = last_vec.clone()
                    sum_delta[w] = delta_vec.clone()
                    sum_mid[w] = mid_vec.clone()
                else:
                    sum_tail[w] += tail_vec
                    sum_last[w] += last_vec
                    sum_delta[w] += delta_vec
                    sum_mid[w] += mid_vec

                cnt_word[w] += 1
                n_tok_used += 1

            n_utt += 1
            if n_utt % 200 == 0:
                have = sum(1 for ww in vocab if cnt_word[ww] > 0)
                print(f"[PASS2] utt={n_utt} tokens_used={n_tok_used} words_touched={have}/{len(vocab)}")

    words = sorted([w for w in vocab if cnt_word[w] > 0])
    counts = []
    tail_mat = []
    last_mat = []
    delta_mat = []
    mid_mat = []

    for w in words:
        c = cnt_word[w]
        v_tail  = _normalize(sum_tail[w] / c)
        v_last  = _normalize(sum_last[w] / c)
        v_delta = _normalize(sum_delta[w] / c)
        v_mid   = _normalize(sum_mid[w] / c)

        tail_mat.append(v_tail.numpy())
        last_mat.append(v_last.numpy())
        delta_mat.append(v_delta.numpy())
        mid_mat.append(v_mid.numpy())
        counts.append(c)

    tail_mat  = np.stack(tail_mat, axis=0)
    last_mat  = np.stack(last_mat, axis=0)
    delta_mat = np.stack(delta_mat, axis=0)
    mid_mat   = np.stack(mid_mat, axis=0)
    counts = np.array(counts, dtype=np.int32)

    print(f"[DONE] type_vocab={len(words)} (nonzero) total_tokens_used={n_tok_used}")
    return words, counts, tail_mat, last_mat, delta_mat, mid_mat


def topk_neighbors(query, words, mat, k=15):
    if query not in words:
        return None
    idx = words.index(query)
    q = mat[idx]
    sims = mat @ q
    order = np.argsort(-sims)
    res = []
    for j in order:
        if j == idx:
            continue
        res.append((words[j], float(sims[j])))
        if len(res) >= k:
            break
    return res


def main():#C:\linux_project\LENS\runs\librispeech_LSTM_WORD_srv_baseline_s000\ckpt\best.pt
    ap = argparse.ArgumentParser()#C:\linux_project\LENS\runs\librispeech_LSTM_PHONE_srv_modified_loss_s000\ckpt\best.pt
    ap.add_argument("--manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")
    ap.add_argument("--ckpt", type=str, default=r"C:\linux_project\LENS\runs\librispeech_LSTM_PHONE_srv_modified_loss_s000\ckpt\best.pt")
    ap.add_argument("--subset", type=str, default="train-clean-100")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--topN", type=int, default=5000)
    ap.add_argument("--max_tokens_per_word", type=int, default=50)
    ap.add_argument("--tail_frames", type=int, default=10)
    ap.add_argument("--mid_margin_frames", type=int, default=3, help="frames to exclude near sf/ef for mid_mean")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--queries", type=str, default="the,that,this,there")
    ap.add_argument("--out", type=str, default="round1_word_embeds_train100_4mats_phoneme.npz")#round1_word_embeds_train100_4mats.npz  round1_word_embeds_devclean_4mats.npz
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.ckpt)

    model = infer_model_from_ckpt(ckpt_path, args.device)

    cnt = pass1_count_words(manifest_path, args.subset)
    vocab, _ = select_vocab(cnt, args.min_count, args.topN)

    words, counts, tail_mat, last_mat, delta_mat, mid_mat = extract_embeddings(
        model=model,
        manifest_path=manifest_path,
        subset=args.subset,
        vocab=vocab,
        max_tokens_per_word=args.max_tokens_per_word,
        tail_frames=args.tail_frames,
        mid_margin_frames=args.mid_margin_frames,
        device=args.device,
    )

    np.savez(
        args.out,
        words=np.array(words, dtype=object),
        counts=counts,
        tail_mat=tail_mat,
        last_mat=last_mat,
        delta_mat=delta_mat,
        mid_mat=mid_mat,
        subset=args.subset,
        min_count=args.min_count,
        topN=args.topN,
        max_tokens_per_word=args.max_tokens_per_word,
        tail_frames=args.tail_frames,
        mid_margin_frames=args.mid_margin_frames,
    )
    print(f"[SAVE] -> {args.out}")

    backend = try_get_pronunciation_backend()
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    mats = [
        ("tail_mean", tail_mat),
        ("last_frame", last_mat),
        ("delta", delta_mat),
        ("mid_mean", mid_mat),
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"[QUERY] {q}")

        q_phones = get_phones(q, backend)
        for name, mat in mats:
            nn = topk_neighbors(q, words, mat, k=15)
            if nn is None:
                print(f"  [{name}] word not in selected vocab (maybe rare in {args.subset}).")
                continue

            print(f"  [{name}] top neighbors:")
            for w2, sim in nn:
                spell_ed = levenshtein(q.lower(), w2.lower())

                ph_ed = None
                if backend[0] is not None:
                    p2 = get_phones(w2, backend)
                    if q_phones is not None and p2 is not None:
                        ph_ed = levenshtein(q_phones, p2)

                if ph_ed is None:
                    print(f"    {w2:15s} sim={sim: .3f}  spellED={spell_ed:2d}")
                else:
                    print(f"    {w2:15s} sim={sim: .3f}  phoneED={ph_ed:2d}  spellED={spell_ed:2d}")


if __name__ == "__main__":
    main()