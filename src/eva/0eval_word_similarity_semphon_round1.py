# eva/0eval_word_similarity_semphon_round1.py
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

from network.lstm_frame_srv_semantic import LSTMFrameSRVSemantic

IGNORE = -100


# ------------------------- utilities -------------------------
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


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


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


def load_glove_like(emb_path: Path, wanted_words: set[str], dim: int | None = None):
    """
    Load word vectors from a text file: "word val1 val2 ...".
    Only loads vectors for wanted_words (case-insensitive).
    Returns dict: word -> np.ndarray
    """
    vecs = {}
    wanted_lower = {w.lower() for w in wanted_words}
    with emb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) <= 2:
                continue
            w = parts[0].lower()
            if w not in wanted_lower:
                continue
            vals = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if dim is not None and vals.shape[0] != dim:
                continue
            # normalize
            vals = vals / (np.linalg.norm(vals) + 1e-8)
            vecs[w] = vals
    return vecs


def sem_cos(w1: str, w2: str, emb: dict):
    v1 = emb.get(w1.lower())
    v2 = emb.get(w2.lower())
    if v1 is None or v2 is None:
        return None
    return float(np.dot(v1, v2))


# ------------------------- model inference -------------------------
def infer_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # ---- infer frame lstm dims ----
    # frame_proj.weight: (srv_dim, frame_hidden)
    srv_dim = sd["frame_proj.weight"].shape[0]
    frame_hidden = sd["frame_proj.weight"].shape[1]

    # frame_lstm.weight_ih_l0: (4*H, in_dim)
    in_dim = sd["frame_lstm.weight_ih_l0"].shape[1]

    frame_layer_keys = [k for k in sd.keys() if k.startswith("frame_lstm.weight_ih_l")]
    frame_layers = len(frame_layer_keys)

    # ---- infer word head dims ----
    # word_proj.0.weight: (word_rep_dim, frame_hidden)
    word_rep_dim = sd["word_proj.0.weight"].shape[0]

    # word_lstm.weight_ih_l0: (4*Hw, word_rep_dim)
    word_lstm_hidden = sd["word_lstm.weight_ih_l0"].shape[0] // 4
    word_layer_keys = [k for k in sd.keys() if k.startswith("word_lstm.weight_ih_l")]
    word_lstm_layers = len(word_layer_keys)

    model = LSTMFrameSRVSemantic(
        in_dim=in_dim,
        srv_dim=srv_dim,
        frame_hidden=frame_hidden,
        frame_layers=frame_layers,
        dropout=0.0,
        word_rep_dim=word_rep_dim,
        word_lstm_hidden=word_lstm_hidden,
        word_lstm_layers=word_lstm_layers,
        rep_dropout=0.0,
        word_dropout_p=0.0,
        rep_noise_std=0.0,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    print(
        f"[MODEL] inferred in_dim={in_dim} srv_dim={srv_dim} "
        f"frame_hidden={frame_hidden} frame_layers={frame_layers} "
        f"word_rep_dim={word_rep_dim} word_lstm_hidden={word_lstm_hidden} word_lstm_layers={word_lstm_layers}"
    )
    return model


# ------------------------- data pass -------------------------
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


@torch.no_grad()
def extract_embeddings_semphon(
    model: LSTMFrameSRVSemantic,
    manifest_path: Path,
    subset: str,
    vocab: set,
    max_tokens_per_word: int,
    tail_frames: int,
    mid_margin_frames: int,
    device: str,
):
    """
    Extract:
      - phon_* : from frame_lstm output h(t)  -> token aggregation (tail/last/delta/mid)
      - sem_*  : from word_lstm contextual states c_i and bottleneck r_i

    Save type embeddings by averaging over up to max_tokens_per_word occurrences per word.
    """
    # type-level accumulators
    sum_phon_tail = {}
    sum_phon_last = {}
    sum_phon_delta = {}
    sum_phon_mid = {}

    sum_sem_r = {}
    sum_sem_ctx_post = {}
    sum_sem_ctx_pre = {}

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

            # --- build token list (only vocab words & token cap) ---
            words_tok = []
            starts = []
            ends = []
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
                words_tok.append(w)
                starts.append(sf)
                ends.append(ef)

            # if no usable tokens, skip forward to save time
            if len(words_tok) == 0:
                n_utt += 1
                continue

            # --- forward frame_lstm to get h(t) ---
            x = feats.unsqueeze(0).to(device)
            x_lens = torch.tensor([T], dtype=torch.long, device=device)

            _, h, out_lens = model.forward_with_hidden(x, x_lens)  # h: (1,T,H)
            h = h[0, :T].detach().cpu()  # (T,H)

            # --- compute phonetic token vectors from h(t) ---
            m = int(mid_margin_frames)
            for w, sf, ef in zip(words_tok, starts, ends):
                # 1) tail_mean
                i0 = max(ef - tail_frames, sf)
                tail_vec = h[i0:ef].mean(dim=0)

                # 2) last_frame
                last_vec = h[ef - 1]

                # 3) delta
                delta_vec = h[ef - 1] - h[sf]

                # 4) mid_mean (avoid boundary transitions)
                a = sf + m
                b = ef - m
                if b > a:
                    mid_vec = h[a:b].mean(dim=0)
                else:
                    mid_vec = h[sf:ef].mean(dim=0)

                tail_vec = _normalize(tail_vec)
                last_vec = _normalize(last_vec)
                delta_vec = _normalize(delta_vec)
                mid_vec = _normalize(mid_vec)

                if w not in sum_phon_tail:
                    sum_phon_tail[w] = tail_vec.clone()
                    sum_phon_last[w] = last_vec.clone()
                    sum_phon_delta[w] = delta_vec.clone()
                    sum_phon_mid[w] = mid_vec.clone()
                else:
                    sum_phon_tail[w] += tail_vec
                    sum_phon_last[w] += last_vec
                    sum_phon_delta[w] += delta_vec
                    sum_phon_mid[w] += mid_vec

                cnt_word[w] += 1
                n_tok_used += 1

            # --- semantic: run word-level LSTM on word reps (same extraction logic as model) ---
            W = len(words_tok)
            word_starts = torch.tensor(starts, dtype=torch.long).view(1, W).to(device)
            word_ends = torch.tensor(ends, dtype=torch.long).view(1, W).to(device)
            word_lens = torch.tensor([W], dtype=torch.long).to(device)

            # rebuild reps_h using model's internal method, but on GPU tensors
            # note: need h on device for this; we can re-use h_device from forward output
            # easiest: re-run forward_with_hidden but keep h_device, so do it here:
            _, h_dev, out_lens_dev = model.forward_with_hidden(x, x_lens)  # h_dev: (1,T,H)

            reps_h = model._extract_word_reps(
                h=h_dev,
                out_lens=out_lens_dev,
                word_starts=word_starts,
                word_ends=word_ends,
                word_lens=word_lens,
                tail_frames=tail_frames,
            )  # (1,W,H)

            r = model.word_proj(reps_h)  # (1,W,word_rep_dim)
            # word_lstm
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                r, word_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = model.word_lstm(packed)
            c_post, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=W)  # (1,W,Hw)

            r = r[0].detach().cpu()               # (W,Dr)
            c_post = c_post[0].detach().cpu()     # (W,Hw)

            # define c_pre[j] = c_post[j-1], c_pre[0]=0
            c_pre = torch.zeros_like(c_post)
            if W > 1:
                c_pre[1:] = c_post[:-1]

            # aggregate semantic vectors by word type
            for j, w in enumerate(words_tok):
                # same cap already enforced by cnt_word during phonetic part, but semantic should match same used tokens.
                # To keep it consistent, we only aggregate semantic for words that we actually counted (<=max_tokens_per_word).
                # We already incremented cnt_word above, so just use those occurrences.
                rv = _normalize(r[j])
                cv_post = _normalize(c_post[j])
                cv_pre = _normalize(c_pre[j])

                if w not in sum_sem_r:
                    sum_sem_r[w] = rv.clone()
                    sum_sem_ctx_post[w] = cv_post.clone()
                    sum_sem_ctx_pre[w] = cv_pre.clone()
                else:
                    sum_sem_r[w] += rv
                    sum_sem_ctx_post[w] += cv_post
                    sum_sem_ctx_pre[w] += cv_pre

            n_utt += 1
            if n_utt % 200 == 0:
                have = sum(1 for ww in vocab if cnt_word[ww] > 0)
                print(f"[PASS2] utt={n_utt} tokens_used={n_tok_used} words_touched={have}/{len(vocab)}")

    # finalize matrices
    words = sorted([w for w in vocab if cnt_word[w] > 0])
    counts = []
    phon_tail = []
    phon_last = []
    phon_delta = []
    phon_mid = []

    sem_r = []
    sem_ctx_post = []
    sem_ctx_pre = []

    for w in words:
        c = cnt_word[w]
        counts.append(c)

        v_tail = _normalize(sum_phon_tail[w] / c)
        v_last = _normalize(sum_phon_last[w] / c)
        v_delta = _normalize(sum_phon_delta[w] / c)
        v_mid = _normalize(sum_phon_mid[w] / c)

        phon_tail.append(v_tail.numpy())
        phon_last.append(v_last.numpy())
        phon_delta.append(v_delta.numpy())
        phon_mid.append(v_mid.numpy())

        # semantic (note: some extremely rare cases could have no sem agg; guard)
        vr = _normalize(sum_sem_r[w] / c)
        vp = _normalize(sum_sem_ctx_post[w] / c)
        vq = _normalize(sum_sem_ctx_pre[w] / c)

        sem_r.append(vr.numpy())
        sem_ctx_post.append(vp.numpy())
        sem_ctx_pre.append(vq.numpy())

    out = {
        "words": np.array(words, dtype=object),
        "counts": np.array(counts, dtype=np.int32),

        # phonetic mats (frame_lstm)
        "tail_mat": np.stack(phon_tail, axis=0).astype(np.float32),
        "last_mat": np.stack(phon_last, axis=0).astype(np.float32),
        "delta_mat": np.stack(phon_delta, axis=0).astype(np.float32),
        "mid_mat": np.stack(phon_mid, axis=0).astype(np.float32),

        # semantic mats (word_lstm)
        "sem_r_mat": np.stack(sem_r, axis=0).astype(np.float32),
        "sem_ctx_post_mat": np.stack(sem_ctx_post, axis=0).astype(np.float32),
        "sem_ctx_pre_mat": np.stack(sem_ctx_pre, axis=0).astype(np.float32),
    }

    print(f"[DONE] type_vocab={len(words)} total_tokens_used={n_tok_used}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")
    ap.add_argument("--ckpt", type=str, default=r"C:\linux_project\LENS\runs\librispeech_LSTM_WORD_SEM_srv_semantic_hier_s000\ckpt\last.pt")
    ap.add_argument("--subset", type=str, default="test-clean")

    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--topN", type=int, default=20000)
    ap.add_argument("--max_tokens_per_word", type=int, default=50)

    ap.add_argument("--tail_frames", type=int, default=10)
    ap.add_argument("--mid_margin_frames", type=int, default=3)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--queries", type=str, default="the,that,this,there,dog,cat,man,woman")
    ap.add_argument("--out", type=str, default="round1_word_embeds_devclean_SEMPHON.npz")

    # optional: external semantic embedding file (glove/fasttext txt)
    ap.add_argument("--sem_emb", type=str, default="", help="path to glove-like txt; used only for printing semanticSim in query neighbors")
    ap.add_argument("--sem_emb_dim", type=int, default=0, help="if >0, enforce embedding dim")

    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.ckpt)

    model = infer_model_from_ckpt(ckpt_path, args.device)

    cnt = pass1_count_words(manifest_path, args.subset)
    
    print("dog count =", cnt.get("dog", 0), "DOG count =", cnt.get("DOG", 0))
    print("cat count =", cnt.get("cat", 0), "CAT count =", cnt.get("CAT", 0))
    
    vocab, _ = select_vocab(cnt, args.min_count, args.topN)

    pack = extract_embeddings_semphon(
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
        **pack,
        subset=args.subset,
        min_count=args.min_count,
        topN=args.topN,
        max_tokens_per_word=args.max_tokens_per_word,
        tail_frames=args.tail_frames,
        mid_margin_frames=args.mid_margin_frames,
    )
    print(f"[SAVE] -> {args.out}")

    # load semantic embedding (optional)
    sem_emb = None
    if args.sem_emb.strip():
        emb_path = Path(args.sem_emb)
        if emb_path.exists():
            dim = args.sem_emb_dim if args.sem_emb_dim > 0 else None
            sem_emb = load_glove_like(emb_path, wanted_words=set(pack["words"].tolist()), dim=dim)
            print(f"[SEM_EMB] loaded {len(sem_emb)} vectors from {emb_path}")
        else:
            print(f"[SEM_EMB] file not found: {emb_path}")

    backend = try_get_pronunciation_backend()
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    # mats to inspect
    mats = [
        ("PHON_tail_mean", pack["tail_mat"]),
        ("PHON_last_frame", pack["last_mat"]),
        ("PHON_delta", pack["delta_mat"]),
        ("PHON_mid_mean", pack["mid_mat"]),
        ("SEM_r_bottleneck", pack["sem_r_mat"]),
        ("SEM_ctx_post (word_lstm)", pack["sem_ctx_post_mat"]),
        ("SEM_ctx_pre (shifted)", pack["sem_ctx_pre_mat"]),
    ]
    words = pack["words"].tolist()

    for q in queries:
        print("\n" + "=" * 90)
        print(f"[QUERY] {q}")

        q_phones = get_phones(q, backend)

        for name, mat in mats:
            nn = topk_neighbors(q, words, mat, k=15)
            if nn is None:
                print(f"  [{name}] word not in selected vocab.")
                continue

            print(f"  [{name}] top neighbors:")
            for w2, sim in nn:
                spell_ed = levenshtein(q.lower(), w2.lower())

                # phonetic ED (if available)
                ph_ed = None
                if backend[0] is not None:
                    p2 = get_phones(w2, backend)
                    if q_phones is not None and p2 is not None:
                        ph_ed = levenshtein(q_phones, p2)

                # semantic cosine from external embedding
                ssim = None
                if sem_emb is not None:
                    ssim = sem_cos(q, w2, sem_emb)

                msg = f"    {w2:15s} sim={sim: .3f}"
                if ph_ed is not None:
                    msg += f"  phoneED={ph_ed:2d}"
                msg += f"  spellED={spell_ed:2d}"
                if ssim is not None:
                    msg += f"  semCos(ext)={ssim: .3f}"
                print(msg)


if __name__ == "__main__":
    main()