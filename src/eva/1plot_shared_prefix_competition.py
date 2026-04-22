import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from network.lstm_frame_srv import LSTMFrameSRV


# ---------- utilities ----------
def try_get_pronouncing():
    try:
        import pronouncing
        return pronouncing
    except Exception:
        return None

def phones_for(word, pronouncing):
    ps = pronouncing.phones_for_word(word.lower())
    if not ps:
        return None
    return ps[0].split()

def shared_prefix_len(p1, p2):
    n = min(len(p1), len(p2))
    k = 0
    for i in range(n):
        if p1[i] != p2[i]:
            break
        k += 1
    return k

def infer_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    out_dim, hidden = sd["proj.weight"].shape
    in_dim = sd["lstm.weight_ih_l0"].shape[1]
    layers = len([k for k in sd.keys() if k.startswith("lstm.weight_ih_l")])

    model = LSTMFrameSRV(
        in_dim=in_dim, out_dim=out_dim,
        hidden=hidden, layers=layers,
        dropout=0.0, bidirectional=False
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"[MODEL] in_dim={in_dim} hidden={hidden} layers={layers} out_dim={out_dim}")
    return model

def load_embeddings_npz(npz_path: Path, method: str):
    d = np.load(npz_path, allow_pickle=True)
    words = d["words"].tolist()
    counts = d["counts"].astype(int)
    key = {"mid":"mid_mat","tail":"tail_mat","last":"last_mat","delta":"delta_mat"}[method]
    E = d[key].astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    w2i = {w:i for i,w in enumerate(words)}
    print(f"[EMB] V={len(words)} method={method} npz={npz_path.name}")
    return words, counts, E, w2i

def scan_manifest_word_counts(manifest_path: Path, subset: str):
    cnt = Counter()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("subset") != subset:
                continue
            for w, sf, ef in obj["words"]:
                cnt[w] += 1
    return cnt


# ---------- main computation ----------
@torch.no_grad()
def compute_shared_prefix_curves(
    model,
    manifest_path: Path,
    subset: str,
    targets: list,
    words_lex: list,
    w2i: dict,
    E_np: np.ndarray,
    phones: dict,
    device: str,
    env_sr: int,
    max_ms: int,
    nmax: int,
    max_per_group: int,
    temp: float,
    correct_only: bool,
    stop_at_offset: bool,
):
    """
    Returns curves:
      target: (L,)
      shared_k: dict k->(L,)
      unrelated: (L,)
    We compute mean probability per word in each group, after sampling up to max_per_group words per group per target.
    """
    E = torch.from_numpy(E_np)  # (V,H) on CPU
    V, H = E.shape
    L = int(round(max_ms * env_sr / 1000.0))
    L = max(1, L)

    # accumulators
    sum_target = np.zeros(L, dtype=np.float64)
    cnt_target = np.zeros(L, dtype=np.int64)

    sum_shared = {k: np.zeros(L, dtype=np.float64) for k in range(1, nmax+1)}
    cnt_shared = {k: np.zeros(L, dtype=np.int64)  for k in range(1, nmax+1)}

    sum_unrel = np.zeros(L, dtype=np.float64)
    cnt_unrel = np.zeros(L, dtype=np.int64)

    used_tokens = 0
    kept_tokens = 0

    rng_global = np.random.default_rng(0)

    # Precompute for speed: list of lex words with phones
    lex_has_phone = [w for w in words_lex if phones.get(w) is not None]

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            obj = json.loads(line)
            if obj.get("subset") != subset:
                continue

            # quick skip
            utt_words = [w for (w, sf, ef) in obj["words"]]
            if not any(w in targets for w in utt_words):
                continue

            coch = np.load(obj["coch_path"])  # (64,T)
            feats = torch.from_numpy(coch.astype("float32")).T.contiguous()  # (T,64)
            T = feats.shape[0]

            x = feats.unsqueeze(0).to(device)
            x_lens = torch.tensor([T], dtype=torch.long, device=device)
            _, h, _ = model.forward_with_hidden(x, x_lens)
            h = h[0, :T].detach().cpu()  # (T,H)
            h = h / (h.norm(dim=1, keepdim=True) + 1e-8)

            for w, sf, ef in obj["words"]:
                if w not in targets:
                    continue
                if w not in w2i:
                    continue
                pw = phones.get(w)
                if pw is None:
                    continue

                sf = int(sf); ef = int(ef)
                if ef <= 0 or sf >= T:
                    continue
                sf = max(sf, 0); ef = min(ef, T)
                if ef <= sf:
                    continue

                used_tokens += 1

                # correctness filter: use frame at ef-1
                if correct_only:
                    h_last = h[ef-1]  # (H,)
                    sims = (E @ h_last)  # (V,)
                    pred = int(torch.argmax(sims).item())
                    if pred != w2i[w]:
                        continue

                # build groups by shared prefix length
                groups = {k: [] for k in range(1, nmax+1)}
                unrelated = []

                # choose candidate lex words (phones available)
                # (optional speed) sample a subset if lexicon huge
                # Here we just iterate lex_has_phone (OK for a few thousand words; for >50k consider sampling)
                for ww in lex_has_phone:
                    if ww == w:
                        continue
                    pp = phones[ww]
                    s = shared_prefix_len(pw, pp)
                    if s == 0:
                        unrelated.append(ww)
                    elif 1 <= s <= nmax:
                        groups[s].append(ww)

                # require at least some 1-shared competitors
                if len(groups[1]) < 3 or len(unrelated) < 20:
                    continue

                # sample fixed number per group (to make mean-prob curves not collapse)
                rng = np.random.default_rng(abs(hash((w, obj["utt_id"]))) % (2**32))
                idx_target = [w2i[w]]

                idx_groups = {}
                for k in range(1, nmax+1):
                    cand = groups[k]
                    if len(cand) == 0:
                        idx_groups[k] = []
                        continue
                    rng.shuffle(cand)
                    cand = cand[:max_per_group]
                    idx_groups[k] = [w2i[x] for x in cand if x in w2i]

                rng.shuffle(unrelated)
                unrel = unrelated[:max_per_group]
                idx_unrel = [w2i[x] for x in unrel if x in w2i]

                # union candidates for softmax
                union = idx_target[:]
                for k in range(1, nmax+1):
                    union += idx_groups[k]
                union += idx_unrel
                union = list(dict.fromkeys(union))
                if len(union) < 10:
                    continue

                # time window
                if stop_at_offset:
                    seg_len = min(L, ef - sf)
                else:
                    seg_len = min(L, T - sf)

                h_seg = h[sf:sf+seg_len]  # (seg_len,H)
                Euni = E[union]           # (U,H)
                logits = (h_seg @ Euni.T) / temp
                probs = torch.softmax(logits, dim=1)  # (seg_len,U)

                pos = {idx: j for j, idx in enumerate(union)}

                # target probability
                pt = probs[:, pos[idx_target[0]]].numpy()
                sum_target[:seg_len] += pt
                cnt_target[:seg_len] += 1

                # each shared-k: mean prob over words in that group
                for k in range(1, nmax+1):
                    idxs = idx_groups[k]
                    if len(idxs) == 0:
                        continue
                    cols = [pos[i] for i in idxs if i in pos]
                    if len(cols) == 0:
                        continue
                    pk = probs[:, cols].mean(dim=1).numpy()
                    sum_shared[k][:seg_len] += pk
                    cnt_shared[k][:seg_len] += 1

                # unrelated: mean prob
                if len(idx_unrel) > 0:
                    cols = [pos[i] for i in idx_unrel if i in pos]
                    if len(cols) > 0:
                        pu = probs[:, cols].mean(dim=1).numpy()
                        sum_unrel[:seg_len] += pu
                        cnt_unrel[:seg_len] += 1

                kept_tokens += 1

            if line_idx % 500 == 0:
                print(f"[SCAN] lines={line_idx} used_tokens={used_tokens} kept_tokens={kept_tokens}")

    def finalize(s, c):
        y = s / np.maximum(c, 1)
        y[c == 0] = np.nan
        return y.astype(np.float32)

    curves = {
        "target": finalize(sum_target, cnt_target),
        "unrelated": finalize(sum_unrel, cnt_unrel),
        "shared": {k: finalize(sum_shared[k], cnt_shared[k]) for k in range(1, nmax+1)}
    }
    print(f"[DONE] used_tokens={used_tokens} kept_tokens={kept_tokens} correct_only={correct_only}")
    return curves, L


def plot(curves, L, env_sr, out_png):
    t_ms = np.arange(L) * (1000.0 / env_sr)

    plt.figure(figsize=(11, 6))
    plt.plot(t_ms, curves["target"], linewidth=4, label="target")
    plt.plot(t_ms, curves["unrelated"], linestyle=":", linewidth=3, label="unrelated")

    for k, y in curves["shared"].items():
        plt.plot(t_ms, y, linewidth=3, label=f"{k} shared")

    plt.xlabel("Time (ms)")
    plt.ylabel("Mean probability")
    plt.title("Shared-prefix phoneme competition")
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")
    ap.add_argument("--ckpt", type=str, default=r"C:\linux_project\LENS\runs\librispeech_LSTM_WORD_srv_baseline_s000\ckpt\best.pt")
    ap.add_argument("--npz", type=str, default=r"C:\linux_project\LENS\round1_word_embeds_devclean_4mats.npz")
    ap.add_argument("--subset", type=str, default="dev-clean")
    ap.add_argument("--method", type=str, default="mid", choices=["mid","tail","last","delta"])
    ap.add_argument("--targets", type=str, default="AUTO", help="comma list or AUTO")
    ap.add_argument("--n_targets", type=int, default=30)
    ap.add_argument("--min_target_count", type=int, default=20)
    ap.add_argument("--min_target_phones", type=int, default=5, help="targets must have >= this many phones")
    ap.add_argument("--nmax", type=int, default=7)
    ap.add_argument("--max_per_group", type=int, default=30)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--env_sr", type=int, default=100)
    ap.add_argument("--max_ms", type=int, default=500)
    ap.add_argument("--correct_only", type=bool, default=True)
    ap.add_argument("--stop_at_offset", action="store_true", help="only plot within word duration (sf->ef)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_png", type=str, default=r"images/shared_prefix_competition.png")
    args = ap.parse_args()

    pronouncing = try_get_pronouncing()
    if pronouncing is None:
        raise SystemExit("Please install pronouncing: pip install pronouncing")

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.ckpt)
    npz_path = Path(args.npz)

    model = infer_model_from_ckpt(ckpt_path, args.device)
    words_lex, counts, E, w2i = load_embeddings_npz(npz_path, args.method)

    # phones for lexicon
    phones = {w: phones_for(w, pronouncing) for w in words_lex}

    # choose targets
    if args.targets.strip().upper() == "AUTO":
        cnt = scan_manifest_word_counts(manifest_path, args.subset)
        cands = []
        for w, c in cnt.items():
            if c < args.min_target_count: 
                continue
            if w not in w2i: 
                continue
            pw = phones.get(w)
            if pw is None or len(pw) < args.min_target_phones:
                continue
            cands.append((w, c))
        cands.sort(key=lambda x: x[1], reverse=True)
        targets = [w for w,_ in cands[:args.n_targets]]
        print(f"[TARGETS] AUTO picked {len(targets)}:", targets[:10], "...")
        if len(targets) == 0:
            raise SystemExit("AUTO found no targets. Lower thresholds or use bigger lexicon npz.")
    else:
        targets = [x.strip() for x in args.targets.split(",") if x.strip()]
        print(f"[TARGETS] manual: {targets}")

    curves, L = compute_shared_prefix_curves(
        model=model,
        manifest_path=manifest_path,
        subset=args.subset,
        targets=targets,
        words_lex=words_lex,
        w2i=w2i,
        E_np=E,
        phones=phones,
        device=args.device,
        env_sr=args.env_sr,
        max_ms=args.max_ms,
        nmax=args.nmax,
        max_per_group=args.max_per_group,
        temp=args.temp,
        correct_only=args.correct_only,
        stop_at_offset=args.stop_at_offset,
    )

    plot(curves, L, args.env_sr, args.out_png)


if __name__ == "__main__":
    main()