import argparse, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from network.lstm_frame_srv import LSTMFrameSRV


# ----------------- util -----------------
def levenshtein(a, b):
    if a == b: return 0
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[-1]

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

def infer_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    proj_w = sd["proj.weight"]  # (out_dim, hidden)
    out_dim = proj_w.shape[0]
    hidden = proj_w.shape[1]
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
    key = {
        "mid": "mid_mat",
        "tail": "tail_mat",
        "last": "last_mat",
        "delta": "delta_mat",
    }[method]
    E = d[key].astype(np.float32)  # (V,H) already normalized in your pipeline
    # ensure normalized
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    w2i = {w:i for i,w in enumerate(words)}
    meta = {k: (d[k].item() if d[k].shape==() else d[k].tolist())
            for k in d.files if k not in ("words","counts","tail_mat","last_mat","delta_mat","mid_mat")}
    print(f"[EMB] V={len(words)} method={method} npz={npz_path.name}")
    return words, counts, E, w2i, meta

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


# ----------------- competitor sets -----------------
def build_competitors(words, w2i, pronouncing, prefix_n=2, suffix_n=2, unrelated_min_ed=4, max_unrelated=50):
    phone = {}
    for w in words:
        ph = phones_for(w, pronouncing)
        phone[w] = ph

    # precompute prefix/suffix signatures
    pref = {}
    suf  = {}
    for w, ph in phone.items():
        if ph is None or len(ph) < max(prefix_n, suffix_n):
            pref[w] = None
            suf[w] = None
        else:
            pref[w] = tuple(ph[:prefix_n])
            suf[w]  = tuple(ph[-suffix_n:])

    def get_sets(target):
        if target not in w2i:
            return None
        ph_t = phone[target]
        if ph_t is None or len(ph_t) < max(prefix_n, suffix_n):
            return None

        cohort = [w for w in words if w != target and pref[w] == pref[target]]
        rhyme  = [w for w in words if w != target and suf[w]  == suf[target]]

        # unrelated: far in phone ED + not in cohort/rhyme
        bad = set(cohort) | set(rhyme) | {target}
        cand = []
        for w in words:
            if w in bad:
                continue
            ph = phone[w]
            if ph is None:
                continue
            if levenshtein(ph_t, ph) >= unrelated_min_ed:
                cand.append(w)

        # sample unrelated to match size (or cap)
        rng = np.random.default_rng(abs(hash(target)) % (2**32))
        rng.shuffle(cand)
        if max_unrelated is not None:
            cand = cand[:max_unrelated]
        return cohort, rhyme, cand

    return phone, get_sets


# ----------------- timecourse extraction -----------------
@torch.no_grad()
def compute_timecourses(
    model, manifest_path: Path, subset: str,
    targets: list,
    words, w2i, E_np: np.ndarray,
    competitor_getter,
    device: str,
    env_sr: int = 100,
    max_ms: int = 600,
    correct_only: bool = True,
    temp: float = 0.07,
):
    """
    Returns dict of curves (cosine + softmax-mass) averaged over tokens:
      cos_curves[group] : (L,)
      prob_curves[group]: (L,)
    groups: target, cohort, rhyme, unrelated
    """
    E = torch.from_numpy(E_np).to("cpu")  # (V,H)
    V, H = E.shape
    L = int(round(max_ms * env_sr / 1000.0))
    L = max(1, L)

    # accumulators (sum over tokens, count per time)
    cos_sum = {g: np.zeros(L, dtype=np.float64) for g in ["target","cohort","rhyme","unrelated"]}
    cos_cnt = {g: np.zeros(L, dtype=np.int64)  for g in ["target","cohort","rhyme","unrelated"]}
    pr_sum  = {g: np.zeros(L, dtype=np.float64) for g in ["target","cohort","rhyme","unrelated"]}
    pr_cnt  = {g: np.zeros(L, dtype=np.int64)  for g in ["target","cohort","rhyme","unrelated"]}

    # fast helpers
    def cos_with_set(h_seg_norm: torch.Tensor, idxs: list[int]) -> np.ndarray:
        # h_seg_norm: (Tseg,H) on CPU
        if len(idxs) == 0:
            return np.full(h_seg_norm.shape[0], np.nan, dtype=np.float32)
        Eset = E[idxs]  # (K,H)
        sims = (h_seg_norm @ Eset.T).mean(dim=1)  # (Tseg,)
        return sims.numpy()

    def softmax_mass(h_seg_norm: torch.Tensor, idxs_groups: dict) -> dict:
        # logits over union of all candidates (target+cohort+rhyme+unrelated)
        union = []
        for g in ["target","cohort","rhyme","unrelated"]:
            union += idxs_groups[g]
        union = list(dict.fromkeys(union))  # unique keep order
        if len(union) == 0:
            return {g: np.full(h_seg_norm.shape[0], np.nan, dtype=np.float32) for g in idxs_groups}

        Euni = E[union]  # (U,H)
        logits = (h_seg_norm @ Euni.T) / temp  # (Tseg,U)
        probs = torch.softmax(logits, dim=1)   # (Tseg,U)

        # map union positions
        pos = {idx: j for j, idx in enumerate(union)}
        out = {}
        for g, idxs in idxs_groups.items():
            if len(idxs) == 0:
                out[g] = np.full(h_seg_norm.shape[0], np.nan, dtype=np.float32)
                continue
            cols = [pos[i] for i in idxs if i in pos]
            if len(cols) == 0:
                out[g] = np.full(h_seg_norm.shape[0], np.nan, dtype=np.float32)
            else:
                out[g] = probs[:, cols].mean(dim=1).numpy()
        return out

    # iterate manifest
    used_tokens = 0
    correct_tokens = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            obj = json.loads(line)
            if obj.get("subset") != subset:
                continue

            # quick check: does this utterance contain any target word?
            words_in_utt = [w for (w, sf, ef) in obj["words"]]
            if not any(w in targets for w in words_in_utt):
                continue

            coch = np.load(obj["coch_path"])  # (64,T)
            feats = torch.from_numpy(coch.astype("float32")).T.contiguous()  # (T,64)
            T = feats.shape[0]

            x = feats.unsqueeze(0).to(device)
            x_lens = torch.tensor([T], dtype=torch.long, device=device)
            _, h, _ = model.forward_with_hidden(x, x_lens)
            h = h[0, :T].detach().cpu()  # (T,H)

            # normalize once
            h = h / (h.norm(dim=1, keepdim=True) + 1e-8)

            for w, sf, ef in obj["words"]:
                if w not in targets:
                    continue
                if w not in w2i:
                    continue

                sf = int(sf); ef = int(ef)
                if ef <= 0 or sf >= T:
                    continue
                sf = max(sf, 0); ef = min(ef, T)
                if ef <= sf:
                    continue

                comp = competitor_getter(w)
                if comp is None:
                    continue
                cohort, rhyme, unrelated = comp
                # require at least a few competitors
                if len(cohort) < 3 or len(rhyme) < 3 or len(unrelated) < 10:
                    continue

                idx_target = [w2i[w]]
                idx_cohort = [w2i[x] for x in cohort if x in w2i]
                idx_rhyme  = [w2i[x] for x in rhyme if x in w2i]
                idx_unrel  = [w2i[x] for x in unrelated if x in w2i]

                if len(idx_cohort) < 3 or len(idx_rhyme) < 3 or len(idx_unrel) < 10:
                    continue

                # correctness filter (optional)
                if correct_only:
                    h_last = h[ef-1]  # (H,)
                    sims = (E @ h_last)  # (V,)
                    pred = int(torch.argmax(sims).item())
                    if pred != idx_target[0]:
                        continue
                    correct_tokens += 1

                # window from onset
                seg_len = min(L, T - sf)
                h_seg = h[sf:sf+seg_len]  # (seg_len,H)

                # cosine curves
                cos_t = (h_seg @ E[idx_target].T).squeeze(1).numpy()
                cos_c = cos_with_set(h_seg, idx_cohort)
                cos_r = cos_with_set(h_seg, idx_rhyme)
                cos_u = cos_with_set(h_seg, idx_unrel)

                # prob-mass curves (softmax over union)
                prob = softmax_mass(h_seg, {
                    "target": idx_target,
                    "cohort": idx_cohort,
                    "rhyme": idx_rhyme,
                    "unrelated": idx_unrel,
                })

                for g, arr in [("target",cos_t),("cohort",cos_c),("rhyme",cos_r),("unrelated",cos_u)]:
                    valid = ~np.isnan(arr)
                    cos_sum[g][:seg_len][valid] += arr[valid]
                    cos_cnt[g][:seg_len][valid] += 1

                for g in ["target","cohort","rhyme","unrelated"]:
                    arr = prob[g]
                    valid = ~np.isnan(arr)
                    pr_sum[g][:seg_len][valid] += arr[valid]
                    pr_cnt[g][:seg_len][valid] += 1

                used_tokens += 1

            if line_idx % 500 == 0:
                print(f"[SCAN] lines={line_idx} used_tokens={used_tokens} correct_tokens={correct_tokens}")

    def finalize(sum_dict, cnt_dict):
        out = {}
        for g in sum_dict:
            y = sum_dict[g] / np.maximum(cnt_dict[g], 1)
            # where cnt=0 => nan
            y[cnt_dict[g] == 0] = np.nan
            out[g] = y.astype(np.float32)
        return out

    cos_curves = finalize(cos_sum, cos_cnt)
    prob_curves = finalize(pr_sum, pr_cnt)

    print(f"[DONE] used_tokens={used_tokens} correct_tokens={correct_tokens} correct_only={correct_only}")
    return cos_curves, prob_curves, L


def plot_curves(cos_curves, prob_curves, L, env_sr, out_png, title="Cohort competition"):
    t_ms = np.arange(L) * (1000.0 / env_sr)

    plt.figure(figsize=(11, 4.5))

    # (B)-like: cosine similarity timecourse
    plt.subplot(1, 2, 1)
    for g in ["target","cohort","rhyme","unrelated"]:
        plt.plot(t_ms, cos_curves[g], marker="o", markersize=3, linewidth=1, label=g)
    plt.xlabel("Time from word onset (ms)")
    plt.ylabel("Cosine similarity")
    plt.title("(B) Cosine competition")
    plt.legend()

    # (C)-like: activation proxy via softmax mass
    plt.subplot(1, 2, 2)
    for g in ["target","cohort","rhyme","unrelated"]:
        plt.plot(t_ms, prob_curves[g], marker="o", markersize=3, linewidth=1, label=g)
    plt.xlabel("Time from word onset (ms)")
    plt.ylabel("Softmax mass (activation proxy)")
    plt.title("(C) Activation proxy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")
    ap.add_argument("--ckpt", type=str, default=r"C:\linux_project\LENS\runs\librispeech_LSTM_WORD_srv_baseline_s000\ckpt\best.pt")
    ap.add_argument("--npz", type=str, default=r"C:\linux_project\LENS\round1_word_embeds_train100_4mats.npz", help="npz with *_mat embeddings (we use this as lexicon vectors)")
    ap.add_argument("--subset", type=str, default="dev-clean")
    ap.add_argument("--method", type=str, default="mid", choices=["mid","tail","last","delta"])
    ap.add_argument("--targets", type=str, default="AUTO", help="comma list or AUTO")
    ap.add_argument("--n_targets", type=int, default=30, help="used when targets=AUTO")
    ap.add_argument("--min_target_count", type=int, default=20, help="AUTO selection token count threshold (in subset)")
    ap.add_argument("--prefix_n", type=int, default=2)
    ap.add_argument("--suffix_n", type=int, default=2)
    ap.add_argument("--unrelated_min_ed", type=int, default=4)
    ap.add_argument("--max_unrelated", type=int, default=60)
    ap.add_argument("--env_sr", type=int, default=100)
    ap.add_argument("--max_ms", type=int, default=600)# 时间
    ap.add_argument("--correct_only", type=bool, default=True)#more better curve and cos value 
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_png", type=str, default=r"images/cohort_competition_devclean_lexTrain100_mid.png")
    args = ap.parse_args()

    pronouncing = try_get_pronouncing()
    if pronouncing is None:
        raise SystemExit("Please install pronouncing: pip install pronouncing")

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.ckpt)
    npz_path = Path(args.npz)

    model = infer_model_from_ckpt(ckpt_path, args.device)
    words, counts, E, w2i, meta = load_embeddings_npz(npz_path, args.method)

    phone, getter = build_competitors(
        words, w2i, pronouncing,
        prefix_n=args.prefix_n,
        suffix_n=args.suffix_n,
        unrelated_min_ed=args.unrelated_min_ed,
        max_unrelated=args.max_unrelated,
    )

    # choose targets
    if args.targets.strip().upper() == "AUTO":
        cnt_subset = scan_manifest_word_counts(manifest_path, args.subset)
        # pick candidates with enough tokens and with competitor sets available
        cands = []
        for w, c in cnt_subset.items():
            if c < args.min_target_count: 
                continue
            if w not in w2i: 
                continue
            comp = getter(w)
            if comp is None:
                continue
            cohort, rhyme, unrelated = comp
            if len(cohort) >= 3 and len(rhyme) >= 3 and len(unrelated) >= 10:
                cands.append((w, c))
        cands.sort(key=lambda x: x[1], reverse=True)
        targets = [w for w,_ in cands[:args.n_targets]]
        print(f"[TARGETS] AUTO picked {len(targets)} targets:", targets[:10], "...")
        if len(targets) == 0:
            raise SystemExit("AUTO found no targets. Use a larger lexicon (train-clean-100) or lower thresholds.")
    else:
        targets = [x.strip() for x in args.targets.split(",") if x.strip()]
        print(f"[TARGETS] manual: {targets}")

    cos_curves, prob_curves, L = compute_timecourses(
        model=model,
        manifest_path=manifest_path,
        subset=args.subset,
        targets=targets,
        words=words,
        w2i=w2i,
        E_np=E,
        competitor_getter=getter,
        device=args.device,
        env_sr=args.env_sr,
        max_ms=args.max_ms,
        correct_only=args.correct_only,
    )

    title = f"LSTM cohort competition ({args.subset}) method={args.method} targets={len(targets)} correct_only={args.correct_only}"
    plot_curves(cos_curves, prob_curves, L, args.env_sr, args.out_png, title=title)


if __name__ == "__main__":
    main()