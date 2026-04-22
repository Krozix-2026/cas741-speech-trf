#eva\2phonetic_eval_allwords.py
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def levenshtein(seq_a, seq_b) -> int:
    """Edit distance for list (phones) or str."""
    if seq_a == seq_b:
        return 0
    if len(seq_a) == 0:
        return len(seq_b)
    if len(seq_b) == 0:
        return len(seq_a)
    dp = list(range(len(seq_b) + 1))
    for i, ca in enumerate(seq_a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(seq_b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(
                dp[j] + 1,      # delete
                dp[j - 1] + 1,  # insert
                prev + cost     # subst
            )
            prev = cur
    return dp[-1]


def load_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    words = d["words"].tolist()
    counts = d["counts"].astype(int)

    mats = {}
    for k in ("tail_mat", "last_mat", "delta_mat", "mid_mat"):
        if k in d.files:
            mats[k] = d[k].astype(np.float32)

    meta = {k: (d[k].item() if d[k].shape == () else d[k].tolist()) for k in d.files if k not in mats and k not in ("words", "counts")}
    return words, counts, mats, meta


def try_import_pronouncing():
    try:
        import pronouncing  # type: ignore
        return pronouncing
    except Exception as e:
        return None


def build_phone_dict(words: List[str]) -> Dict[str, Optional[List[str]]]:
    """
    Return dict: word -> phones(list) or None if not found.
    Uses first pronunciation from CMUdict.
    """
    pronouncing = try_import_pronouncing()
    if pronouncing is None:
        raise RuntimeError("Missing dependency: pronouncing. Install via: pip install pronouncing")

    phone = {}
    for w in words:
        ps = pronouncing.phones_for_word(w.lower())
        if not ps:
            phone[w] = None
        else:
            phone[w] = ps[0].split()
    return phone


def topk_neighbor_indices(mat: np.ndarray, k: int) -> np.ndarray:
    """
    mat: (V,H) normalized rows recommended.
    Return neighbors: (V,k) indices excluding self.
    Uses full similarity matrix mat @ mat.T, which is OK for V up to a few 10k.
    For very large V, switch to chunking.
    """
    V = mat.shape[0]
    sims = mat @ mat.T  # (V,V)
    # exclude self by setting diagonal very small
    np.fill_diagonal(sims, -1e9)
    # argpartition for topk, then sort within topk
    idx = np.argpartition(-sims, kth=np.arange(k), axis=1)[:, :k]  # (V,k) unordered
    row = np.arange(V)[:, None]
    vals = sims[row, idx]
    order = np.argsort(-vals, axis=1)
    idx_sorted = idx[row, order]
    return idx_sorted


def signflip_pvalue(diffs: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    """
    One-sided sign-flip permutation test for mean(diffs) > 0.
    Returns: (p_value, observed_mean)
    """
    rng = np.random.default_rng(seed)
    obs = float(np.mean(diffs))
    if n_perm <= 0:
        return 1.0, obs
    # sign flip
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=diffs.shape[0])
        stat = float(np.mean(diffs * signs))
        if stat >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return p, obs


def paired_signflip_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    """
    Paired sign-flip test on delta = b - a.
    One-sided: mean(delta) > 0 means b has larger value than a.
    For our case (ED smaller is better), comparing nn_meanED:
      delta = nnED_B - nnED_A
      If delta > 0 => A better (smaller ED).
    So interpret carefully in reporting.
    """
    delta = b - a
    return signflip_pvalue(delta, n_perm=n_perm, seed=seed)


def eval_one_method(
    method_name: str,
    mat: np.ndarray,
    words: List[str],
    counts: np.ndarray,
    phones: Dict[str, Optional[List[str]]],
    k: int,
    n_rand: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    For each word (with phones), compute:
      nn_meanED: mean phoneED over topK neighbors (neighbors with phones only)
      rand_meanED: mean phoneED over random K neighbors (phones only), averaged over n_rand repeats
      improvement = rand_meanED - nn_meanED  (positive = neighbors phonetically closer than random)
    Returns:
      summary dict
      per_word dict: word -> metrics
    """
    rng = np.random.default_rng(seed)
    V = len(words)

    # eligible indices: word has phones
    elig = np.array([phones[w] is not None for w in words], dtype=bool)
    elig_idx = np.where(elig)[0]
    if len(elig_idx) < k + 5:
        raise RuntimeError(f"Too few words with pronunciations: {len(elig_idx)}")

    # precompute topK indices for all words (fast)
    nbr_idx = topk_neighbor_indices(mat, k=max(k * 3, k + 10))  # oversample, then filter by phone availability

    # cache for phoneED between word pairs
    ed_cache: Dict[Tuple[int, int], int] = {}

    def phone_ed(i: int, j: int) -> int:
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in ed_cache:
            return ed_cache[key]
        pa = phones[words[a]]
        pb = phones[words[b]]
        assert pa is not None and pb is not None
        d = levenshtein(pa, pb)
        ed_cache[key] = d
        return d

    per_word: Dict[str, Dict[str, float]] = {}
    used_words = 0
    nn_list = []
    rand_list = []
    imp_list = []

    for i in range(V):
        w = words[i]
        if phones[w] is None:
            continue

        # ----- topK neighbors with phones -----
        candidates = nbr_idx[i]
        nn = []
        for j in candidates:
            if i == j:
                continue
            if phones[words[j]] is None:
                continue
            nn.append(j)
            if len(nn) >= k:
                break
        if len(nn) < k:
            # not enough neighbors with phones; skip
            continue

        nn_eds = [phone_ed(i, j) for j in nn]
        nn_mean = float(np.mean(nn_eds))

        # ----- random baseline -----
        # sample from eligible (phones only) excluding i
        pool = elig_idx[elig_idx != i]
        # to avoid tiny pools, guard
        if len(pool) < k:
            continue

        rand_means = []
        for _ in range(n_rand):
            js = rng.choice(pool, size=k, replace=False)
            eds = [phone_ed(i, int(j)) for j in js]
            rand_means.append(float(np.mean(eds)))
        rand_mean = float(np.mean(rand_means))

        imp = rand_mean - nn_mean

        per_word[w] = {
            "count": float(counts[i]),
            "nn_meanED": nn_mean,
            "rand_meanED": rand_mean,
            "improvement": imp,
        }
        used_words += 1
        nn_list.append(nn_mean)
        rand_list.append(rand_mean)
        imp_list.append(imp)

    nn_arr = np.array(nn_list, dtype=np.float32)
    rand_arr = np.array(rand_list, dtype=np.float32)
    imp_arr = np.array(imp_list, dtype=np.float32)

    summary = {
        "method": method_name,
        "n_words_used": float(used_words),
        "k": float(k),
        "n_rand": float(n_rand),
        "mean_nnED": float(np.mean(nn_arr)) if used_words > 0 else float("nan"),
        "mean_randED": float(np.mean(rand_arr)) if used_words > 0 else float("nan"),
        "mean_improvement": float(np.mean(imp_arr)) if used_words > 0 else float("nan"),
        "median_improvement": float(np.median(imp_arr)) if used_words > 0 else float("nan"),
        "std_improvement": float(np.std(imp_arr)) if used_words > 0 else float("nan"),
    }
    return summary, per_word


def write_per_word_csv(
    out_csv: Path,
    words: List[str],
    phones: Dict[str, Optional[List[str]]],
    per_word_all: Dict[str, Dict[str, Dict[str, float]]],
):
    """
    per_word_all: method -> word -> metrics
    """
    methods = list(per_word_all.keys())
    methods.sort()

    # union of words that appear in any method
    all_words = set()
    for m in methods:
        all_words.update(per_word_all[m].keys())
    all_words = sorted(all_words)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["word", "phones"]
        for m in methods:
            header += [f"{m}_nnMeanED", f"{m}_randMeanED", f"{m}_improvement"]
        w.writerow(header)

        for ww in all_words:
            ph = phones.get(ww, None)
            ph_str = " ".join(ph) if ph is not None else ""
            row = [ww, ph_str]
            for m in methods:
                d = per_word_all[m].get(ww, None)
                if d is None:
                    row += ["", "", ""]
                else:
                    row += [f"{d['nn_meanED']:.4f}", f"{d['rand_meanED']:.4f}", f"{d['improvement']:.4f}"]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=r"C:\linux_project\LENS\round1_word_embeds_devclean_4mats.npz", help="npz with words/counts and *_mat arrays")
    ap.add_argument("--k", type=int, default=15, help="top-K neighbors")
    ap.add_argument("--n_rand", type=int, default=50, help="random baselines per word")
    ap.add_argument("--n_perm", type=int, default=2000, help="sign-flip permutations for p-value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="phonetic_eval_out")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    words, counts, mats, meta = load_npz(npz_path)
    print(f"[LOAD] V={len(words)} methods={list(mats.keys())}")

    # Build pronunciations
    phones = build_phone_dict(words)
    n_has = sum(1 for w in words if phones[w] is not None)
    print(f"[PHONES] words_with_pron={n_has}/{len(words)}")

    # Ensure mats are row-normalized (should already be, but safe)
    for name, mat in mats.items():
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mats[name] = mat / norms

    # Evaluate each method
    summaries = []
    per_word_all: Dict[str, Dict[str, Dict[str, float]]] = {}

    for key, mat in mats.items():
        method = key.replace("_mat", "")  # tail, last, delta, mid
        print(f"\n[EVAL] method={method}")
        summary, per_word = eval_one_method(
            method_name=method,
            mat=mat,
            words=words,
            counts=counts,
            phones=phones,
            k=args.k,
            n_rand=args.n_rand,
            seed=args.seed + hash(method) % 10000,
        )
        # p-value for improvement > 0
        diffs = np.array([per_word[w]["improvement"] for w in per_word.keys()], dtype=np.float32)
        p, obs = signflip_pvalue(diffs, n_perm=args.n_perm, seed=args.seed + 12345)
        summary["p_value_improvement_gt0"] = float(p)
        summary["obs_mean_improvement"] = float(obs)

        summaries.append(summary)
        per_word_all[method] = per_word

        print(f"  n_words_used={int(summary['n_words_used'])}")
        print(f"  mean_nnED={summary['mean_nnED']:.4f}  mean_randED={summary['mean_randED']:.4f}")
        print(f"  mean_improvement={summary['mean_improvement']:.4f}  p={p:.4g}")

    # Rank by mean_improvement (bigger is better)
    summaries_sorted = sorted(summaries, key=lambda x: x["mean_improvement"], reverse=True)

    # Pairwise comparisons between methods (optional, but useful)
    # Compare nn_meanED: smaller is better. We'll test delta = nnED_B - nnED_A
    # If mean(delta) > 0 => A better than B.
    pairwise = []
    methods = list(per_word_all.keys())
    methods.sort()
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            A = methods[i]
            B = methods[j]
            common = sorted(set(per_word_all[A].keys()) & set(per_word_all[B].keys()))
            if len(common) < 20:
                continue
            nnA = np.array([per_word_all[A][w]["nn_meanED"] for w in common], dtype=np.float32)
            nnB = np.array([per_word_all[B][w]["nn_meanED"] for w in common], dtype=np.float32)
            p_ab, mean_delta = paired_signflip_pvalue(nnA, nnB, n_perm=args.n_perm, seed=args.seed + 777)
            # mean_delta = mean(nnB - nnA). if >0 => A has smaller ED => A better
            pairwise.append({
                "A": A,
                "B": B,
                "n_common": int(len(common)),
                "mean(nnB-nnA)": float(mean_delta),
                "p_value_A_better_than_B": float(p_ab),
            })

    # Save summary JSON
    out_json = out_dir / "phonetic_eval_summary.json"
    payload = {
        "npz": str(npz_path),
        "meta": meta,
        "k": args.k,
        "n_rand": args.n_rand,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "ranked_by_mean_improvement": summaries_sorted,
        "all_summaries": summaries,
        "pairwise_nnED_signflip_tests": pairwise,
        "notes": {
            "improvement": "rand_meanED - nn_meanED; positive means topK neighbors are phonetically closer than random.",
            "p_value_improvement_gt0": "sign-flip permutation p-value for mean(improvement) > 0 (one-sided).",
            "pairwise": "tests on nn_meanED (smaller is better). mean(nnB-nnA)>0 indicates A better than B.",
        }
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SAVE] {out_json}")

    # Save per-word CSV
    out_csv = out_dir / "phonetic_eval_per_word.csv"
    write_per_word_csv(out_csv, words, phones, per_word_all)
    print(f"[SAVE] {out_csv}")

    # Print ranking
    print("\n" + "=" * 80)
    print("[RANKING] by mean_improvement (higher is better)")
    for r, s in enumerate(summaries_sorted, 1):
        print(
            f"{r:02d}. {s['method']:8s}  "
            f"n={int(s['n_words_used']):4d}  "
            f"nnED={s['mean_nnED']:.3f}  randED={s['mean_randED']:.3f}  "
            f"imp={s['mean_improvement']:.3f}  p={s['p_value_improvement_gt0']:.3g}"
        )


if __name__ == "__main__":
    main()