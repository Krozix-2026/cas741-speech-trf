# eva/1plot_word_distance_semphon_round1.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    words = d["words"].tolist()
    counts = d["counts"].astype(int)
    keys = [k for k in d.files if k.endswith("_mat")]
    mats = {k: d[k].astype(np.float32) for k in keys}
    return words, counts, mats, d

def topk(words, mat, query, k=20):
    if query not in words:
        return None
    i = words.index(query)
    q = mat[i]
    sims = mat @ q
    order = np.argsort(-sims)
    res = []
    for j in order:
        if j == i:
            continue
        res.append((words[j], float(sims[j])))
        if len(res) >= k:
            break
    return res

def plot_neighbors(words, counts, mat, query, k=20, title="", out=None):
    nn = topk(words, mat, query, k=k)
    if nn is None:
        print(f"[WARN] '{query}' not in vocab.")
        return
    labels = [w for w, _ in nn][::-1]
    sims = [s for _, s in nn][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(labels)), sims)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("cosine similarity")
    plt.title(f"{title}: nearest neighbors of '{query}'")
    plt.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print("[SAVE]", out)
    else:
        plt.show()

def plot_heatmap(words, counts, mat, topK=80, out=None, title=""):
    idx = np.argsort(-counts)[:topK]
    sub_words = [words[i] for i in idx]
    X = mat[idx]
    S = X @ X.T
    plt.figure(figsize=(10, 9))
    plt.imshow(S, aspect="auto")
    plt.colorbar(label="cosine similarity")
    plt.title(f"{title}: similarity heatmap (top {topK} by count)")
    plt.xticks(range(topK), sub_words, rotation=90, fontsize=7)
    plt.yticks(range(topK), sub_words, fontsize=7)
    plt.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print("[SAVE]", out)
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=r"C:\linux_project\LENS\round1_word_embeds_devclean_SEMPHON.npz")
    ap.add_argument("--query", type=str, default="man")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--topK_heatmap", type=int, default=80)
    ap.add_argument("--mat", type=str, default="tail_mat",
                    help="choose from keys ending with _mat, e.g. tail_mat / sem_ctx_post_mat")
    ap.add_argument("--do_heatmap", action="store_true")
    ap.add_argument("--out_dir", type=str, default="./images")
    args = ap.parse_args()

    words, counts, mats, raw = load_npz(args.npz)
    if args.mat not in mats:
        print("[ERROR] unknown --mat:", args.mat)
        print("Available mats:", sorted(mats.keys()))
        raise SystemExit(1)

    mat = mats[args.mat]
    title = args.mat

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_neighbors(
        words, counts, mat, args.query, k=args.k,
        title=title,
        out=str(out_dir / f"neighbors_{title}_{args.query}.png")
    )

    if args.do_heatmap:
        plot_heatmap(
            words, counts, mat, topK=args.topK_heatmap,
            title=title,
            out=str(out_dir / f"heatmap_{title}_top{args.topK_heatmap}.png")
        )

if __name__ == "__main__":
    main()