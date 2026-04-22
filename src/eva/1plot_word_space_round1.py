#eva\1plot_word_space_round1.py
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    words = d["words"].tolist()
    counts = d["counts"].astype(int)
    tail = d["tail_mat"].astype(np.float32)
    last = d["last_mat"].astype(np.float32)
    return words, counts, tail, last


def pca_2d(X):
    """X: (N,H) row-normalized recommended. returns (N,2)"""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD is stable
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    return Z


def classical_mds_2d(D):
    """
    Classical MDS from distance matrix D (N,N).
    Returns coordinates (N,2).
    """
    N = D.shape[0]
    # double-centering
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ (D ** 2) @ J
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # take top2 positive
    lam = np.maximum(eigvals[:2], 0.0)
    Z = eigvecs[:, :2] * np.sqrt(lam + 1e-12)
    return Z


def build_cosine_distance(X):
    """X row-normalized: cosine sim = X@X.T, distance = 1-sim"""
    S = X @ X.T
    D = 1.0 - S
    # numeric cleanup
    D = np.clip(D, 0.0, 2.0)
    return D


def scatter_with_labels(Z, words, counts, highlight=set(), label_top=40, title="", out=None):
    # size by log count
    sizes = 15 + 20 * (np.log1p(counts) / np.log1p(counts.max()))
    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=sizes, alpha=0.65)

    # label most frequent (readable) + highlighted
    idx_by_count = np.argsort(-counts)
    to_label = set(idx_by_count[:label_top].tolist())
    for i, w in enumerate(words):
        if w in highlight:
            to_label.add(i)

    for i in to_label:
        w = words[i]
        if w in highlight:
            plt.scatter([Z[i, 0]], [Z[i, 1]], s=sizes[i] * 2.2, alpha=0.9)
            plt.text(Z[i, 0], Z[i, 1], w, fontsize=11, fontweight="bold")
        else:
            plt.text(Z[i, 0], Z[i, 1], w, fontsize=8)

    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=220)
        print("[SAVE]", out)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=r"C:\linux_project\LENS\round1_word_embeds_devclean.npz")
    ap.add_argument("--mode", type=str, default="tail", choices=["tail", "last"])
    ap.add_argument("--method", type=str, default="mds", choices=["pca", "mds"])
    ap.add_argument("--topK", type=int, default=200, help="plot topK words by count (avoid clutter)")
    ap.add_argument("--label_top", type=int, default=40, help="label top N by count")
    ap.add_argument("--highlight", type=str, default="", help="comma-separated words to highlight")
    args = ap.parse_args()

    words, counts, tail, last = load_npz(args.npz)
    X = tail if args.mode == "tail" else last

    # pick topK frequent
    order = np.argsort(-counts)[:args.topK]
    words_k = [words[i] for i in order]
    counts_k = counts[order]
    Xk = X[order]

    # choose projection
    if args.method == "pca":
        Z = pca_2d(Xk)
        title = f"PCA 2D ({args.mode}) topK={args.topK}"
        out = f"space_pca_{args.mode}_top{args.topK}.png"
    else:
        D = build_cosine_distance(Xk)
        Z = classical_mds_2d(D)
        title = f"Classical MDS 2D (cosine distance) ({args.mode}) topK={args.topK}"
        out = f"space_mds_{args.mode}_top{args.topK}.png"

    highlight = set([w.strip() for w in args.highlight.split(",") if w.strip()])
    scatter_with_labels(
        Z, words_k, counts_k,
        highlight=highlight,
        label_top=args.label_top,
        title=title,
        out=out
    )


if __name__ == "__main__":
    main()