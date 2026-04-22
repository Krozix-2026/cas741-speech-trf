#eva\1plot_word_distance_round1.py
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

def plot_neighbors(words, counts, mat, query, k=20, title="tail_mean", out=None):
    nn = topk(words, mat, query, k=k)
    if nn is None:
        print(f"[WARN] '{query}' not in vocab. Choose another query.")
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
        plt.savefig(out, dpi=200)
        print("[SAVE]", out)
    else:
        plt.show()

def plot_heatmap(words, counts, mat, topK=80, out=None, title="tail_mean"):
    # pick topK by counts (more stable embeddings)
    idx = np.argsort(-counts)[:topK]
    sub_words = [words[i] for i in idx]
    X = mat[idx]  # already normalized
    S = X @ X.T   # cosine similarity matrix

    plt.figure(figsize=(10, 9))
    plt.imshow(S, aspect="auto")
    plt.colorbar(label="cosine similarity")
    plt.title(f"{title}: similarity heatmap (top {topK} by count)")
    plt.xticks(range(topK), sub_words, rotation=90, fontsize=7)
    plt.yticks(range(topK), sub_words, fontsize=7)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=200)
        print("[SAVE]", out)
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()#C:\linux_project\LENS\round1_word_embeds_devclean.npz
    ap.add_argument("--npz", type=str, default=r"C:\linux_project\LENS\round1_word_embeds_train100_4mats_phoneme.npz")
    ap.add_argument("--query", type=str, default="can")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--topK_heatmap", type=int, default=80)
    ap.add_argument("--mode", type=str, default="tail", choices=["tail", "last"])
    ap.add_argument("--do_heatmap", action="store_true")
    args = ap.parse_args()

    words, counts, tail, last = load_npz(args.npz)
    mat = tail if args.mode == "tail" else last
    title = "tail_mean" if args.mode == "tail" else "last_frame"

    # 如果没给 query，就自动选一个高频词当例子
    if args.query is None:
        q = words[int(np.argmax(counts))]
        print(f"[INFO] No --query provided. Using most frequent word: '{q}'")
    else:
        q = args.query

    plot_neighbors(
        words, counts, mat, q, k=args.k,
        title=title,
        out=f"./images/neighbors_{title}_{q}.png"
    )

    if args.do_heatmap:
        plot_heatmap(
            words, counts, mat, topK=args.topK_heatmap,
            title=title,
            out=f"./images/heatmap_{title}_top{args.topK_heatmap}.png"
        )

if __name__ == "__main__":
    main()