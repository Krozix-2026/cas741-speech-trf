import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from network.lstm_frame_srv_semantic import LSTMFrameSRVSemantic


# ----------------------------
# Utils
# ----------------------------
def softmax_from_scores(scores: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    # scores: (K,)
    return torch.softmax(scores / tau, dim=-1)


def entropy(p: torch.Tensor) -> float:
    # p: (K,) sum=1
    p = torch.clamp(p, 1e-12, 1.0)
    return float(-(p * torch.log(p)).sum().item())


def surprisal(p: torch.Tensor, idx: int) -> float:
    # -log p_true
    pt = float(torch.clamp(p[idx], 1e-12, 1.0).item())
    return float(-np.log(pt))


def infer_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    srv_dim = sd["frame_proj.weight"].shape[0]
    frame_hidden = sd["frame_proj.weight"].shape[1]
    in_dim = sd["frame_lstm.weight_ih_l0"].shape[1]
    frame_layers = len([k for k in sd.keys() if k.startswith("frame_lstm.weight_ih_l")])

    word_rep_dim = sd["word_proj.0.weight"].shape[0]
    word_lstm_hidden = sd["word_lstm.weight_ih_l0"].shape[0] // 4
    word_lstm_layers = len([k for k in sd.keys() if k.startswith("word_lstm.weight_ih_l")])

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
        f"[MODEL] in_dim={in_dim} srv_dim={srv_dim} "
        f"frame_hidden={frame_hidden} frame_layers={frame_layers} "
        f"word_rep_dim={word_rep_dim} word_hidden={word_lstm_hidden} word_layers={word_lstm_layers}"
    )
    return model


def build_vocab_from_manifest(manifest_path: Path, topk: int):
    """
    Must match training vocab construction:
      itos = ["<unk>"] + topk most common words (by raw token in manifest)
    """
    cnt = Counter()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for w, sf, ef in obj["words"]:
                cnt[w] += 1

    itos = ["<unk>"] + [w for w, _ in cnt.most_common(topk)]
    stoi = {w: i for i, w in enumerate(itos)}
    freqs = np.array([0] + [cnt[w] for w in itos[1:]], dtype=np.int64)
    print(f"[VOCAB] size={len(itos)} topk={topk}")
    return itos, stoi, freqs


def parse_contexts(s: str):
    """
    "he is a|he will|she is a"
    -> list of list tokens
    """
    ctxs = []
    for part in s.split("|"):
        part = part.strip()
        if not part:
            continue
        toks = [t.strip().lower() for t in part.split() if t.strip()]
        if toks:
            ctxs.append(toks)
    return ctxs


@dataclass
class Occ:
    utt_id: str
    coch_path: str
    words: list            # list of raw word tokens (original case)
    starts: list           # list[int]
    ends: list             # list[int]
    target_idx: int        # index of target word in words
    ctx_name: str          # context string


def find_occurrences(manifest_path: Path, subset_set: set[str], contexts: list[list[str]],
                     max_per_ctx: int, min_target_count: int,
                     stoi: dict, freqs: np.ndarray, cand_ids_set: set[int]):
    """
    Find occurrences where words[j:j+m] == context and target is words[j+m].
    Filter target by:
      - in stoi (else UNK)
      - frequency >= min_target_count (to avoid super rare targets)
      - target_id in candidate set (since we evaluate distribution over candidate words)
    """
    buckets = defaultdict(list)

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            subset = obj.get("subset")
            if subset not in subset_set:
                continue

            seq_raw = [w for (w, sf, ef) in obj["words"]]
            seq = [w.lower() for w in seq_raw]
            sfs = [int(sf) for (w, sf, ef) in obj["words"]]
            efs = [int(ef) for (w, sf, ef) in obj["words"]]

            for ctx in contexts:
                m = len(ctx)
                if len(seq) <= m:
                    continue
                ctx_name = " ".join(ctx)

                if len(buckets[ctx_name]) >= max_per_ctx:
                    continue

                # slide
                for j in range(0, len(seq) - m):
                    if len(buckets[ctx_name]) >= max_per_ctx:
                        break
                    if seq[j:j+m] != ctx:
                        continue
                    t = j + m
                    if t >= len(seq):
                        continue
                    w_t = seq_raw[t]
                    wid = stoi.get(w_t, 0)
                    if wid == 0:
                        continue
                    if freqs[wid] < min_target_count:
                        continue
                    if wid not in cand_ids_set:
                        continue

                    buckets[ctx_name].append(
                        Occ(
                            utt_id=obj["utt_id"],
                            coch_path=obj["coch_path"],
                            words=seq_raw,
                            starts=sfs,
                            ends=efs,
                            target_idx=t,
                            ctx_name=ctx_name,
                        )
                    )

    for k, v in buckets.items():
        print(f"[FOUND] ctx='{k}' n={len(v)}")
    return buckets


@torch.no_grad()
def compute_one_occ_curves(
    model: LSTMFrameSRVSemantic,
    occ: Occ,
    srv_table: torch.Tensor,      # (V,D) normalized
    cand_ids: torch.Tensor,       # (K,)
    stoi: dict,
    tail_frames: int,
    frame_points: list[int],
    tau: float,
    w_ctx: float,
    w_ac: float,
    device: str,
):
    """
    Returns dict with:
      - times_frames
      - surprisal_ctx (scalar)
      - entropy_ctx (scalar)
      - surprisal_ac[list], entropy_ac[list]
      - surprisal_comb[list], entropy_comb[list]
      - also returns prob_trajs for combined on a few top words (optional outside)
    """
    coch = np.load(occ.coch_path)  # (64,T)
    if coch.ndim != 2 or coch.shape[0] != 64:
        raise ValueError(f"Bad coch shape {coch.shape}: {occ.coch_path}")
    feats = torch.from_numpy(coch.astype("float32")).T.contiguous()  # (T,64)
    T = feats.shape[0]

    x = feats.unsqueeze(0).to(device)
    x_lens = torch.tensor([T], dtype=torch.long, device=device)

    # Build word boundary tensors for this utterance
    W = len(occ.words)
    word_starts = torch.tensor(occ.starts, dtype=torch.long, device=device).view(1, W)
    word_ends = torch.tensor(occ.ends, dtype=torch.long, device=device).view(1, W)
    word_lens = torch.tensor([W], dtype=torch.long, device=device)

    # Forward full model once
    y_frame, out_lens, pred_sem = model(
        x, x_lens,
        word_starts=word_starts,
        word_ends=word_ends,
        word_lens=word_lens,
        tail_frames=tail_frames,
        detach_frame_for_sem=False,
    )
    # y_frame: (1,T,D) pred_sem: (1,W,D)
    y_frame = y_frame[0, :T].float()          # (T,D)
    pred_sem = pred_sem[0].float()            # (W,D)

    target_idx = occ.target_idx
    if target_idx <= 0:
        return None

    target_word = occ.words[target_idx]
    target_id = stoi.get(target_word, 0)
    if target_id == 0:
        return None

    # candidate index for true word
    # (cand_ids is tensor of ids)
    cand_ids_cpu = cand_ids.cpu().numpy()
    pos = np.where(cand_ids_cpu == target_id)[0]
    if len(pos) == 0:
        return None
    true_pos = int(pos[0])

    # ---- ctx-only distribution ----
    # use prediction made at previous word position (i-1): pred_sem[i-1] predicts word i
    p_ctx_vec = pred_sem[target_idx - 1]  # (D,)
    p_ctx_vec = F.normalize(p_ctx_vec, dim=-1, eps=1e-8)

    cand_table = srv_table.index_select(0, cand_ids)  # (K,D) normalized
    scores_ctx = cand_table @ p_ctx_vec               # (K,) since both normalized => cosine
    probs_ctx = softmax_from_scores(scores_ctx, tau=tau)
    sur_ctx = surprisal(probs_ctx, true_pos)
    ent_ctx = entropy(probs_ctx)

    # ---- acoustic / combined over frame_points ----
    sf = int(occ.starts[target_idx])
    ef = int(occ.ends[target_idx])
    sf = max(0, min(sf, T))
    ef = max(0, min(ef, T))
    if ef <= sf:
        return None

    sur_ac = []
    ent_ac = []
    sur_comb = []
    ent_comb = []

    for L in frame_points:
        t_obs = min(sf + int(L), ef)
        if t_obs <= sf:
            t_obs = sf + 1
        i0 = max(sf, t_obs - int(tail_frames))
        ac_vec = y_frame[i0:t_obs].mean(dim=0)  # (D,)
        ac_vec = F.normalize(ac_vec, dim=-1, eps=1e-8)

        scores_ac = cand_table @ ac_vec
        probs_ac = softmax_from_scores(scores_ac, tau=tau)

        # combined: logit add (scores are already "energies")
        scores_comb = (w_ctx * scores_ctx) + (w_ac * scores_ac)
        probs_comb = softmax_from_scores(scores_comb, tau=tau)

        sur_ac.append(surprisal(probs_ac, true_pos))
        ent_ac.append(entropy(probs_ac))

        sur_comb.append(surprisal(probs_comb, true_pos))
        ent_comb.append(entropy(probs_comb))

    return {
        "ctx_name": occ.ctx_name,
        "utt_id": occ.utt_id,
        "target_word": target_word,
        "frame_points": frame_points,
        "sur_ctx": sur_ctx,
        "ent_ctx": ent_ctx,
        "sur_ac": np.array(sur_ac, dtype=np.float32),
        "ent_ac": np.array(ent_ac, dtype=np.float32),
        "sur_comb": np.array(sur_comb, dtype=np.float32),
        "ent_comb": np.array(ent_comb, dtype=np.float32),
        "true_pos": true_pos,
        "scores_ctx": scores_ctx.detach().cpu().numpy(),  # optional debug
    }


def mean_and_sem(arrs: list[np.ndarray]):
    X = np.stack(arrs, axis=0)  # (N,T)
    mu = X.mean(axis=0)
    se = X.std(axis=0, ddof=1) / max(1.0, np.sqrt(X.shape[0]))
    return mu, se


def plot_surprisal(groups, frame_points, out_png):
    plt.figure(figsize=(10, 6))
    for ctx_name, g in groups.items():
        # ctx-only is scalar
        sur_ctx = np.array([x["sur_ctx"] for x in g], dtype=np.float32)
        ctx_line = float(sur_ctx.mean())

        sur_ac_list = [x["sur_ac"] for x in g]
        sur_comb_list = [x["sur_comb"] for x in g]

        mu_ac, se_ac = mean_and_sem(sur_ac_list)
        mu_c, se_c = mean_and_sem(sur_comb_list)

        # acoustic-only
        plt.plot(frame_points, mu_ac, label=f"{ctx_name} | acoustic-only")
        plt.fill_between(frame_points, mu_ac - se_ac, mu_ac + se_ac, alpha=0.18)

        # combined
        plt.plot(frame_points, mu_c, label=f"{ctx_name} | combined")
        plt.fill_between(frame_points, mu_c - se_c, mu_c + se_c, alpha=0.18)

        # ctx-only horizontal
        plt.hlines(ctx_line, xmin=frame_points[0], xmax=frame_points[-1],
                   linestyles="dashed", linewidth=1.2, label=f"{ctx_name} | ctx-only")

    plt.xlabel("frames into target word (streaming tail window)")
    plt.ylabel("surprisal = -log P(true)")
    plt.title("Contextual meaning test (Scheme A): surprisal vs time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    print("[SAVE]", out_png)


def plot_entropy(groups, frame_points, out_png):
    plt.figure(figsize=(10, 6))
    for ctx_name, g in groups.items():
        ent_ctx = np.array([x["ent_ctx"] for x in g], dtype=np.float32)
        ctx_line = float(ent_ctx.mean())

        ent_ac_list = [x["ent_ac"] for x in g]
        ent_comb_list = [x["ent_comb"] for x in g]

        mu_ac, se_ac = mean_and_sem(ent_ac_list)
        mu_c, se_c = mean_and_sem(ent_comb_list)

        plt.plot(frame_points, mu_ac, label=f"{ctx_name} | acoustic-only")
        plt.fill_between(frame_points, mu_ac - se_ac, mu_ac + se_ac, alpha=0.18)

        plt.plot(frame_points, mu_c, label=f"{ctx_name} | combined")
        plt.fill_between(frame_points, mu_c - se_c, mu_c + se_c, alpha=0.18)

        plt.hlines(ctx_line, xmin=frame_points[0], xmax=frame_points[-1],
                   linestyles="dashed", linewidth=1.2, label=f"{ctx_name} | ctx-only")

    plt.xlabel("frames into target word (streaming tail window)")
    plt.ylabel("entropy")
    plt.title("Contextual meaning test (Scheme A): entropy vs time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    print("[SAVE]", out_png)


@torch.no_grad()
def plot_one_example_topwords(
    model, occ: Occ,
    srv_table: torch.Tensor,
    cand_ids: torch.Tensor,
    stoi: dict,
    tail_frames: int,
    frame_points: list[int],
    tau: float,
    w_ctx: float,
    w_ac: float,
    device: str,
    out_png: str,
    topM: int = 6,
):
    """
    For one occurrence, plot probability trajectories of top words under combined distribution.
    """
    coch = np.load(occ.coch_path)
    feats = torch.from_numpy(coch.astype("float32")).T.contiguous()
    T = feats.shape[0]
    x = feats.unsqueeze(0).to(device)
    x_lens = torch.tensor([T], dtype=torch.long, device=device)

    W = len(occ.words)
    word_starts = torch.tensor(occ.starts, dtype=torch.long, device=device).view(1, W)
    word_ends = torch.tensor(occ.ends, dtype=torch.long, device=device).view(1, W)
    word_lens = torch.tensor([W], dtype=torch.long, device=device)

    y_frame, out_lens, pred_sem = model(
        x, x_lens, word_starts, word_ends, word_lens,
        tail_frames=tail_frames, detach_frame_for_sem=False
    )
    y_frame = y_frame[0, :T].float()
    pred_sem = pred_sem[0].float()

    i = occ.target_idx
    if i <= 0:
        return
    target_word = occ.words[i]
    target_id = stoi.get(target_word, 0)

    cand_table = srv_table.index_select(0, cand_ids)  # (K,D)
    scores_ctx = cand_table @ F.normalize(pred_sem[i-1], dim=-1, eps=1e-8)

    sf = max(0, min(int(occ.starts[i]), T))
    ef = max(0, min(int(occ.ends[i]), T))
    if ef <= sf:
        return

    # first compute combined probs at final time to decide topM words
    L_last = frame_points[-1]
    t_obs = min(sf + int(L_last), ef)
    i0 = max(sf, t_obs - int(tail_frames))
    ac_vec = F.normalize(y_frame[i0:t_obs].mean(dim=0), dim=-1, eps=1e-8)
    scores_ac = cand_table @ ac_vec
    scores_comb = w_ctx * scores_ctx + w_ac * scores_ac
    probs_last = softmax_from_scores(scores_comb, tau=tau).detach().cpu().numpy()

    # pick top words
    top_idx = np.argsort(-probs_last)[:topM].tolist()
    # ensure true word included if possible
    if target_id != 0:
        cand_np = cand_ids.cpu().numpy()
        pos = np.where(cand_np == target_id)[0]
        if len(pos) > 0:
            true_pos = int(pos[0])
            if true_pos not in top_idx:
                top_idx[-1] = true_pos

    cand_np = cand_ids.cpu().numpy()
    top_word_ids = [int(cand_np[j]) for j in top_idx]

    # build id->word
    # (stoi is word->id; we need reverse lookup)
    # quick reverse map from candidate ids (small)
    id2word = {v: k for k, v in stoi.items()}
    top_words = [id2word.get(wid, f"id{wid}") for wid in top_word_ids]

    # compute trajectories
    P = np.zeros((len(top_idx), len(frame_points)), dtype=np.float32)

    for t_i, L in enumerate(frame_points):
        t_obs = min(sf + int(L), ef)
        i0 = max(sf, t_obs - int(tail_frames))
        ac_vec = F.normalize(y_frame[i0:t_obs].mean(dim=0), dim=-1, eps=1e-8)
        scores_ac = cand_table @ ac_vec
        scores_comb = w_ctx * scores_ctx + w_ac * scores_ac
        probs = softmax_from_scores(scores_comb, tau=tau).detach().cpu().numpy()
        for r, j in enumerate(top_idx):
            P[r, t_i] = probs[j]

    # plot
    plt.figure(figsize=(10, 6))
    for r, w in enumerate(top_words):
        plt.plot(frame_points, P[r], label=w)

    plt.xlabel("frames into target word (streaming tail window)")
    plt.ylabel("P(word | context + acoustic)")
    plt.title(f"One example: ctx='{occ.ctx_name}' target='{target_word}'  (utt={occ.utt_id})")
    plt.legend(fontsize=9)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    print("[SAVE]", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl")
    ap.add_argument("--subset", type=str, default="test-clean",
                    help="comma-separated, e.g. dev-clean,test-clean")
    ap.add_argument("--ckpt", type=str, default=r"C:\linux_project\LENS\runs\librispeech_LSTM_WORD_SEM_srv_semantic_hier_s000\ckpt\last.pt")
    ap.add_argument("--srv_table", type=str, default="",
                    help="default: sibling of run_dir, i.e. ../srv_table.pt")

    ap.add_argument("--vocab_topk", type=int, default=20000)
    ap.add_argument("--cand_topk", type=int, default=5000, help="candidate set size for softmax approx")

    ap.add_argument("--contexts", type=str, default="i want to|i had to|i began to|i started to")
    ap.add_argument("--max_per_ctx", type=int, default=120)
    ap.add_argument("--min_target_count", type=int, default=20)

    ap.add_argument("--tail_frames", type=int, default=10)
    ap.add_argument("--frame_points", type=str, default="1,2,3,5,8,10,12,15,18,20,25,30",
                    help="frames into target word to evaluate")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--w_ctx", type=float, default=1.0)
    ap.add_argument("--w_ac", type=float, default=1.0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="eva_out_contextA")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.ckpt)

    subset_set = set([s.strip() for s in args.subset.split(",") if s.strip()])
    contexts = parse_contexts(args.contexts)
    frame_points = [int(x.strip()) for x in args.frame_points.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # srv table path
    if args.srv_table.strip():
        srv_path = Path(args.srv_table)
    else:
        # ckpt: .../run_dir/ckpt/last.pt -> run_dir/srv_table.pt
        srv_path = ckpt_path.parent.parent / "srv_table.pt"
    if not srv_path.exists():
        raise FileNotFoundError(f"srv_table not found: {srv_path}")

    # load model + srv table
    model = infer_model_from_ckpt(ckpt_path, args.device)

    srv_obj = torch.load(srv_path, map_location="cpu")
    srv_table_cpu = srv_obj["table"] if isinstance(srv_obj, dict) and "table" in srv_obj else srv_obj
    srv_table = srv_table_cpu.float().to(args.device)  # (V,D) normalized
    V, D = srv_table.shape
    print(f"[SRV] table shape = {srv_table.shape} from {srv_path}")

    # vocab and candidate set
    itos, stoi, freqs = build_vocab_from_manifest(manifest_path, topk=args.vocab_topk)

    # candidate ids: top cand_topk by frequency (exclude <unk>=0)
    order = np.argsort(-freqs[1:])  # ids 1..topk
    cand_ids = (order[: args.cand_topk] + 1).astype(np.int64)
    cand_ids_set = set(int(x) for x in cand_ids.tolist())
    cand_ids_t = torch.tensor(cand_ids, dtype=torch.long, device=args.device)
    print(f"[CAND] K={len(cand_ids)} (most frequent words)")

    # find occurrences
    buckets = find_occurrences(
        manifest_path=manifest_path,
        subset_set=subset_set,
        contexts=contexts,
        max_per_ctx=args.max_per_ctx,
        min_target_count=args.min_target_count,
        stoi=stoi,
        freqs=freqs,
        cand_ids_set=cand_ids_set,
    )

    # compute curves
    groups = {}
    example_occ = None

    for ctx_name, occs in buckets.items():
        curves = []
        for occ in occs:
            res = compute_one_occ_curves(
                model=model,
                occ=occ,
                srv_table=srv_table,
                cand_ids=cand_ids_t,
                stoi=stoi,
                tail_frames=args.tail_frames,
                frame_points=frame_points,
                tau=args.tau,
                w_ctx=args.w_ctx,
                w_ac=args.w_ac,
                device=args.device,
            )
            if res is not None:
                curves.append(res)
                if example_occ is None:
                    example_occ = occ
        if len(curves) > 0:
            groups[ctx_name] = curves
        print(f"[CURVES] ctx='{ctx_name}' kept={len(curves)}/{len(occs)}")

    if not groups:
        raise RuntimeError("No valid occurrences found (try lowering --min_target_count or increasing --cand_topk).")

    # plots
    plot_surprisal(groups, frame_points, out_dir / "surprisal_vs_time.png")
    plot_entropy(groups, frame_points, out_dir / "entropy_vs_time.png")

    # one example: top-word trajectories
    if example_occ is not None:
        plot_one_example_topwords(
            model=model,
            occ=example_occ,
            srv_table=srv_table,
            cand_ids=cand_ids_t,
            stoi=stoi,
            tail_frames=args.tail_frames,
            frame_points=frame_points,
            tau=args.tau,
            w_ctx=args.w_ctx,
            w_ac=args.w_ac,
            device=args.device,
            out_png=str(out_dir / "one_example_topword_probs.png"),
            topM=6,
        )

    # save numeric results for later analysis
    out_json = out_dir / "metrics.json"
    payload = {}
    for ctx_name, g in groups.items():
        payload[ctx_name] = {
            "n": len(g),
            "targets_top10": Counter([x["target_word"] for x in g]).most_common(10),
            "mean_sur_ctx": float(np.mean([x["sur_ctx"] for x in g])),
            "mean_ent_ctx": float(np.mean([x["ent_ctx"] for x in g])),
        }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[SAVE]", out_json)


if __name__ == "__main__":
    main()