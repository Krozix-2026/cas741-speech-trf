import torch
import torch.nn as nn




def framewise_bce_loss(logits_chunk,  # (B,Tc,V), raw logits
                       metas,         # list of dicts with "marks"
                       t0, t1,
                       device):
    B, Tc, V = logits_chunk.shape
    targets = torch.zeros(B, Tc, V, dtype=torch.float32, device=device)

    # 填充逐帧 one-hot 目标（静音默认全 0）
    for b in range(B):
        for (s, e, wid) in metas[b]["marks"]:
            if e <= t0 or s >= t1:
                continue
            s_loc = max(0, s - t0)
            e_loc = min(Tc, e - t0)
            if e_loc > s_loc:
                targets[b, s_loc:e_loc, wid] = 1.0

    # BCE with logits = sigmoid + BCE
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    

    loss = bce(logits_chunk, targets)
    return loss


def framewise_ce_loss(logits_chunk,  # (B,Tc,V) raw logits
                      metas,         # list of dicts with "marks"
                      t0, t1,
                      device,
                      blank_id: int = None):
    """
    If blank_id is not None, fill unlabelled frames with blank_id and train CE on all frames.
    Otherwise, compute CE only on the labelled frames (mask out others).
    """
    B, Tc, V = logits_chunk.shape

    if blank_id is not None:
        # Train on ALL frames (word frames = wid; silence/unlabelled = blank_id)
        targets = torch.full((B, Tc), blank_id, dtype=torch.long, device=device)
        for b in range(B):
            for (s, e, wid) in metas[b]["marks"]:
                s_loc = max(0, s - t0)
                e_loc = min(Tc, e - t0)
                if e_loc > s_loc:
                    targets[b, s_loc:e_loc] = int(wid)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits_chunk.transpose(1,2), targets)
        # (B,Tc,V) -> (B,V,Tc) for CE over last dim
        return loss
    else:
        # Train ONLY on labelled frames (word frames); ignore others
        all_idx = []
        all_tgt = []
        for b in range(B):
            for (s, e, wid) in metas[b]["marks"]:
                s_loc = max(0, s - t0)
                e_loc = min(Tc, e - t0)
                if e_loc > s_loc:
                    # collect (b, t) indices and their class
                    ts = torch.arange(s_loc, e_loc, device=device)
                    bs = torch.full((len(ts),), b, dtype=torch.long, device=device)
                    all_idx.append(torch.stack([bs, ts], dim=1))  # (L,2)
                    all_tgt.append(torch.full((len(ts),), int(wid), dtype=torch.long, device=device))
        if not all_idx:
            return logits_chunk.sum()*0.0  # no loss this chunk
        idx = torch.cat(all_idx, dim=0)         # (N, 2)
        tgt = torch.cat(all_tgt, dim=0)         # (N,)
        # gather logits at (b,t, :)
        gathered = logits_chunk[idx[:,0], idx[:,1], :]  # (N,V)
        loss = nn.CrossEntropyLoss(reduction="mean")(gathered, tgt)
        return loss