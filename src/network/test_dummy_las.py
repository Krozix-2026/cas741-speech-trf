# test_dummy_las.py
# Minimal smoke test for kaituoxu/Listen-Attend-Spell (LAS)
#
# What it does:
#  1) rnd "audio features":  T x D   (like log-mel / fbank features)
#  2) forward() -> scalar CE loss
#  3) recognize() -> beam search hypothesis -> decoded "text"
#
# Note:
#  - weights are random => output is gibberish; this is a pipeline/shape test.
#  - decoder hidden size MUST match encoder output dim (dot-attn requires same dim).

from __future__ import annotations

import sys
from pathlib import Path
from argparse import Namespace

import torch

IGNORE_ID = -1


def add_repo_paths(repo_root: Path) -> None:
    # The repo's scripts import modules like: from encoder import Encoder
    # so we mimic that by adding src/models and src/utils to PYTHONPATH.
    sys.path.insert(0, str(repo_root / "src" / "models"))
    sys.path.insert(0, str(repo_root / "src" / "utils"))

def process_dict(dict_path):
    with open(dict_path, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0]
                 for entry in dictionary]
    sos_id = char_list.index('<sos>')
    eos_id = char_list.index('<eos>')
    return char_list, sos_id, eos_id

def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def main():
    repo_root = Path(__file__).resolve().parent
    add_repo_paths(repo_root)

    
    from encoder import Encoder
    from decoder import Decoder
    from seq2seq import Seq2Seq


    torch.manual_seed(0)

    # ---------------------------
    # 1) Define a tiny LAS model
    # ---------------------------
    vocab_size = 40
    sos_id = 1
    eos_id = 2
    # dummy "dictionary"
    char_list = ["<blank>", "<sos>", "<eos>"] + [f"c{i}" for i in range(3, vocab_size)]

    # "audio feature dim" (train.py default is 80)  :contentReference[oaicite:2]{index=2}
    einput = 64

    # encoder hidden
    ehidden = 128
    elayer = 2
    ebidirectional = True

    # IMPORTANT:
    # encoder output dim = ehidden * (2 if bidirectional else 1)
    enc_out_dim = ehidden * (2 if ebidirectional else 1)

    # decoder hidden MUST equal enc_out_dim (because DotProductAttention does q·k)
    dhidden = enc_out_dim
    dembed = 64
    dlayer = 1

    encoder = Encoder(
        input_size=einput,
        hidden_size=ehidden,
        num_layers=elayer,
        dropout=0.0,
        bidirectional=ebidirectional,
        rnn_type="lstm",
    )
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=dembed,
        sos_id=sos_id,
        eos_id=eos_id,
        hidden_size=dhidden,
        num_layers=dlayer,
        bidirectional_encoder=ebidirectional,
    )
    model = Seq2Seq(encoder, decoder)

    # ---------------------------
    # 2) Random fake "audio"
    # ---------------------------
    T = 120  # time frames
    x = torch.randn(T, einput).float()  # (T, D)
    print("x:", x.shape)#[120, 64]

    # pack_padded_sequence in this repo assumes a batch; we use batch=1
    padded_input = x.unsqueeze(0)  # (N=1, T, D)
    print("padded_input:", padded_input.shape)#[1, 120, 64]

    # Keep lengths on CPU for max compatibility with pack_padded_sequence
    input_lengths = torch.tensor([T], dtype=torch.long)  # (N=1,)

    # ---------------------------
    # 3) Random fake target tokens
    # ---------------------------
    U = 25  # output length (tokens, without sos/eos; decoder will add them)
    # avoid IGNORE_ID (-1); also avoid using <blank>/<sos>/<eos> for cleaner display
    y = torch.randint(low=3, high=vocab_size, size=(1, U), dtype=torch.long)
    print("y:", y.shape)#[1, 25]

    # (Optional) if you want to test padding behavior:
    # y_pad = y.new_full((1, U + 10), IGNORE_ID)
    # y_pad[:, :U] = y
    # y = y_pad

    # ---------------------------
    # 4) Forward -> loss
    # ---------------------------
    model.train()
    loss = model(padded_input, input_lengths, y)
    print(f"[OK] forward() loss = {loss.item():.4f}")

    # ---------------------------
    # 5) Recognize -> hypothesis
    # ---------------------------
    # decoder.recognize_beam reads args.beam_size/nbest/decode_max_len  :contentReference[oaicite:3]{index=3}
    args = Namespace(beam_size=3, nbest=1, decode_max_len=50)

    model.eval()
    with torch.no_grad():
        hyps = model.recognize(x, torch.tensor([T], dtype=torch.long), char_list, args)

    best = hyps[0]
    text, token, tokenid, score = parse_hypothesis(best, char_list)
    print(f"[OK] recognize() best score = {float(score):.3f}")
    print(f"[OK] tokenid: {tokenid}")
    print(f"[OK] token  : {token}")
    print(f"[OK] text   : {text}")


if __name__ == "__main__":
    main()
