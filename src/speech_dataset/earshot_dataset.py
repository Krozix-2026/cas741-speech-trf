import numpy as np
from typing import List, Tuple, Dict, DefaultDict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# Continuous 10s stream dataset
# -------------------------------
class ContinuousStreamDataset(Dataset):
    def __init__(self,
                 token_store: Dict[str, Dict[str, List[np.ndarray]]],
                 speakers: List[str],
                 vocab_words: List[str],
                 words_by_spk: Dict[str, List[str]],
                 sr_hz: int = 100,
                 seg_len_sec: float = 10.0,
                 train: bool = True,
                 segments_per_epoch: Optional[int] = None,
                 insert_silence: bool = True,
                 noise_snr_db: Optional[float] = 20.0,
                 withheld_map: Optional[Dict[str, set]] = None,   # NEW
                 use_withheld_only: bool = False                  # NEW
                 ):
        self.bank = token_store
        self.speakers = list(speakers)
        self.words = list(vocab_words)
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.words_by_spk = words_by_spk

        # NEW: store withheld info
        self.withheld_map = withheld_map or {}
        self.use_withheld_only = use_withheld_only

        self.V = len(self.words)
        self.seg_T = int(seg_len_sec * sr_hz)
        self.train = train
        self.insert_silence = insert_silence
        self.noise_snr_db = noise_snr_db
        self.rng = np.random.RandomState(0 if train else 1)

        if segments_per_epoch is None:
            self.segments_per_epoch = 25 * (32 if train else 4)
        else:
            self.segments_per_epoch = segments_per_epoch

    def _add_bandwise_snr_noise(self, x):
        if (not self.train) or (self.noise_snr_db is None):
            return x
        snr_db = self.noise_snr_db
        x = x.copy()
        snr = 10 ** (snr_db / 10.0)
        for b in range(x.shape[1]):
            p = float(np.mean(x[:, b] ** 2)) + 1e-12
            nvar = p / snr
            x[:, b] += self.rng.normal(0.0, np.sqrt(nvar), size=x.shape[0])
        return x


    def __len__(self) -> int:
        # must return an int; make it super explicit
        return int(self.segments_per_epoch)

    def _rand_silence_T(self) -> int:
        return int(self.rng.randint(20, 51))  # 200–500ms @100Hz

    def __getitem__(self, idx):
        X = np.zeros((self.seg_T, 64), dtype=np.float32)
        Y = np.zeros((self.seg_T, self.V), dtype=np.float32) #localist one-hot

        marks = []
        t = 0

        while t < self.seg_T:
            spk = self.rng.choice(self.speakers)

            # base candidate words this speaker actually has recordings for
            base_words = [w for w in self.words_by_spk.get(spk, []) if w in self.word2id]

            # NEW: per-speaker filter based on withheld_map and mode
            if self.withheld_map:
                wset = self.withheld_map.get(spk, set())
                if self.use_withheld_only:
                    # validation/test: only the withheld words for THIS speaker
                    valid_words = [w for w in base_words if w in wset]
                else:
                    # training: exclude the withheld words for THIS speaker
                    valid_words = [w for w in base_words if w not in wset]
            else:
                valid_words = base_words

            if not valid_words:
                # no eligible words for this speaker under current mode; pick another
                continue

            k_words = 1 + (self.rng.rand() < 0.5)
            for _ in range(k_words):
                w = self.rng.choice(valid_words)
                toks = self.bank.get(spk, {}).get(w, [])
                if not toks:
                    continue
                j = int(self.rng.randint(len(toks)))
                tok = toks[j]
                if not isinstance(tok, np.ndarray) or tok.ndim != 2 or tok.shape[1] != 64:
                    continue
                T_w = tok.shape[0]
                if t >= self.seg_T:
                    break
                T_copy = min(T_w, self.seg_T - t)
                X[t:t+T_copy] = tok[:T_copy]
                wid = self.word2id[w]
                Y[t:t+T_copy, wid] = 1.0
                marks.append((t, t+T_copy, wid))
                t += T_copy
                if t >= self.seg_T:
                    break

            if self.train and self.insert_silence and t < self.seg_T:
                T_sil = min(self._rand_silence_T(), self.seg_T - t)
                t += T_sil

        X = self._add_bandwise_snr_noise(X)
        return torch.from_numpy(X), torch.from_numpy(Y), {"marks": marks, "word2id": self.word2id, "words": self.words}


