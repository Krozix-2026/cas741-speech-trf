# _experiment.py
# ------------------------------------------------------------
# Appleseed TRFExperiment (eelbrain + trftools) pipeline config
#
# 关键点：
# - eelbrain Dataset: len(ds) 是变量(列)数；case(行)数用 ds.n_cases
# - 本版 label_events() 会：
#   1) 仅保留 trigger 162/167
#   2) (onset, offset) 配对得到每段时长
#   3) 自动选择 REAL(11) vs PILOTS(11b)
#   4) 若事件多于 11 段：用 DP 选出最匹配刺激长度的 11 对 (on, off)，丢弃多余触发
#
# 运行前强烈建议删除旧 cache（若你之前跑过截断/错误版本）：
#   C:\Dataset\Appleseed_BIDS_new\derivatives\eelbrain\cache\
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths  # noqa: F401

from eelbrain import load  # noqa: F401
from eelbrain.pipeline import *
from trftools.pipeline import *


# -------------------------
# Stimuli table
# -------------------------
DIR = Path(__file__).parent
STIMULI = load.tsv(DIR / "appleseed_stimuli.txt", types="fv")

# 真实被试：听 11（不包含 11b）
STIMULI_REAL_BASE = STIMULI.sub("stimulus != '11b'")    # 11条：1-11
# pilot 被试：听 11b（不包含 11）
STIMULI_PILOTS_BASE = STIMULI.sub("stimulus != '11'")   # 11条：1-10+11b

print("[DEBUG] stimuli file:", DIR / "appleseed_stimuli.txt")
print(f"[DEBUG] STIMULI: n_cases={STIMULI.n_cases} n_vars={len(STIMULI)} values={list(STIMULI['stimulus'])}")
print(f"[DEBUG] REAL  : n_cases={STIMULI_REAL_BASE.n_cases} n_vars={len(STIMULI_REAL_BASE)} tail={list(STIMULI_REAL_BASE['stimulus'][-3:])}")
print(f"[DEBUG] PILOT : n_cases={STIMULI_PILOTS_BASE.n_cases} n_vars={len(STIMULI_PILOTS_BASE)} tail={list(STIMULI_PILOTS_BASE['stimulus'][-3:])}")


# -------------------------
# Parcellations
# -------------------------
LATERAL_TEMPORAL = (
    "transversetemporal", "superiortemporal", "bankssts", "middletemporal",
    "inferiortemporal", "temporalpole"
)
IFG = ("parsopercularis", "parsorbitalis", "parstriangularis")
MFG = ("caudalmiddlefrontal", "rostralmiddlefrontal")
LATERAL_FRONTAL = (
    "caudalmiddlefrontal", "frontalpole", "parsopercularis", "parsorbitalis",
    "parstriangularis", "precentral", "rostralmiddlefrontal", "superiorfrontal"
)
LATERAL_PARIETAL = ("postcentral", "inferiorparietal", "superiorparietal", "supramarginal")
OTHER_MEDIAL = ("cuneus", "lateralorbitofrontal", "medialorbitofrontal", "paracentral", "precuneus", "fusiform")
FTP_LATERAL = set(LATERAL_FRONTAL + LATERAL_TEMPORAL + LATERAL_PARIETAL + OTHER_MEDIAL)
OCCIPITAL = ("lateraloccipital", "pericalcarine", "lingual")


class Appleseed(TRFExperiment):
    auto_delete_results = True
    # screen_log_level = 'debug'

    defaults = {
        "task": "Appleseed",
        "split": "01",
        "raw": "1-40",
        "rej": "",
        "cov": "emptyroom",
        "inv": "fixed-6-MNE-0",
    }

    groups = {
        "righthanders": SubGroup("all", []),
    }

    raw = {
        "raw": RawSource(connectivity="auto"),
        "tsss": RawMaxwell("raw", st_duration=10.0, ignore_ref=True, st_correlation=0.9, st_only=True),
        "1-40": RawFilter("tsss", 1, 40),
        "ica": RawICA("1-40", "Appleseed", n_components=0.99),
        "ica-20": RawFilter("ica", None, 20, cache=False),
        "tsss-ica": RawApplyICA("tsss", "ica", cache=False),
        "ica-0-20": RawFilter("tsss-ica", None, 20, cache=False),
    }

    parcs = {
        "lateraltemporal": SubParc("aparc", LATERAL_TEMPORAL, views="lateral"),
        "superiortemporal": SubParc("aparc", ("superiortemporal", "transversetemporal")),
        "ftp": SubParc("aparc", sorted(FTP_LATERAL)),
        "temporal_to_ftp": SubParc("aparc", sorted(FTP_LATERAL.difference(LATERAL_TEMPORAL))),
        "stg_to_ftp": SubParc("aparc", sorted(FTP_LATERAL.difference(["transversetemporal", "superiortemporal"]))),
        "occipital": SubParc("aparc", OCCIPITAL),
        "ifg": SubParc("aparc", IFG),
        "lhg": SubParc("aparc", ("transversetemporal-lh",), views="lateral"),
    }

    variables = {"event": LabelVar("trigger", {162: "onset", 167: "offset"})}

    # -------------------------
    # Empty-room covariance fallback
    # -------------------------
    def make_cov(self, *args, **kwargs):
        try:
            return super().make_cov(*args, **kwargs)
        except ValueError as e:
            if "task='emptyroom'" not in str(e):
                raise

        root = Path(self.root)
        cov_path = self.get("cov-file", make=False)

        candidates = sorted(root.glob("sub-emptyroom/**/meg/*_meg.fif"))
        if not candidates:
            candidates = sorted(root.glob("sub-emptyroom/**/meg/*.fif"))
        if not candidates:
            raise FileNotFoundError(f"No empty-room MEG FIF found under: {root/'sub-emptyroom'}")

        er_fif = candidates[-1]
        raw_er = mne.io.read_raw_fif(er_fif, preload=False, verbose="ERROR")
        raw_er.load_data()
        raw_er.filter(1.0, 40.0, verbose="ERROR")

        cov = mne.compute_raw_covariance(raw_er, method="empirical", verbose="ERROR")
        mne.write_cov(cov_path, cov, overwrite=True)
        print(f"[INFO] empty-room cov built from {er_fif} -> {cov_path}")
        return cov_path

    # -------------------------
    # Debug: load_epochs
    # -------------------------
    def load_epochs(self, *args, **kwargs):
        kwargs.setdefault("add_bads", False)
        ds = super().load_epochs(*args, **kwargs)

        if not getattr(self, "_printed_source_debug", False):
            self._printed_source_debug = True
            print("\n========== [DEBUG] load_epochs() sanity ==========")
            print("ds.n_cases =", ds.n_cases)
            print("ds.n_vars  =", len(ds))
            print("ds.info keys =", sorted(ds.info.keys()))
            print("subject =", ds.info.get("subject"))
            print("task =", ds.info.get("task"))
            print("session =", ds.info.get("session"))
            if "trigger" in ds:
                print("unique triggers =", np.unique(np.asarray(ds["trigger"])))
            print("=================================================\n")

        return ds

    # -------------------------
    # Predictor safety checks (optional)
    # -------------------------
    def load_predictor(self, *args, **kwargs):
        x = super().load_predictor(*args, **kwargs)
        data = x.x

        bad = ~np.isfinite(data)
        if bad.any():
            print(f"[WARN] non-finite in predictor: {args[0] if args else '<pred>'} -> replacing with 0")
            x = x.copy()
            x.x[bad] = 0.0
            data = x.x

        if data.ndim == 2:
            std0 = data.std(axis=0) == 0
            if std0.any():
                idx = np.where(std0)[0]
                print(f"[WARN] flat predictor columns in {args[0] if args else '<pred>'}: n={len(idx)} -> {idx[:10]} ...")

        return x

    # -------------------------
    # Helper: pair onsets with subsequent offsets
    # -------------------------
    @staticmethod
    def _pair_on_off(trig: np.ndarray, times: np.ndarray):
        on_idx = np.where(trig == 162)[0]
        off_idx = np.where(trig == 167)[0]
        pairs = []
        j = 0
        for oi in on_idx:
            while j < len(off_idx) and off_idx[j] < oi:
                j += 1
            if j >= len(off_idx):
                break
            pairs.append((oi, off_idx[j]))
            j += 1
        if not pairs:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
        pairs = np.asarray(pairs, dtype=int)
        dur = times[pairs[:, 1]] - times[pairs[:, 0]]
        return pairs, dur

    # -------------------------
    # Helper: DP select best K pairs to match expected lengths
    # -------------------------
    @staticmethod
    def _select_best_k_subsequence(dur: np.ndarray, lens: np.ndarray):
        """
        Select a subsequence of len(lens) from dur (in order) minimizing L1 error sum.
        Returns indices into dur (selected positions).
        """
        m = len(dur)
        k = len(lens)
        if m < k:
            raise RuntimeError(f"Not enough (on,off) pairs: have {m}, need {k}")

        INF = 1e18
        dp = np.full((m + 1, k + 1), INF, dtype=float)
        take = np.zeros((m + 1, k + 1), dtype=bool)

        dp[0, 0] = 0.0
        for i in range(1, m + 1):
            dp[i, 0] = 0.0
            jmax = min(i, k)
            for j in range(1, jmax + 1):
                # skip i-1
                best = dp[i - 1, j]
                best_take = False
                # take i-1 as match for j-1
                cand = dp[i - 1, j - 1] + abs(dur[i - 1] - lens[j - 1])
                if cand < best:
                    best = cand
                    best_take = True
                dp[i, j] = best
                take[i, j] = best_take

        # backtrack
        sel = []
        i, j = m, k
        while j > 0:
            if take[i, j]:
                sel.append(i - 1)
                i -= 1
                j -= 1
            else:
                i -= 1
        sel.reverse()
        return np.asarray(sel, dtype=int)

    # -------------------------
    # label_events: robust alignment to 11 segments (11 or 11b)
    # -------------------------
    def label_events(self, ds):
        if ds.info.get("task") != "Appleseed":
            return ds
        if "trigger" not in ds:
            return ds

        trig = np.asarray(ds["trigger"])
        # 只保留 162/167
        keep = np.isin(trig, (162, 167))
        if not np.all(keep):
            ds = ds.sub(keep)
            trig = trig[keep]

        if "time" not in ds:
            # 极少数情况下事件数据没有 time，无法配对
            raise RuntimeError("Events dataset has no 'time' variable; cannot align stimuli.")
        times = np.asarray(ds["time"], float)

        # (on, off) 配对，得到每段时长
        pairs, dur = self._pair_on_off(trig, times)

        # 自动选择 REAL vs PILOTS（只用尾部 3 段更稳）
        def tail_score(stim_base, tail=3):
            lens = np.asarray(stim_base["length"], float)
            if len(dur) == 0 or len(lens) == 0:
                return np.inf
            t = min(tail, len(dur), len(lens))
            return np.nanmedian(np.abs(dur[-t:] - lens[-t:]))

        s_real = tail_score(STIMULI_REAL_BASE)
        s_pil = tail_score(STIMULI_PILOTS_BASE)
        stim_base = STIMULI_REAL_BASE if s_real <= s_pil else STIMULI_PILOTS_BASE
        lens = np.asarray(stim_base["length"], float)

        # 目标段数 K=11：如果 pairs 多于 11，选最匹配的 11 对
        K = stim_base.n_cases
        if len(dur) != K:
            if len(dur) < K:
                n_on = int(np.sum(trig == 162))
                n_off = int(np.sum(trig == 167))
                raise RuntimeError(
                    f"Not enough paired segments: subject={ds.info.get('subject')} "
                    f"pairs={len(dur)} need={K} (on={n_on}, off={n_off})"
                )

            sel = self._select_best_k_subsequence(dur, lens)
            # 保持 (on,off) 的顺序，不要排序
            keep_idx = []
            for pi in sel:
                oi, fi = pairs[pi]
                keep_idx.extend([int(oi), int(fi)])

            ds = ds[keep_idx]
            trig = np.asarray(ds["trigger"])
            times = np.asarray(ds["time"], float)
            # 重新计算（可选 debug）
            pairs, dur = self._pair_on_off(trig, times)
            print(f"[INFO] Pruned extra triggers: subject={ds.info.get('subject')} kept_pairs={len(dur)} dropped_pairs={len(sel)}?")

        # 现在 ds 应该是 2*K 个事件，stimuli 用 repeat(2)
        stimuli = stim_base.repeat(2)

        if ds.n_cases != stimuli.n_cases:
            n_on = int(np.sum(trig == 162))
            n_off = int(np.sum(trig == 167))
            raise RuntimeError(
                f"Event/stimulus mismatch after pruning: subject={ds.info.get('subject')} "
                f"events={ds.n_cases} (on={n_on}, off={n_off}) stimuli={stimuli.n_cases}"
            )

        ds.update(stimuli)
        return ds

    # -------------------------
    # Epochs
    # -------------------------
    epochs = {
        "apple": PrimaryEpoch("Appleseed", "event == 'onset'", tmin=0, tmax="length", samplingrate=100),
        "seg1-2": SecondaryEpoch("apple", "stimulus.isin(('1', '2'))"),
        "seg5-6": SecondaryEpoch("apple", "stimulus.isin(('5', '6'))"),
        "seg7-8": SecondaryEpoch("apple", "stimulus.isin(('7', '8'))"),
        "seg1-5": SecondaryEpoch("apple", "stimulus.isin(('1', '2', '3', '4', '5'))"),
        "seg6-11": SecondaryEpoch("apple", "stimulus.isin(('6', '7', '8', '9', '10', '11', '11b'))"),
    }

    # -------------------------
    # Predictors
    # -------------------------
    predictors = {
        "gammatone": FilePredictor(resample="resample", sampling="continuous"),
        "phonotactic": FilePredictor(columns=True, resample="bin", sampling="discrete"),
        "phone": FilePredictor(columns=True, resample="bin", sampling="discrete"),
        "c5phone": FilePredictor(columns=True, resample="bin", sampling="discrete"),
    }

    # -------------------------
    # Models
    # -------------------------
    models = {
        "gte8": "gammatone-8 + gammatone-edge30-8",
        "phonotactics": "phonotactic-surprisal + phonotactic-entropy",
        "c5-phone-0v1_": "c5phone-p0 + c5phone-p1_",
        "c5-cohort-u": "c5phone-u_surprisal + c5phone-u_entropy + c5phone-u_phoneme_entropy",
        "c5-cohort-c": "c5phone-c_surprisal + c5phone-c_entropy + c5phone-c_phoneme_entropy",
        "c5-cohort-u-0v1_": (
            "c5phone-u_surprisal-p0 + c5phone-u_entropy-p0 + c5phone-u_phoneme_entropy-p0 + "
            "c5phone-u_surprisal-p1_ + c5phone-u_entropy-p1_ + c5phone-u_phoneme_entropy-p1_"
        ),
        "c5-cohort-c-0v1_": (
            "c5phone-c_surprisal-p0 + c5phone-c_entropy-p0 + c5phone-c_phoneme_entropy-p0 + "
            "c5phone-c_surprisal-p1_ + c5phone-c_entropy-p1_ + c5phone-c_phoneme_entropy-p1_"
        ),
    }


# e = Appleseed(r"C:\Dataset\Appleseed_BIDS_new")
