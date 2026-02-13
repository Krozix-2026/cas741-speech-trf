from pathlib import Path
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths

from eelbrain import load, combine, Factor, Var
from eelbrain.pipeline import *
from trftools.pipeline import *


DIR = Path(__file__).parent
STIMULI = load.tsv(DIR / 'appleseed_stimuli.txt', types='fv')
STIMULI_PILOTS = STIMULI.sub("stimulus != '11'").repeat(2)
STIMULI_REAL = STIMULI.sub("stimulus != '11b'").repeat(2)


STIM_IDS = np.array(STIMULI['stimulus'], dtype=str)
STIM_LEN = np.array(STIMULI['length'], dtype=float)

# Parcellations
LATERAL_TEMPORAL = ('transversetemporal', 'superiortemporal', 'bankssts', 'middletemporal', 'inferiortemporal', 'temporalpole')
IFG = ('parsopercularis', 'parsorbitalis', 'parstriangularis')
MFG = ('caudalmiddlefrontal', 'rostralmiddlefrontal')
LATERAL_FRONTAL = ('caudalmiddlefrontal', 'frontalpole', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 'precentral', 'rostralmiddlefrontal', 'superiorfrontal')
LATERAL_PARIETAL = ('postcentral', 'inferiorparietal', 'superiorparietal', 'supramarginal')
OTHER_MEDIAL = ('cuneus', 'lateralorbitofrontal', 'medialorbitofrontal', 'paracentral', 'precuneus', 'fusiform')
FTP_LATERAL = set(LATERAL_FRONTAL + LATERAL_TEMPORAL + LATERAL_PARIETAL + OTHER_MEDIAL)
# Occipital lobe complemtary with FTP:
OCCIPITAL = ('lateraloccipital', 'pericalcarine', 'lingual')


class Appleseed(TRFExperiment):

    auto_delete_cache = 'delete'
    screen_log_level = 'debug'


    
    defaults = {
        # 'session': 'appleseed',
        'task': 'Appleseed',
        'split': '01',
        # 'raw': 'ica',
        'raw': '1-40',
        'rej': '',
        'cov': 'emptyroom',
        'inv': 'fixed-6-MNE-0',
    }

    # sessions = ['Appleseed', 'emptyroom', 'tone']

    groups = {
        'righthanders': SubGroup('all', ['11']),
    }

    raw = {
        'raw': RawSource(connectivity='auto'),
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=.9, st_only=True),
        '1-40': RawFilter('tsss', 1, 40),
        'ica': RawICA('1-40', 'Appleseed', n_components=0.99),
        'ica-20': RawFilter('ica', None, 20, cache=False),
        'tsss-ica': RawApplyICA('tsss', 'ica', cache=False),
        'ica-0-20': RawFilter('tsss-ica', None, 20, cache=False),
    }

    parcs = {
        'lateraltemporal': SubParc('aparc', LATERAL_TEMPORAL, views='lateral'),
        'superiortemporal': SubParc('aparc', ('superiortemporal', 'transversetemporal')),
        'ftp': SubParc('aparc', sorted(FTP_LATERAL)),
        'temporal_to_ftp': SubParc('aparc', sorted(FTP_LATERAL.difference(LATERAL_TEMPORAL))),
        'stg_to_ftp': SubParc('aparc', sorted(FTP_LATERAL.difference(['transversetemporal', 'superiortemporal']))),
        'occipital': SubParc('aparc', OCCIPITAL),
        'ifg': SubParc('aparc', IFG),
        'lhg': SubParc('aparc', ('transversetemporal-lh',), views='lateral'),
        
    }

    variables = {'event': LabelVar('trigger', {162: 'onset', 167: 'offset'})}

    # def fix_events(self, ds):
    #     if ds.info['subject'] == 'R2676' and ds.info['session'] == 'Appleseed':
    #         return combine([ds[:10], ds[11:]])
    #     return ds

    # def label_events(self, ds):
    #     if ds.info['session'] == 'Appleseed':
    #         if ds.info['subject'] in ('R2650', 'R2652'):
    #             ds.update(STIMULI_PILOTS)
    #         else:
    #             ds.update(STIMULI_REAL)
    #     return ds


    
    def make_cov(self, *args, **kwargs):
        try:
            return super().make_cov(*args, **kwargs)
        except ValueError as e:
            if "task='emptyroom'" not in str(e):
                raise

        root = Path(self.root)

        cov_path = self.get('cov-file', make=False)

        # 直接找 sub-emptyroom 下所有 meg fif（你的数据是 sub-emptyroom/ses-.../meg/*.fif）
        candidates = sorted(root.glob("sub-emptyroom/**/meg/*_meg.fif"))
        if not candidates:
            # 有些数据可能是 *_meg.fif.gz 或者命名略不同
            candidates = sorted(root.glob("sub-emptyroom/**/meg/*.fif"))
        if not candidates:
            raise FileNotFoundError(f"No empty-room MEG FIF found under: {root/'sub-emptyroom'}")

        # 先选一个最“新”的 session（通常 ses-YYYYMMDD，字典序≈时间序）
        er_fif = candidates[-1]
        
        raw_er = mne.io.read_raw_fif(er_fif, preload=False, verbose="ERROR")
        raw_er.load_data()
        raw_er.filter(1.0, 40.0, verbose="ERROR")

        cov = mne.compute_raw_covariance(raw_er, method="empirical", verbose="ERROR")
        mne.write_cov(cov_path, cov, overwrite=True)
        print(f"[INFO] empty-room cov built from {er_fif} -> {cov_path}")

        return cov_path
        
        
        
    def fix_events(self, ds):
        if ds.info['subject'] == 'R2676' and ds.info.get('task') == 'Appleseed':
            return combine([ds[:10], ds[11:]])
        return ds

    # def label_events(self, ds):
    #     if ds.info.get('task') == 'Appleseed':
    #         if ds.info['subject'] in ('R2650', 'R2652'):
    #             ds.update(STIMULI_PILOTS)
    #         else:
    #             ds.update(STIMULI_REAL)
    #     return ds
    def load_epochs(self, *args, **kwargs):
        # TRF pipeline 在 Windows 上更稳：不要把 bad channels 列塞进 ds
        kwargs.setdefault('add_bads', False)

        ds = super().load_epochs(*args, **kwargs)

        # ---------- DEBUG: print only once per process ----------
        if not getattr(self, "_printed_source_debug", False):
            self._printed_source_debug = True
            print("\n========== [DEBUG] load_epochs() sanity ==========")
            print("ds.n_cases =", len(ds))
            print("ds.info keys =", sorted(ds.info.keys()))
            print("subject =", ds.info.get("subject"))
            print("task =", ds.info.get("task"))
            print("session =", ds.info.get("session"))
            print("has trigger?", "trigger" in ds)
            if "trigger" in ds:
                # 看看 event 标记是不是你预期的 162/167
                u = np.unique(np.array(ds["trigger"]))
                print("unique triggers (first 20) =", u[:20])

            # 关键：看看 brain 数据是什么、是不是 source
            # 在很多 eelbrain pipeline 里，MEG/EEG 数据字段可能叫 'meg'/'eeg'/'src'/'source'
            for k in ("src", "source", "meg", "eeg"):
                if k in ds:
                    y = ds[k]
                    print(f"\n[DEBUG] ds['{k}'] =", y)
                    try:
                        print("dims =", y.dims)
                    except Exception as e:
                        print("dims: <failed>", e)

                    # 如果是 source NDVar，这里通常能拿到 source 维度对象
                    try:
                        sd = y.get_dim("source")
                        print("source dim =", sd)
                    except Exception as e:
                        print("get_dim('source'): <failed>", e)
                    break

            print("=================================================\n")

        return ds

    
    def load_predictor(self, *args, **kwargs):
        x = super().load_predictor(*args, **kwargs)

        data = x.x  # numpy array
        # 1) non-finite -> 0
        bad = ~np.isfinite(data)
        if bad.any():
            print(f"[WARN] non-finite in predictor: {args[0]} -> replacing with 0")
            x = x.copy()
            x.x[bad] = 0.0
            data = x.x

        # 2) flat check: std==0 along time axis
        # 这里假设 data shape 是 (time, features) 或 (features, time)。
        # eelbrain 的 NDVar 通常 time 是第一个维度；保险起见用 x.get_dim('time') 方式更精确，但先用简单的：
        arr = data
        # 如果是 2D：time x feature
        if arr.ndim == 2:
            std0 = arr.std(axis=0) == 0
            if std0.any():
                idx = np.where(std0)[0]
                print(f"[WARN] flat predictor columns in {args[0]}: n={len(idx)} -> {idx[:10]} ...")
        return x
    
    
    def label_events(self, ds):
        # 1) 只保留 onset trigger（避免 162/167 混在一起造成行数翻倍）
        #    如果你确认数据里只有 162，也可以删掉这行
        if 'trigger' in ds:
            ds = ds.sub("trigger == 162")

        # 2) 用 SOA 匹配最近的 stimulus length
        soa = np.array(ds['SOA'], dtype=float)
        idx = np.abs(soa[:, None] - STIM_LEN[None, :]).argmin(axis=1)

        stim = STIM_IDS[idx]
        length = STIM_LEN[idx]

        # 3) 容错：SOA 偏差太大就标记 UNKNOWN（避免 silent wrong labeling）
        diff = np.abs(soa - length)
        bad = diff > 5.0   # 5 秒阈值你可以调；你之前见过 0.9~1.3s 的偏差是正常的
        if bad.any():
            stim = stim.astype(object)
            stim[bad] = 'UNK'
            length = length.copy()
            length[bad] = np.nan

            # 想看具体是哪些行 mismatch，就打印几条
            for t, s, d in zip(np.array(ds['time'])[bad][:10], soa[bad][:10], diff[bad][:10]):
                print(f"[WARN] SOA mismatch: time={t:.3f}s SOA={s:.3f} diff={d:.3f}s -> UNK")

        # 4) 写回 ds（这一步不会要求行数相等，所以不会再出现 27 vs 22）
        ds['stimulus'] = Factor(stim)
        ds['length'] = Var(length)

        return ds
    
    epochs = {
        # 'apple': PrimaryEpoch('Appleseed', "event == 'onset'", tmin=0, tmax='length', samplingrate=100),
        'apple': PrimaryEpoch('Appleseed', "event == 'onset'", tmin=0, tmax='length', samplingrate=100),
        'seg1-2': SecondaryEpoch('apple', "stimulus.isin(('1', '2'))"),
        'seg5-6': SecondaryEpoch('apple', "stimulus.isin(('5', '6'))"),
        'seg7-8': SecondaryEpoch('apple', "stimulus.isin(('7', '8'))"),
        'seg1-5': SecondaryEpoch('apple', "stimulus.isin(('1', '2', '3', '4', '5'))"),
        'seg6-11': SecondaryEpoch('apple', "stimulus.isin(('6', '7', '8', '9', '10', '11', '11b'))"),
    }

    predictors = {
        # 连续声学特征：用滤波+decimate 更合理
        'gammatone': FilePredictor(resample='resample', sampling='continuous'),

        # 事件/离散类（惊讶度、熵、phone impulse 等）：binning 更合理
        'phonotactic': FilePredictor(columns=True, resample='bin', sampling='discrete'),
        'phone': FilePredictor(columns=True, resample='bin', sampling='discrete'),
        'c5phone': FilePredictor(columns=True, resample='bin', sampling='discrete'),
    }

    models = {
        'gte8': "gammatone-8 + gammatone-edge30-8",
        'phonotactics': "phonotactic-surprisal + phonotactic-entropy",
        'c5-phone-0v1_': "c5phone-p0 + c5phone-p1_",
        'c5-cohort-u': "c5phone-u_surprisal + c5phone-u_entropy + c5phone-u_phoneme_entropy",
        'c5-cohort-c': "c5phone-c_surprisal + c5phone-c_entropy + c5phone-c_phoneme_entropy",
        'c5-cohort-u-0v1_': "c5phone-u_surprisal-p0 + c5phone-u_entropy-p0 + c5phone-u_phoneme_entropy-p0 + c5phone-u_surprisal-p1_ + c5phone-u_entropy-p1_ + c5phone-u_phoneme_entropy-p1_",
        'c5-cohort-c-0v1_': "c5phone-c_surprisal-p0 + c5phone-c_entropy-p0 + c5phone-c_phoneme_entropy-p0 + c5phone-c_surprisal-p1_ + c5phone-c_entropy-p1_ + c5phone-c_phoneme_entropy-p1_",
    }

# e = Appleseed(r"C:\Dataset\Appleseed-BIDS")
