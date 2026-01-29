from eelbrain.pipeline import *
from trftools.pipeline import *
from eelbrain import Factor, load

from pathlib import Path
import os

DATA_ROOT = Path(os.environ.get("ALICE_ROOT", r"C:\Dataset\Appleseed")).expanduser()


MODELS = {
    # --- acoustic baselines ---
    'gt-log1': "gammatone-1 + gammatone-on-1",
    'gt-log8-edge30': "gammatone-8 + gammatone-edge30-8",
    'gt-log8-on':     "gammatone-8 + gammatone-on-8",
    'gt-lin8-edge30': "gammatone-lin-8 + gammatone-edge30-8",
    'gt-pow8-edge30': "gammatone-pow-8 + gammatone-edge30-8",

    # --- phone-level predictors (phone.pickle 里真实存在的列) ---
    'phone-onsets': "phone-any",             # 所有 phone onset (如果 any 是脉冲序列/强度)
    'phone-onsets-pos': "phone-p0 + phone-p1_",  # 位置拆分（首音素 vs 其余）
    'phone-cohort': "phone-surprisal + phone-entropy + phone-phoneme_entropy",
    'phone-cb': "phone-surprisal + phone-entropy",

    # --- word-level predictors ( word.pickle 里真实存在的列) ---
    'word-basic': "word-unigram_surprisal + word-surprisal + word-entropy",

    # --- useful combos ---
    'gt+phone-cohort': "gammatone-8 + gammatone-edge30-8 + phone-surprisal + phone-entropy + phone-phoneme_entropy",
    'gt+word': "gammatone-8 + gammatone-edge30-8 + word-unigram_surprisal + word-surprisal + word-entropy",
    'gt+phone+word': "gammatone-8 + gammatone-edge30-8 + phone-surprisal + phone-entropy + phone-phoneme_entropy + word-unigram_surprisal + word-surprisal + word-entropy",
}


MODELS['rnn'] = "RNN-sum + RNN-onset"


STG = ('transversetemporal', 'superiortemporal')
LATERAL_TEMPORAL = STG + ('bankssts', 'middletemporal', 'inferiortemporal')
LATERAL_FRONTAL = ('caudalmiddlefrontal', 'frontalpole', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 'precentral', 'rostralmiddlefrontal', 'superiorfrontal')
LATERAL_PARIETAL = ('postcentral', 'inferiorparietal', 'superiorparietal', 'supramarginal')
OTHER_TEMPORAL = ('fusiform', 'temporalpole')
OTHER_MEDIAL = ('cuneus', 'lateralorbitofrontal', 'medialorbitofrontal', 'paracentral', 'precuneus')

OCCIPITAL = ('lateraloccipital', 'pericalcarine', 'lingual')
MEDIAL_TEMPORAL = ('entorhinal', 'parahippocampal')

# Combinations
LATERAL = LATERAL_TEMPORAL + OTHER_TEMPORAL + LATERAL_FRONTAL + LATERAL_PARIETAL
WHOLEBRAIN = LATERAL_TEMPORAL + OTHER_TEMPORAL + LATERAL_FRONTAL + LATERAL_PARIETAL + OTHER_MEDIAL
WHOLEBRAIN_2 = WHOLEBRAIN + OCCIPITAL + MEDIAL_TEMPORAL




STIMULI = [*map(str, range(1, 12)), '11b']  # 12 segments: 1..11, 11b


SEGMENT_DURATION = {
    '1': 224.6, '2': 320.3, '3': 213.4, '4': 210.1, '5': 235.4, '6': 270.0,
    '7': 231.7, '8': 250.1, '9': 331.4, '10': 232.0, '11': 284.9, '11b': 205.5,
}

class Appleseed(TRFExperiment):
    sessions = ('Appleseed', 'emptyroom')
    stim_var = 'stimulus'
    
    screen_log_level = 'debug'

    
    defaults = {
        'epoch': 'cont',
        'rej': '',
        'cov': 'emptyroom',
        'raw': 'noica1-20',
        'inv': 'fixed-1-MNE-0',
        'group': 'all',
    }
    
    # ... raw/predictors/models/parcs 先照搬/再慢慢精简
    raw = {
        # 'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
        # '0-40': RawFilter('tsss', 0, 40, cache=False),
        # '1-40': RawFilter('tsss', 1, 40, cache=False),
        # 'ica': RawICA('1-40', 'Appleseed', 'extended-infomax', n_components=0.99),
        # 'ica0-40': RawApplyICA('0-40', 'ica'),
        # 'ica1-20': RawFilter('ica0-40', None, 20),
        '0-40': RawFilter('raw', 0, 40, cache=False),
        'noica1-20': RawFilter('raw', None, 20, cache=False),
        # 'noica1-20': RawFilter('tsss', None, 20),
    }
    
    variables = {
        # 'trialType': LabelVar('trigger', {...}),
    }
    
    epochs = {
        # 这里的 selection expression 需要你根据实际 events 改
        'cont': ContinuousEpoch('Appleseed', "(trigger == 162) & (SOA > 180)", 1, 2, samplingrate=100),
    }
    
  
    
    predictors = {
        'gammatone': FilePredictor('bin'),
        'phone': FilePredictor(columns=True),
        'word': FilePredictor(columns=True),# 建议打开 columns=True，才能用 word-LogFreq 等
        'phonotactic': FilePredictor(columns=True),
        'punctuation': FilePredictor(columns=True),

        # c5phone/c5word：
        'c5phone': FilePredictor(columns=True),
        'c5word': FilePredictor(columns=True),
        'c5word_up': FilePredictor(columns=True),
    }
        
    parcs = {
        'superiortemporal': SubParc('aparc', STG),
        'lateraltemporal': SubParc('aparc', LATERAL_TEMPORAL),
        'wholebrain': SubParc('aparc', WHOLEBRAIN),
        'wholebrain-2': SubParc('aparc', WHOLEBRAIN_2),
        'stg_to_wholebrain': SubParc('aparc', sorted(set(WHOLEBRAIN).difference(STG))),
        'stg_to_wholebrain-2': SubParc('aparc', sorted(set(WHOLEBRAIN_2).difference(STG))),
        'wholebrain_to_2': SubParc('aparc', sorted(set(WHOLEBRAIN_2).difference(WHOLEBRAIN))),
        'lateral': SubParc('aparc', LATERAL),
        'stg_to_lateral': SubParc('aparc', sorted(set(LATERAL).difference(STG))),
    }

    def label_events(self, ds):
        # 避免被框架多次调用时重复贴标/重复打印
        if 'stimulus' in ds:
            return ds

        remaining = dict(SEGMENT_DURATION)
        stim = []
        last = ''

        for i in range(ds.n_cases):
            trig = int(ds[i, 'trigger'])
            soa = float(ds[i, 'SOA'])

            if trig == 162 and soa > 180:
                key = min(remaining, key=lambda k: abs(remaining[k] - soa))
                diff = abs(remaining[key] - soa)

                # 你的数据里出现 ~1s 的偏差很正常（padding/四舍五入/边界处理）
                # 放宽到 2s 或 3s 更合理
                if diff > 3.0:
                    print(f"[WARN] SOA mismatch: SOA={soa:.3f}, closest {key}={remaining[key]:.3f} (diff={diff:.3f})")

                last = key
                stim.append(key)
                remaining.pop(key)
            else:
                # 对 167 或者异常短的 162：不消耗 remaining，避免错位
                stim.append(last)

        ds['stimulus'] = Factor(stim)
        return ds



e = Appleseed(r'C:\Dataset\Appleseed')


if __name__ == "__main__":

    PRED = Path(r"C:\Dataset\Appleseed\predictors")
    ds_phone = load.unpickle(PRED / "1~phone.pickle")
    ds_word = load.unpickle(PRED / "1~c5word.pickle")
    print(load.unpickle(PRED / "1~phone.pickle").keys())
    print(load.unpickle(PRED / "1~c5word.pickle").keys())
    print("phone terms:", ["phone-" + k for k in ds_phone.keys() if k not in ("time","phone")])
    print("word terms:", ["word-" + k for k in ds_word.keys() if k not in ("time","word","grid_word")])

    e = Appleseed(r'C:\Dataset\Appleseed')
    ds = e.load_events()
    ds = e.label_events(ds)

    print(ds.keys())
    print(ds)
    print(sorted(set(ds['trigger'])))
    print(ds.sub("trigger == 162")[['T', 'stimulus']][:15])


