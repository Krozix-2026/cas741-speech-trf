from eelbrain import *
from eelbrain.pipeline import *
from trftools.pipeline import *


COHORT_VARS = ['surprisal', 'entropy', 'phoneme_entropy']
CB_VARS = ['surprisal', 'entropy']


def phone_model(positions, names=('',)):
    names = [f'-{n}' if n else n for n in names]
    return " + ".join(f'phone{n}-p{p}' for n in names for p in positions)


MODELS = {
    ###################################################
    # Current Biology model + UP
    # --------------------------
    # components
    'gt8': "gammatone-8 + gammatone-edge-8",
    'gt-log8': "gammatone-log-8 + gammatone-log-edge-8",
    # phone onsets
    'phone-0v12345': phone_model(['0', '12345']),
    # 'phone-0v1v2v3v4v5': phone_model(range(6)),
    # 'phone-0v1v2345': phone_model(['0', '1', '2345']),
    # cohort
    'cohort': "phone-surprisal + phone-entropy + phone-phoneme_entropy",
    'cb-cohort': "phone-surprisal + phone-entropy",
    'cohort-0v12345': phone_model(['0', '12345'], COHORT_VARS),
    'cb-cohort-0v12345': phone_model(['0', '12345'], CB_VARS),
    'cb-cohort-0': phone_model(['0'], CB_VARS),
    'cb-cohort-12345': phone_model(['12345'], CB_VARS),
}
to_step = [
    'phone-0v12345',
    'cohort',
    'cohort-0v12345',
    'cohort-0',
    'cohort-12345',
    'cohort-1v2345',
    'phone-surprisal',
    'phone-surprisal-p0',
    'phone-surprisal-p12345',
    'phone-entropy',
    'phone-entropy-p0',
    'phone-entropy-p12345',
]
for x in to_step:
    if x in MODELS:
        # x is a model
        x_step = ' + '.join([f'{t}-step' for t in MODELS[x].split(' + ')])
        MODELS[f'{x}-step'] = x_step
        MODELS[f'{x}-is'] = f"{MODELS[x]} + {x_step}"
    else:
        # x is a term
        MODELS[f'{x}-is'] = f"{x} + {x}-step"

MODELS['cb-cohort-syllable'] = 'phone-surprisal-monosyllabic + phone-surprisal-multisyllabic + phone-entropy-monosyllabic + phone-entropy-multisyllabic'

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

# Epochs for subset of words
SUB_EPOCHS = {
    f'cont-{n}': ContinuousEpoch('Burgundy', f"(trialType == 'item') & (trialType.count('item') < {n})", 1, 2, samplingrate=100)
    for n in range(200, 1000, 100)
}


class Burgundy(TRFExperiment):

    auto_delete_cache = 'ask'
    screen_log_level = 'debug'

    sessions = ('Burgundy', 'emptyroom')

    defaults = {
        'epoch': 'cont',
        'rej': '',
        'cov': 'emptyroom',
        'raw': 'ica1-20',
        'inv': 'fixed-1-MNE-0',
        'group': 'good2',
    }

    raw = {
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
        '0-40': RawFilter('tsss', 0, 40, cache=False),
        '1-40': RawFilter('tsss', 1, 40, cache=False),
        'ica': RawICA('1-40', 'Burgundy', 'extended-infomax', n_components=0.99),
        'ica0-40': RawApplyICA('0-40', 'ica'),
        'ica1-20': RawFilter('ica', None, 20),
    }

    variables = {
        'trialType': LabelVar('trigger', {162: 'item', 163: 'item', 166: 'probe', 167: 'probe'}),
        'status': LabelVar('trigger', {162: 'good_item', 163: 'post_probe', 166: 'no_probe', 167: 'yes_probe'}),
    }

    epochs = {
        'all': PrimaryEpoch('Burgundy', "trialType == 'item'", tmax=1.5, decim=2, n_cases=1000),
        'cont': ContinuousEpoch('Burgundy', "trialType == 'item'", 1, 2, samplingrate=100),
        **SUB_EPOCHS,
        'not_post_probe': SecondaryEpoch(base='all', sel="status == 'good_item'")}

    groups = {
        # 2645: extreme noise in data
        'okay-data': SubGroup('all', exclude=['R2645']),
        # 2627: fatigue and an earbud falling out
        'good-data': SubGroup('all', exclude=['R2627', 'R2645']),
        # 2349, 2636, 2637 excluded due to accuracy less than 1 SD from mean
        'good': SubGroup('good-data', exclude=['R2349', 'R2636', 'R2637']),
        # 2646: Many broken/flat sensors; outlier surprisal response
        'good2': SubGroup('good', exclude=['R2646']),
        'good2-data': SubGroup('good-data', exclude=['R2646']),
    }

    stim_var = 'item'

    predictors = {
        # sound
        'gammatone': FilePredictor('bin'),
        # lexical
        'phone': FilePredictor(columns=True),
        'word': FilePredictor(),
        # Earshot
        'Earshot': SessionPredictor('bin'),
        # 'RNN': SessionPredictor('bin'),
    }

    models = MODELS

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

    def load_log(self):
        path = self.format('{root}/meg/{subject}/{subject}-all_stims.log')
        return load.txt.tsv(path, ['subject', 'trial_number', 'event', 'item'], delimiter='\t', skiprows=5, ignore_missing=True)

    def label_events(self, ds):
        if ds.n_cases > 1:
            # get the log file and add the item
            log_ds = self.load_log()
            log_ds[:, 'response'] = 'x'
            for i in range(0, log_ds.n_cases):
                if log_ds[i, 'event'] == 'Response':
                    log_ds[i-1, 'response'] = log_ds[i, 'item']
            log_ds = log_ds.sub("event != 'Response'")
            ds['item'] = log_ds['item']
            ds['response'] = log_ds['response']
        return ds

    @staticmethod
    def get_accuracy():
        accuracies = []
        for subject in e:
            ds = e.load_events()
            probes = ds.sub("trialType == 'probe'")
            accurate = 0 
            for i in range(0,probes.n_cases): 
                if probes['status'][i] == 'yes_probe': 
                    if probes['response'][i] == '2':
                        accurate = accurate + 1 
                if probes['status'][i] == 'no_probe': 
                    if probes['response'][i] == '1':
                        accurate = accurate + 1
            accuracies.append((subject, accurate))

        return Dataset.from_caselist(['subject', 'accurate'], accuracies)


e = Burgundy(r'C:\Dataset\Burgundy')
