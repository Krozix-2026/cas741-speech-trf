from burgundy import e

STG = {
    'raw': 'ica1-20',
    'samplingrate': 50,
    'cv': True,
    'partitions': -4,
    'inv': 'fixed-6-MNE-0',
    'mask': 'superiortemporal',
    'tstart': -0.100,
    'tstop': 1.000,
    'error': 'l2',
    'selective_stopping': 1,
}

BASE = "gt-log8 + phone-0v12345"

# PHCTC_STEM = "Earshot-PhCTC-LibriSpeechGT-64ch-BiLSTM512-L1-ep39-hu-abs"
PHCTC_STEM = "Earshot-PhoneCTC-LibriSpeechGT-64ch-LSTM2048-L1-ep165-hu-abs"

SUM   = f"{PHCTC_STEM}-sum"
ONSET = f"{PHCTC_STEM}-onset"
BOTH  = f"{SUM} + {ONSET}"

JOBS = [
    e.trf_job(BASE, **STG),

    # full
    e.trf_job(f"{BASE} + {BOTH}", **STG),

    # partial (for unique contribution)
    e.trf_job(f"{BASE} + {SUM}", **STG),
    e.trf_job(f"{BASE} + {ONSET}", **STG),
]
