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

# BASE = "gt-log8 + phone-0v12345"
BASE = "log-8 + phone"

def stem(h: int) -> str:
    return f"Earshot-LSTM-{h}-OneHot-M1K-train-hu-abs"

def sum_term(h: int) -> str:
    return f"{stem(h)}-sum"

def onset_term(h: int) -> str:
    return f"{stem(h)}-onset"

def both_terms(h: int) -> str:
    return f"{sum_term(h)} + {onset_term(h)}"

JOBS = [
    # baseline
    e.trf_job(BASE, **STG),

    # full models
    e.trf_job(f"{BASE} + {both_terms(512)}", **STG),
    e.trf_job(f"{BASE} + {both_terms(1024)}", **STG),
    e.trf_job(f"{BASE} + {both_terms(2048)}", **STG),

    # partial models (needed for @ unique contribution)
    e.trf_job(f"{BASE} + {sum_term(512)}", **STG),
    e.trf_job(f"{BASE} + {onset_term(512)}", **STG),

    e.trf_job(f"{BASE} + {sum_term(1024)}", **STG),
    e.trf_job(f"{BASE} + {onset_term(1024)}", **STG),

    e.trf_job(f"{BASE} + {sum_term(2048)}", **STG),
    e.trf_job(f"{BASE} + {onset_term(2048)}", **STG),
]


