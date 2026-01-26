"""Jobs with cross-validation"""
from appleseed import e


LTL = {
    'raw': 'ica-20',
    'samplingrate': 100,
    'inv': 'fixed-6-MNE-0',
    'mask': 'lateraltemporal',
    'tstart': -0.100,
    'tstop': 1.000,
    'cv': True,
    'partitions': -4,
    'selective_stopping': 1,
}
FTP = {**LTL, 'mask': 'ftp'}
STL = {**LTL, 'mask': 'superiortemporal'}
FTP_5 = {**FTP, 'partitions': -5}
STL_5 = {**STL, 'partitions': -5}

FULL = 'gte8 + phone-p0 + phone-p1_ + phone-surprisal + phone-entropy'
JOBS = [
    e.model_job(f'{FULL} @ phone-surprisal', epoch='apple', **STL),
    e.model_job(f'{FULL} @ phone-entropy', epoch='apple', **STL),
    e.model_job(f'{FULL} @ phone-surprisal', epoch='seg1-2', **FTP_5),
    e.model_job(f'{FULL} @ phone-entropy', epoch='seg1-2', **FTP_5),
    e.model_job(f'{FULL} @ phone-surprisal', epoch='seg5-6', **STL_5),
    e.model_job(f'{FULL} @ phone-entropy', epoch='seg5-6', **STL_5),
    e.model_job(f'{FULL} @ phone-surprisal', epoch='seg7-8', **STL_5),
    e.model_job(f'{FULL} @ phone-entropy', epoch='seg7-8', **STL_5),
]
