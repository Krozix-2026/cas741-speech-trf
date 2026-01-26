# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from burgundy import e


WHOLEBRAIN = {
    'raw': 'ica1-20',
    'samplingrate': 50,
    'cv': True,
    'partitions': -4,
    'inv': 'fixed-6-MNE-0',
    'mask': 'wholebrain',
    'tstart': -0.100,
    'tstop': 1.000,
    'error': 'l2',
    'selective_stopping': 1,
}
STG = {**WHOLEBRAIN, 'mask': 'superiortemporal'}
STG_TO_WHOLEBRAIN = {**WHOLEBRAIN, 'mask': 'stg_to_wholebrain'}

# Power ~ n words
JOBS = [
    e.model_job(f"gt-log8 + phone-0v12345 +@ phone-surprisal", **STG, epoch=f'cont'),
]
JOBS += [
    e.model_job(f"gt-log8 + phone-0v12345 +@ phone-surprisal", **STG, epoch=f'cont-{n}')
    for n in range(200, 1000, 100)
]
