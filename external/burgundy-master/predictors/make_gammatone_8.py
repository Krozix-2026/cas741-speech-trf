"""Predictors based on gammatone spectrograms"""
from pathlib import Path

import numpy as np
from eelbrain import *
from cssl.regressors._dynamics import detect_edges


DATA_ROOT = Path('~/Data/burgundy').expanduser()
GAMMATONE_DIR = DATA_ROOT / 'Gammatone'
PREDICTORS_DIR = DATA_ROOT / 'predictors'


for gt_path in GAMMATONE_DIR.glob('*.pickle'):
    name = gt_path.stem.lower()
    gt = load.unpickle(gt_path)

    # pre-processing
    gt = gt.clip(0, out=gt)

    # gt, tag = gt ** 0.6, ''
    gt, tag = (gt + 1).log(), '-log'

    gte = detect_edges(gt, c=30)

    # envelope
    # gt1 = gt.sum('frequency')
    # save.pickle(gt1, PREDICTORS_DIR / f'{name}|gammatone{tag}-1.pickle')
    # gt1o = gt1.diff('time').clip(0)
    # save.pickle(gt1o, PREDICTORS_DIR / f'{name}|gammatone{tag}-1-hwrd.pickle')
    # gte1 = gte.sum('frequency')
    # save.pickle(gte1, PREDICTORS_DIR / f'{name}|gammatone{tag}-edge-1.pickle')

    # 8 bins
    gt8 = gt.bin(nbins=8, func=np.sum, dim='frequency')
    save.pickle(gt8, PREDICTORS_DIR / f'{name}~gammatone{tag}-8.pickle')
    gt8o = gt8.diff('time').clip(0)
    save.pickle(gt8o, PREDICTORS_DIR / f'{name}~gammatone{tag}-8-hwrd.pickle')
    gte8 = gte.bin(nbins=8, func=np.sum, dim='frequency')
    save.pickle(gte8, PREDICTORS_DIR / f'{name}~gammatone{tag}-edge-8.pickle')
