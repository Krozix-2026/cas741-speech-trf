# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline
from functools import reduce
from math import log10
import operator
from pathlib import Path
import os

from eelbrain import *
import trftools

DATA_ROOT = Path(os.environ.get("ALICE_ROOT", r"C:\Dataset\Appleseed")).expanduser()
STIMULUS_DIR = DATA_ROOT / "stimuli"
PREDICTORS_DIR = DATA_ROOT / "predictors"


from itertools import combinations


#Appleseed
STIMULI = [*map(str, range(1, 12)), '11b']
print("STIMULI:", STIMULI)

dss = []
for stim in STIMULI:
    grid = trftools.align.TextGrid.from_file(STIMULUS_DIR / f'segment_{stim}.TextGrid')
    words = [r for r in grid.realizations if r.pronunciation != ' ']
    ds = Dataset()
    ds['word'] = Factor([r.graphs for r in words])
    ds['n_phones'] = Var([len(r.phones) for r in words])
    ds[:, 'stim'] = stim
    dss.append(ds)
ds = combine(dss)
p = plot.Histogram('n_phones', ds=ds, bins=[x-0.45 for x in range(14)])
p.figure.axes[0].grid()
