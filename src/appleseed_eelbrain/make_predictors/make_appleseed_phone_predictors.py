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
from itertools import chain
from math import log10
import operator
from pathlib import Path

import numpy
from eelbrain import *
import trftools
import cssl

from settings import LEXICON_PATH


PREDICTOR_DIR = Path('~/Data/Appleseed/predictors').expanduser() 
STIMULUS_DIR = Path('~/iCloud/Research/Appleseed/stimuli').expanduser()
GRID_DIR = STIMULUS_DIR / 'mfa' / '1'
STIMULI = [*map(str, range(1, 12)), '11b']

lexicon = load.unpickle(LEXICON_PATH)
# -

# # Count words

# +
word_count = 0
phoneme_count = 0
for stim in STIMULI:
    print(stim, end=', ')
    grid = trftools.align.TextGrid.from_file(GRID_DIR / f'segment_{stim}.TextGrid')
    grid = grid.strip_stress()
    
    word_count += sum(not r.is_silence() for r in grid.realizations)
    phoneme_count += sum(bool(p.strip()) for p in grid.phones)
    
print(f"\n{word_count} words; {phoneme_count} phonemes")


# -

# # Make predictor

# +
def r_pos(r):
    if r.pronunciation.strip():
        return range(len(r.phones))
    else:
        return [-1]


for stim in STIMULI:
    print(stim, end=', ')
    grid = trftools.align.TextGrid.from_file(GRID_DIR / f'segment_{stim}.TextGrid')
    grid = grid.strip_stress()
    ds = Dataset({
        'time': Var(grid.times),
        'phone': Factor(grid.phones),
        'surprisal': Var(lexicon.surprisal(grid, smooth=1)),
        'entropy': Var(lexicon.entropy(grid)),
        'phoneme_entropy': Var(lexicon.phoneme_entropy(grid)),
    }, info={'tstop': grid.tstop, 'segment': stim})
    pos = numpy.array(list(chain.from_iterable((r_pos(r) for r in grid.realizations))))
    # masks
    ds['any'] = Var(pos >= 0)
    ds['p0'] = Var(pos == 0)
    ds['p1_'] = Var(pos > 0)
    # save
    save.pickle(ds, PREDICTOR_DIR / f'{stim}|phone.pickle')


# +
# update positions without recomputing rest

# for stim in STIMULI:
#     print(stim, end=', ')
#     grid = trftools.align.TextGrid.from_file(GRID_DIR / f'segment_{stim}.TextGrid')
#     grid = grid.strip_stress()

#     ds = load.unpickle(PREDICTOR_DIR / f'{stim}|phone.pickle')
    
#     pos = numpy.array(list(chain.from_iterable((r_pos(r) for r in grid.realizations))))
#     # masks
#     ds['any'] = Var(pos >= 0)
#     ds['p0'] = Var(pos == 0)
#     ds['p1_'] = Var(pos > 0)

#     save.pickle(ds, PREDICTOR_DIR / f'{stim}|phone.pickle')
# -

ds.head()
