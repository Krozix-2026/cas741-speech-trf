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

from eelbrain import *
import trftools

from settings import PREDICTORS_DIR
# -

paths = PREDICTORS_DIR.glob('*|phone.pickle')
ds = combine([load.unpickle(path) for path in paths])
ds = ds.sub("pos >= 0")

# +
from itertools import combinations

def scatter_table(xs, ds, alpha=0.5, line_top=0):
    t = fmtxt.Table('l' * (len(xs)-1))
    for i, y in enumerate(xs):
        for x in xs[i+1:]:
            p = plot.Scatter(y, x, 'pos', markers='.', ds=ds, alpha=alpha)
            if line_top:
                p._axes[0].plot([0, line_top], [0, line_top], color='k')
            t.cell(p.image(format='png'))
        if i == len(xs) - 2:
            t.cell(p.plot_colorbar(orientation='vertical', h=3.5, width=0.2).image())
        t.endline()
    return t


# -

xs = ['surprisal', 'entropy', 'phoneme_entropy']
scatter_table(xs, ds, 0.2, line_top=5)

# # Appleseed

# +
STIMULUS_DIR = Path('~/iCloud/Research/Appleseed/stimuli').expanduser()
GRID_DIR = STIMULUS_DIR / 'mfa' / '1'
STIMULI = [*map(str, range(1, 12)), '11b']

dss = []
for stim in STIMULI:
    grid = trftools.align.TextGrid.from_file(GRID_DIR / f'segment_{stim}.TextGrid')
    words = [r for r in grid.realizations if r.pronunciation != ' ']
    ds = Dataset()
    ds['word'] = Factor([r.graphs for r in words])
    ds['n_phones'] = Var([len(r.phones) for r in words])
    ds[:, 'stim'] = stim
    dss.append(ds)
ds = combine(dss)
p = plot.Histogram('n_phones', ds=ds, bins=[x-0.45 for x in range(14)])
p.figure.axes[0].grid()
