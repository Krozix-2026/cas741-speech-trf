# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Phoneme-based predictors
#
# - Include masks for positions.
# - Phoneme 4 is the last important UP before the tail (see `show-grid`).

# +
from functools import reduce
from math import log10
import operator
from pathlib import Path

from eelbrain import *
import trftools

from settings import GRID_DIR, PREDICTORS_DIR, LEXICON_PATH
# -

# ## Generate predictors

lexicon = load.unpickle(LEXICON_PATH)

# +
KEYS = {
    'any': '',
    'surprisal': '-surprisal',
    'entropy': '-entropy',
}

def silence(t):
    return Dataset({
        'phone': Factor([' ']),
        'surprisal': Var([0]),
        'entropy': Var([0]),
        'phoneme_entropy': Var([0]),
        'any': Var([False]),
        'time': Var([t]),
        'pos': Var([-1]),
        'up': Var([False]),
    })

for grid_path in GRID_DIR.glob('*.TextGrid'):
    stim = grid_path.stem.lower()
    print(stim, end=', ')
    grid = trftools.align.TextGrid.from_file(grid_path, word_tier='word', phone_tier='phone')

    # stimulus info
    info = {'tstop': (grid.n_samples + 1) * grid.tstep - grid.tmin, 'stim': stim}

    # process realizations
    realizations = list(grid.realizations)
    dss = []
    r = realizations.pop(0)
    
    # pre-silence
    if r.graphs == ' ':
        assert len(r.times) == 1
        dss.append(silence(r.times[0]))
        r = realizations.pop(0)
    
    # main word
    assert r.phones[0] != ' '
    r = r.strip_stress()
    ds = Dataset({
        'phone': Factor(r.phones),
        'surprisal': Var(lexicon.surprisal(r.pronunciation)),
        'entropy': Var(lexicon.entropy(r.pronunciation)),
        'phoneme_entropy': Var(lexicon.phoneme_entropy(r.pronunciation)),
    })
    ds[:, 'any'] = True
    ds.index('pos')
    ds['time'] = Var(r.times)
    ds['up'] = ds['pos'] == lexicon.uniqueness_point(r.pronunciation, 'morpheme')
    dss.append(ds)

    #post-silence
    if realizations:
        r = realizations.pop(0)
        assert not realizations
        assert r.graphs == ' '
        assert len(r.times) == 1
        dss.append(silence(r.times[0]))
    else:
        dss.append(silence(info['tstop']))
        info['tstop'] += 0.100

    # combine realizations
    item = combine(dss)
    item.info.update(info)
    # word-level variables
    words = lexicon.lookup(stim.upper())
    assert len(words) == 1
    count = sum(word.activation for word in words)
    item[:, 'log10wf'] = 6.33 - log10(count)
    # masks
    index = item['pos']
    for pos in range(5):
        item[f'p{pos}'] = Var(index == pos)
    item['p5'] = Var(index > 4)
    for ps in ['01234', '12345', '2345']:
        item[f'p{ps}'] = reduce(operator.or_, (item[f'p{p}'] for p in ps))
    # word offset
    item['offset'] = Var([i == item.n_cases - 1 for i in range(item.n_cases)])

    # save
    dst = PREDICTORS_DIR / f'{stim}|phone.pickle'
    dst.rename(PREDICTORS_DIR / f'{stim}|phone-bkp.pickle')
    save.pickle(item, dst)
# -

# ## Add n-syllable mask

# +
DATA_DIR = Path('.').resolve().parent / 'data'

n_ds = load.tsv(DATA_DIR / 'word n-syllables.txt')
# -

for word, n_syllables in n_ds.zip('word', 'n_syllables'):
    path = PREDICTORS_DIR / f'{word}~phone.pickle'
    ds = load.unpickle(path)
    is_multisyllabic = n_syllables > 1
    ds[:, 'monosyllabic'] = not is_multisyllabic
    ds[:, 'multisyllabic'] = is_multisyllabic
    save.pickle(ds, path)

ds

# # `Maintenance`

# ## Compare with backup

stims = ['sloth', 'taint', 'straddle', 'turquoise']
doc = fmtxt.FMText()
for stim in stims:
    new = load.unpickle(PREDICTORS_DIR / f'{stim}|phone.pickle')
    bkp = load.unpickle(PREDICTORS_DIR / f'{stim}|phone-bkp.pickle')
    tables = fmtxt.FloatingLayout([new, bkp])
    doc.append(fmtxt.Section(stim.capitalize(), tables))
doc

all_stims = [grid_path.stem.lower() for grid_path in GRID_DIR.glob('*.TextGrid')]
ds_old = combine([load.unpickle(PREDICTORS_DIR / f'{stim}|phone-bkp.pickle') for stim in all_stims])
ds_new = combine([load.unpickle(PREDICTORS_DIR / f'{stim}|phone.pickle') for stim in all_stims])

plot.Scatter(ds_new['phoneme_entropy'], ds_old['phoneme_entropy'])
