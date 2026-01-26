"""Phoneme-based predictors"""
from eelbrain import load, plot, Dataset

from phone import iter_realizations
from settings import SEGMENTED_LEXICON_PATH


lexicon = load.unpickle(SEGMENTED_LEXICON_PATH)
rows = []
for stim, r in iter_realizations():
    n = len(r.phones)
    up = lexicon.uniqueness_point(r.pronunciation, 'morpheme')
    rows.append([stim, n, up])
ds = Dataset.from_caselist(['item', 'n', 'up'], rows)
plot.Histogram('up', ds=ds)
plot.Histogram('n', ds=ds)
