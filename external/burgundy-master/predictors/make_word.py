"""Phoneme-based predictors"""
from math import log10
from eelbrain import load, save, Dataset, Var

from phone import iter_words
from settings import PREDICTORS_DIR, SEGMENTED_LEXICON_PATH


lexicon = load.unpickle(SEGMENTED_LEXICON_PATH)
LOCATIONS = {
    '': None,
    '-up': ('morpheme', 1),
    '-fup': ('form', 1),
    # '9p': ('morpheme', 9),
    # 'f9p': ('form', 9),
}

for item in iter_words():
    words = lexicon.lookup(item.name)
    assert len(words) == 1
    count = sum(word.subtlex_count for word in words)
    value = 6.33 - log10(count)
    print(f" {item.name.ljust(10)}: {value:.2f}")
    r = item.info['r']
    for key, loc in LOCATIONS.items():
        dst = PREDICTORS_DIR / f'{item.name}|word-log10wf{key}.pickle'
        if dst.exists():
            continue
        # find time
        if loc is None:
            up = 0
        else:
            up = lexicon.uniqueness_point(r.pronunciation, *loc)
        if up == len(r.phones):
            t = r.tstop
        else:
            t = item[up, 'time']
        # save
        ds = Dataset({
            'time': Var([t]),
            'value': Var([value]),
        })
        save.pickle(ds, dst)
