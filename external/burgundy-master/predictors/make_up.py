"""Phoneme-based predictors"""
from eelbrain import load, save, Dataset, Var

from phone import iter_words
from settings import PREDICTORS_DIR, SEGMENTED_LEXICON_PATH


lexicon = load.unpickle(SEGMENTED_LEXICON_PATH)

DO = {
    'up': ('morpheme', 1),
    'fup': ('form', 1),
    '9p': ('morpheme', 9),
    'f9p': ('form', 9),
}


for item in iter_words():
    r = item.info['r']
    for key, (model, n) in DO.items():
        dst = PREDICTORS_DIR / f'{item.name}|phone-{key}.pickle'
        if dst.exists():
            continue
        up = lexicon.uniqueness_point(r.pronunciation, model, n)
        ds = Dataset({
            'time': item['time'],
            'value': Var([i == up for i in range(item.n_cases)]),
        })
        save.pickle(ds, dst)
