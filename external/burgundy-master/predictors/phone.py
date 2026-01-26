"""Phoneme-based predictors"""
from eelbrain import load, Dataset, Factor, Var
import trftools

from settings import GRID_DIR, LEXICON_PATH


def iter_grids():
    for grid_path in GRID_DIR.glob('*.TextGrid'):
        stim = grid_path.stem.lower()
        grid = trftools.align.TextGrid.from_file(grid_path, word_tier='word', phone_tier='phone')
        yield stim, grid


def iter_realizations():
    for stim, grid in iter_grids():
        rs = [r for r in grid.realizations if r.graphs.strip()]
        assert len(rs) == 1
        r = rs.pop()
        yield stim, r.strip_stress()


def iter_words():
    lexicon = load.unpickle(LEXICON_PATH)
    for stim, r in iter_realizations():
        item = Dataset({
            'phones': Factor(r.phones),
            'time': Var(r.times),
            'surprisal': Var(lexicon.surprisal(r.pronunciation)),
            'entropy': Var(lexicon.entropy(r.pronunciation)),
        }, name=stim, info={'r': r})
        item.index()
        yield item
