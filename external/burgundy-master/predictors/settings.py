from pathlib import Path

ROOT = Path(__file__).parents[1] / 'BURGUNDY'
WAV_DIR = ROOT / 'Stimuli'
GAMMATONE_DIR = ROOT / 'Gammatone'
GRID_DIR = ROOT / 'TextGrids'
LEXICON_PATH = ROOT / 'lexicon.pickle'

DATA_ROOT = Path('~/Data/burgundy').expanduser()
PREDICTORS_DIR = DATA_ROOT / 'predictors'
