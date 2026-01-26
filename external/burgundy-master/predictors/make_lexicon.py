# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Generate base lexicon

# +
from collections import defaultdict
from pathlib import Path

from eelbrain import save
import trftools
import speech.lexicon

from settings import GRID_DIR, LEXICON_PATH

# +
celex = speech.lexicon.Celex()

# Use CMUPD segmentations from Appleseed
MFA_DIR = Path("~/iCloud/Research/Appleseed/stimuli/mfa_dict").expanduser()
SEG_FILE = MFA_DIR / 'segmentations.txt'
appleseed_segmentations = speech.lexicon.read_segmentations(SEG_FILE)
segmentations = defaultdict(lambda: None, appleseed_segmentations)

# base 
cmupd = trftools.dictionaries.read_cmupd(strip_stress=True)
subtlex = trftools.dictionaries.read_subtlex()
frequencies = {w: entry['FREQcount'] for w, entry in subtlex.items()}
# -

# ## Pronunciations
# - CMUPD contais >70'000 words not in SUBTLEX, mainly
#   - Words with apostrophe
#   - Names

pronunciations = defaultdict(set, {w: p for w, p in cmupd.items() if w in subtlex})

missing = [w for w in cmupd if w not in subtlex]
print(f"{len(missing)} of {len(cmupd)} CMUPD words are not in SUBTLEX")
print(', '.join(missing))

# # Missing entries

segmentations['EXAMINATION']

SEGMENTATIONS = {
    'EXAM': 'EXAMINE',
}
for grid_path in GRID_DIR.glob('*.TextGrid'):
    grid = trftools.align.TextGrid.from_file(grid_path, word_tier='word', phone_tier='phone')
    # find pronunciation
    rs = [r for r in grid.realizations if r.graphs.strip()]
    assert len(rs) == 1
    r = rs.pop()
    phones = ' '.join([p.rstrip('012') for p in r.phones])
    # check databases
    word = grid_path.stem.upper()
    # pronunciation
    pronunciations[word].add(phones)
    # segmentation
    if word not in segmentations:
        if word in SEGMENTATIONS:
            segmentation = SEGMENTATIONS[word]
        else:
            segmentation = celex.parse_word(word, missing='ignore')
            if segmentation is None:
                segmentation = word
                print(f"MISSING segmentation: {word}")
        segmentations[word] = segmentation

lexicon = speech.lexicon.generate_lexicon(pronunciations, frequencies, segmentations)
save.pickle(lexicon, LEXICON_PATH)
