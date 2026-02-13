# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Model syntax
# Convert model names from |-based predictor syntax to ~-based syntax. Predictor-files are the only affected file paths.

# +
from pathlib import Path

from eelbrain import *


ROOT = Path('/Volumes/Seagate BarracudaFastSSD/Appleseed')

# +
old_name_file = ROOT / 'eelbrain-cache' / 'model-names.pickle.backup'
new_name_file = ROOT / 'eelbrain-cache' / 'model-names.pickle'

data = load.unpickle(old_name_file)
# -

out = []
for key, model in data:
    if '|' in model or not key.startswith('model'):
        print(key, end=', ')
        continue
    out.append((key, model.replace('p12345', 'p1_')))

save.pickle(out, new_name_file)
