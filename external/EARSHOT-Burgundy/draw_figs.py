from functools import partial

from eelbrain import *
from eelbrain._stats.stats import dispersion
from eelbrain.plot._colors import adjust_hls
import matplotlib
from matplotlib import pyplot
import numpy
import seaborn


from jobs import e, WHOLEBRAIN
from constants2 import model
from data import load_roi_data
from rc import LOSS, LOSS_COLORS
import rc

HIDDEN_FLAT = [
    '512', '320x320', '256x256x256', '192x192x192x192',
]

# norm: divided by baseline model in each hemisphere
rnn = partial(model, HIDDEN_FLAT[0], target_space='OneHot', loss=f'dw1024to10', k=32)

ds_full = load_roi_data(f"log-8 + phone + {rnn()}")
dss_reduced = {
    'onset': load_roi_data(f"log-8 + phone + {rnn(transform='sum')}"),
    'sum': load_roi_data(f"log-8 + phone + {rnn(transform='onset')}"),
    'rnn': load_roi_data(f"log-8 + phone + "),
    'auditory': load_roi_data(f"phone + {rnn()}"),
}
# Variance component in full 'det_roi' attributable to key:
for key, ds_reduced in dss_reduced.items():
    ds_full[key] = ds_full['det_roi'] - ds_reduced['det_roi']
# Drop hemisphere
hemi_ds = ds_full.aggregate("subject", drop='hemi')
# -

# ### RNN sum vs. onset

value_rnn = hemi_ds['rnn'].mean()  # Attributable to RNN
print(f"RNN: {value_rnn:.1%}, thereof:")
for key in ['onset', 'sum']:  # 
    value = hemi_ds[key].mean()
    prop = value / value_rnn
    print(f"{key}: {prop:.1%}")
    print(test.TTestOneSample(key, 'subject', data=hemi_ds))