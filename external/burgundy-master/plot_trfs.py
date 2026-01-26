from eelbrain import *
from jobs import e, STG, LTL


# ress = e.load_trf_test('gte1+word_onset', **STG, make=True)
ress = e.load_trfs(-1, 'gte8+cohort01-red', **LTL)

# print(ress.table())
# for key, res in ress.items():
#     plot.brain.butterfly(res.difference, vmax=vmax, name=key, h=2)

vmax = 0.01
y = ress['gammatone_edge_8'].mean('case').sum('frequency')
plot.brain.butterfly(y, vmax=vmax, name='Acoustic edges', h=1.5)

vmax = 0.005
keys = [
    'word_onset',
    'phone_surprisal_1',
    'phone_entropy_1',
]

for key in keys:
    y = ress[key].mean('case')
    plot.brain.butterfly(y, vmax=vmax, name=key, h=1.5)
