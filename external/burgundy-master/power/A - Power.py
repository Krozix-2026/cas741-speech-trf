# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from eelbrain import *
from matplotlib import pyplot

from burgundy import e

from jobs import STG

# +
NS = list(range(200, 1100, 100))

dss = []
for n in NS:
    epoch = 'cont' if n == 1000 else f'cont-{n}'
    ds = e.load_trfs(-1, f"gt-log8 + phone-0v12345 +@ phone-surprisal", **STG, epoch=epoch)
    ds[:, 'n_words'] = n
    dss.append(ds)
data = combine(dss)
# -

for hemi in ['lh', 'rh']:
    data[f'det_{hemi}'] = data['det'].mean(source=hemi)

figure, axes = pyplot.subplots(1, 2, figsize=(7, 3))
for hemi, ax in zip(['lh', 'rh'], axes):
    p = plot.Barplot(f'det_{hemi} * 100', 'n_words', data=data, axes=ax)
    ax.set_title(hemi.upper())
pyplot.tight_layout()

rows = []
for hemi in ['lh', 'rh']:
    print(hemi)
    for n in NS:
        res = test.TTestOneSample(f'det_{hemi}', sub=f"n_words == {n}", data=data, tail=1)
        print(f'{n:>4}: d={res.d:.2f}')
        rows.append([hemi, n, res.d])
ds = Dataset.from_caselist(['hemi', 'n', 'd'], rows)

df = ds.as_dataframe

p = plot.Timeplot('d', 'n', 'hemi', data=ds, ylabel="Cohen's d", xlabel='Number of words', title="Surprisal effect size ~ number of words")
