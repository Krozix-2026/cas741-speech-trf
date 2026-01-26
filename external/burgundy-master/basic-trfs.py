# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline
from itertools import chain

import cssl.roi
from eelbrain import *
from trftools.notebooks import Layout
from trftools.pipeline import ResultCollection

from burgundy import e
from basic_jobs import WHOLEBRAIN

brain_view = (-18, -28, 50)
# -

# # Phoneme model

LABELS = {
    'gammatone-8': 'Spectrogram',
    'gammatone-edge-8': 'Onset-Spectrogram',
    'phone-p0': 'Word onset',
    'phone-p12345': 'Phone',
    'phone-surprisal': 'Surprisal',
    'phone-entropy': 'Cohort entropy',
    'phone-phoneme_entropy': 'Phone entropy',
}
kwargs = dict(**WHOLEBRAIN, make=True, vmax=.002, labels=LABELS)

# ## Surprisal only

e.show_trf_test('gt8 + phone-0v12345 + phone-surprisal', terms='phone-*', **kwargs, 
#                 surf='pial',
                brain_view=(None, None, 60),
               )

# ## Surprisal & Entropy

MODEL = 'gt8 + phone-0v12345 + phone-surprisal + phone-entropy'
e.show_trf_test(MODEL, terms='phone-*', **kwargs, surf='pial')

# # Visualize

# ## ROI

# +
import cssl.roi
MODEL_FULL = "gt8 + phone-0v12345 + phone-surprisal + phone-entropy"
TEST = {
    'smooth': 0.005,
    'metric': 'det',
}

res_surprisal = e.load_model_test(f'{MODEL_FULL} | phone-surprisal', **WHOLEBRAIN, **TEST)
mask = set_parc(res_surprisal.p <= 0.05, 'aparc')
plot.brain.brain(mask)
# -

# store the vertices for which we want the end result
fsa_vertices = mask.source.vertices
# morphing is easier with a complete source space
mask = complete_source_space(mask)
# morph both hemispheres to the left hemisphere
mask_from_lh, mask_from_rh = xhemi(mask)
# take the union; morphing interpolates, so re-cast values to booleans
mask_lh = (mask_from_lh > 0) | (mask_from_rh > 0)
# morph the new ROI to the right hemisphere
mask_rh = morph_source_space(mask_lh, vertices_to=[[], mask_lh.source.vertices[0]], xhemi=True)
# cast back to boolean
mask_rh = mask_rh > 0
# combine the two hemispheres
mask_sym = concatenate([mask_lh, mask_rh], 'source')
# morph the result back to the source brain (fsaverage)
mask = morph_source_space(mask_sym, 'fsaverage', fsa_vertices)
# convert to boolean mask (morphing involves interpolation, so the output is in floats)
mask = round(mask).astype(bool)
plot.brain.brain(mask)

# ## Load TRFs

MODEL = "gt8 + phone-0v12345 + phone-surprisal"
SCALE = 'original'
SCALE = None
trfs = e.load_trfs(-1, MODEL, scale=SCALE, **WHOLEBRAIN)

# ## PCA

masked_trf = trfs['phone_surprisal'].sub(source=mask)
roi_ds = cssl.roi.pca_roi_timecourse(masked_trf, trfs, hemi='both', tstart=0, tstop=1)
f = roi_ds['component'].max('source')
roi_ds['component'] /= f
roi_ds['source'] *= f

args = dict(ds=roi_ds, tstart=0, tstop=1, pmin=0.05)
ress_pca = {hemi: testnd.TTestOneSample('source', sub=f"hemi == {hemi!r}", **args) for hemi in ['lh', 'rh']}
ress_pca['hemi'] = testnd.TTestRelated('source', 'hemi', match='subject', **args)
fmtxt.FloatingLayout([res.clusters.sub("p<=0.05").sorted('tstart').as_table(title=hemi) for hemi, res in ress_pca.items()])

p = plot.UTSStat('source', 'hemi', match='subject', ds=roi_ds)

# ## RMS

dss = []
for hemi in ['lh', 'rh']: 
    ds = trfs['subject',]
    ds['y'] = trfs['phone_surprisal'].rms(source=mask.sub(source=hemi))
    ds[:, 'hemi'] = hemi
    dss.append(ds)
ds = combine(dss)

res = testnd.TTestRelated('y', 'hemi', match='subject', ds=ds, tstart=0, tstop=0.900)
res

p = plot.UTSStat('y', 'hemi', match='subject', ds=ds)

# ## Anatomical

ndvar_binned = trfs['phone_surprisal'].mean('case').sub(source=mask).bin(0.020, 0.050, 0.150, 'mean')
sp = plot.brain.SequencePlotter()
sp.set_brain_args(surf='inflated')
sp.add_ndvar(ndvar_binned)
p = sp.plot_table(view='lateral', orientation='vertical')

# +
args = dict(axw=2.5, axh=1.5, xlim=(-0.050, 0.950), frame='t', clip=True, margins=True)  # , xlim=(-0.050, 0.450)

xs_gt = ['gammatone_8', 'gammatone_edge_8']
xs = ['phone_p0', 'phone_p12345', 'phone_surprisal_p12345']
ys = [
    *(trfs_1000[x].mean('frequency') for x in xs_gt), 
    *(trfs_1000[x] for x in xs)]
display(Layout([
    plot.Butterfly(y, '.source.hemi', ncol=1, title=y.name, linewidth=0.5, **args)#, vmax=.004)
    for y in ys
]))
# -

# ## Phoneme-surprisal
#
# TODO:
#  - susprisal and word onset
#  - Butterfly + main peaks

# +
from pathlib import Path
import matplotlib as mpl


DST = Path('~/iCloud/Research/Burgundy/plots-dissertation').expanduser()
mpl.style.use('default')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
# Change all line-widths
# for key in mpl.rcParams:
#     if 'linewidth' in key:
#         mpl.rcParams[key] *= 0.5
# The size of saved figures depends on the figure size (w, h) and DPI
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300

# +
x = 'gt8 + phone-0v12345 +| phone-surprisal-p12345'
res = e.load_model_test(x, epoch='cont', **LTL_5P, smooth=0.005)

cmap = plot.soft_threshold_colormap('lux-a', 0.001, 0.003)
poss = {
    'smoothwm': (-18, -10, 35), 
    'inflated': (-18, -28, 50),
}

l = Layout()
sp = plot.brain.SequencePlotter()
sp.add_ndvar(res.masked_difference(), cmap=cmap)
for surf, pos in poss.items():
    sp.set_brain_args(surf=surf)
    sp.set_parallel_view(*pos)
    p = sp.plot_table(view='lateral', orientation='vertical', w=2)
    p.save(DST / f'z surprisal {surf}.pdf', transparent=True)
    l.add(p)

p_cb = plot.ColorBar(cmap, w=1.1, h=1.1, width=0.1, clipmin=0, ticks=[0, 0.003], label='∆z')
p_cb.save(DST / f'z colorbar.pdf', transparent=True)
l.add(p_cb)

display(l)

# +
args = dict(w=3, axh=1., xlim=(-0.050, 0.600), 
            frame='t', clip=True, margins=True)
colors = 'k'#, {'lh': 'b', 'rh': 'r'}


# ts = [0.080, 0.120, 0.300, 0.350]
ts = [0.070, 0.290]
y = trfs_1000['phone_surprisal_p12345'].mean('case')
vmax = 0.003
p = plot.Butterfly(y, '.source.hemi', vmax=vmax, ncol=1, linewidth=0.5, **args, axtitle=False)
for ax in p.figure.axes:
    ax.grid(axis='x')
    ax.set_yticks([-vmax, 0, vmax])
p.figure.axes[1].set_yticklabels(['', '', ''])
p.figure.axes[0].set_yticklabels([-1, 0, 1])
for t in ts:
    p.add_vline(t, linestyle='--', color=(1, .9, 0))
p.save(DST / 'TRF surprisal.pdf')

# +
l = Layout()
cmap = plot.soft_threshold_colormap('xpolar-a', vmax / 10, vmax)
sp = plot.brain.SequencePlotter()
sp.set_parallel_view(-18, -28, 50)
for t in ts:
    sp.add_ndvar(y.sub(time=t), cmap=cmap, label=f'{int(t*1000)} ms')
for surf, pos in poss.items():
    sp.set_brain_args(surf=surf)
    sp.set_parallel_view(*pos)
    p = sp.plot_table(view='lateral', orientation='vertical', w=2)
    p.save(DST / f'TRF surprisal {surf}.pdf', transparent=True)
    l.add(p)

p_cb = plot.ColorBar(cmap, w=1.1, h=1.1, width=0.1, ticks={-vmax: -1, 0: 0, vmax:1}, label='MNE')
p_cb.save(DST / f'TRF colorbar.pdf', transparent=True)
l.add(p_cb)
    
display(l)
