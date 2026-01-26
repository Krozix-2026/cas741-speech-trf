# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setup

# +
from itertools import chain
from pathlib import Path

import cssl.roi
import joblib
import numpy
from eelbrain import *
from eelbrain.plot._figure import AbsoluteLayoutFigure
from eelbrain._colorspaces import UNAMBIGUOUS_COLORS
from matplotlib import pyplot, font_manager
from matplotlib.colors import to_hex
from trftools.pipeline import ResultCollection
from trftools.align import TextGrid
from trftools.dictionaries._arpabet import IPA
import cssl

from basic_jobs import e, STG, TEMPORAL, WHOLEBRAIN


memory = joblib.Memory('.')

TEST = {
    'smooth': 0.005,
    'metric': 'det',
}

W_PAGE = 7
FONT = 'Arial'
FONT_SIZE = 8
LINEWIDTH = 1 #0.5
RC = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'font.style': 'normal',
    # make sure equations use same font
    'mathtext.fontset': 'custom',
    'font.cursive': FONT,
    'font.serif': FONT,
    # subplot
    'figure.subplot.top': 0.95,
    # legend
    'legend.frameon': True,
    # line width
    'axes.linewidth': LINEWIDTH,
#     'boxplot.boxprops.linewidth': 1.0,
#     'boxplot.capprops.linewidth': 1.0,
#     'boxplot.flierprops.linewidth': 1.0,
#     'boxplot.meanprops.linewidth': 1.0,
#     'boxplot.medianprops.linewidth': 1.0,
#     'boxplot.whiskerprops.linewidth': 1.0,
    'grid.linewidth': LINEWIDTH,
#     'hatch.linewidth': LINEWIDTH,
    'lines.linewidth': LINEWIDTH,
    'patch.linewidth': LINEWIDTH,
    'xtick.major.width': LINEWIDTH,
    'xtick.minor.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH,
    'ytick.minor.width': LINEWIDTH,
    # colors
#     'axes.prop_cycle': 
}
pyplot.rcParams.update(RC)

CORTEX = ('.8', '.6')

def clean_ax(ax):
    ## removing the spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ## removing the tick marks
    ax.tick_params(bottom="off", left="off")
    ## no y-ticks
    ax.set_yticklabels(())

DST = Path('~/iCloud/Research/Burgundy/Figures/plots-rev').expanduser()
DST.mkdir(exist_ok=True)
DATA_DIR = Path('.').resolve() / 'data'
WAV_DIR = DATA_DIR / 'Stimuli'
GRID_DIR = DATA_DIR / 'TextGrids'
PREDICTOR_DIR = Path('/Volumes/Seagate BarracudaFastSSD/Burgundy/predictors')

APPLESEED_PREDICTOR_DIR = Path('/Volumes/Seagate BarracudaFastSSD/Appleseed/predictors')

MODEL = "gt8 + phone-0v12345 + phone-surprisal"
MODEL_FULL = "gt8 + phone-0v12345 + phone-surprisal + phone-entropy"
LABELS = {
    'gammatone_8': 'Envelope spectrogram',
    'gammatone_edge_8': 'Onset spectrogram',
    'phone_p0': 'Word onset',
    'phone_p12345': 'Phoneme onset',
    'phone_surprisal': 'Phoneme surprisal',
    'phone_entropy': 'Cohort entropy',
}
# -

# # Methods
# - Data: Single source, single subject, 
# - Plot predictor * TRF = response 

', '.join(e.get_field_values('subject'))

# ## MEG data

# +
SUBJECT, Y_SCALE = 'R2626', 2e10
SUBJECT, Y_SCALE = 'R2633', 5e10
# SUBJECT, Y_SCALE = 'R2641', 1e10  # response to entropy
# SUBJECT, Y_SCALE = 'R2648', 1e10  # response to entropy
SUBJECT, Y_SCALE = 'R2649', 4e10  # pretty good
SUBJECT, Y_SCALE = 'R2659', 1e10  # pretty good
WORD = 'relic'
WORD = 'dialogue'
N_WORDS = 3
T_PAD = 0.050
ISI = 0.267

trfs = e.load_trf(MODEL_FULL, **WHOLEBRAIN, subject=SUBJECT)
# -

res_surprisal = e.load_model_test(f'{MODEL_FULL} @ phone-surprisal', **WHOLEBRAIN, **TEST)
y_pick_source = res_surprisal.difference
# y_pick_source = trfs.proportion_explained
y = set_parc(y_pick_source, 'aparc')  # bug in concatenate messed up parc
y = y.sub(source=(['superiortemporal-lh']))#, 'transversetemporal-lh']))
source_index = y.argmax()

events = e.load_events()
# display(events)
i_start = events['item'].index(WORD)[0]
data_tstart = events[i_start, 'T'] - T_PAD
data_tstop = events[i_start+N_WORDS, 'T'] - ISI + T_PAD
# stimulus properties
WORDS = list(events[i_start:i_start+N_WORDS, 'item'])
# time: start sample data at t=0
t0 = events[i_start, 'T'] - T_PAD
T0 = [round(t - t0, 2) for t in events[i_start:i_start+N_WORDS, 'T']]

stc = e.load_raw_stc('superiortemporal', True, tstart=data_tstart, tstop=data_tstop)
y_stc = stc.sub(source=source_index)

trfs.x

# ## Predictors

# +
# load and concatenate wav files
SRATE = 10000  # 44100 // 4
time = UTS.from_range(0, stc.time.tstop, 1 / SRATE)
wav = 0
for t0, word in zip(T0, WORDS):
    wav_i = load.wav(WAV_DIR / f"{word.upper()}.wav")
    wav_i = resample(wav_i.astype(float), SRATE)
    wav_i = set_tmin(wav_i, t0)
    wav += set_time(wav_i, time)

# TextGrids
grids = []
for word in WORDS:
    grid = TextGrid.from_file(GRID_DIR / f"{word.upper()}.TextGrid", phone_tier='phone', word_tier='word')
    grid = grid.strip_stress()
    grids.append(grid)


def load_predictor(code):
    "Load predictor as NDVar"
    time = UTS.from_range(wav.time.tmin, wav.time.tstop, 0.01)
    x = 0  # to be broadcast to time
    for t0, word in zip(T0, WORDS):
        t0 = round(t0, 2)
        xi = e.load_predictor(f'{word}~{code}', tmin=0, tstep=0.01)
        xi = set_tmin(xi, t0)
        x += set_time(xi, time)
    return x

def to_stem(ax, y, color='k'):
    "Make stem-plot"
    color = UNAMBIGUOUS_COLORS.get(color, color)
    nonzero = y.x != 0
    nonzero[0] = True
    nonzero[-1] = True
    ax.set_yticks(())
    ax.set_xticks(())
    h = ax.stem(y.time.times[nonzero], y.x[nonzero], bottom=0, 
                linefmt=to_hex(color), markerfmt=' ', 
                basefmt=f'#808080', 
#                 basefmt=f'k-', 
                use_line_collection=True)


def plot_trf(trf, ax, color, clip=False, **kwargs):
    color = UNAMBIGUOUS_COLORS.get(color, color)
#     print(trf.max())
#     print(y.x)
#     plot.UTS(y, colors=[color], axes=[ax], ylabel='*', xlabel=False)#, frame='none')
#     if trf.ndim == 1:
    ax.plot(trf.time.times, trf.x.T, color=color, clip_on=clip, **kwargs)
    ax.set_yticks(())
    ax.set_xticks(())
    ax.set_xlim(trf.time.tmin, trf.time.tstop)
    ax.set_ylim(1, -1)
#     ax.set_ylabel(ylabel, size=20, rotation='horizontal', ha='right', va='center', labelpad=10)



# -

# ## Plot

# +
from eelbrain._colorspaces import lch_to_rgb, rgb_to_lch, UNAMBIGUOUS_COLORS

PREDICTOR_COLORS = {
    'gammatone_8': 'k',
    'gammatone_edge_8': 'k',
    'phone_p0': 'black',
    'phone_p12345': 'reddish purple',
    'phone_surprisal': 'sky blue',
    'phone_entropy': 'orange',
}

# Figure
h = 6.5
t_per_inch = 1.17
left = 0.5
x_sep = 0.5  # space between plots
width = wav.time.tstop / t_per_inch
left_trf = left + width + x_sep
width_trf = 1.1 / t_per_inch
left_response = left_trf + width_trf + x_sep
ylim = 1
ylims = (-ylim, ylim)
xlim = None  # (-x_pad, 2)
args = dict(xlim=xlim, xlabel=False, xticklabels=False, yticklabels=False)
y0 = h - 0.5

fig = AbsoluteLayoutFigure(2, h=h, w=W_PAGE)
predicted_responses = []

# Stimuli
wav /= wav.max() * 1.2
for x, ylabel in [[left_response, False], [left, 'Stimuli\n']]:
    ax = fig.add_axes(y0, ylims, x, width, frameon=False)
    # ax.tick_params('both', length=0, grid_color=(.8, .8, .8))
    # ax.grid(True, 'major', 'both', color='0.8')
    plot.UTS(wav, frame='none', colors='k', ylabel=ylabel, legend=False, axes=[ax], **args)
    # ax.text(0.400, ylim/2, LABELS[x], {'size': 12})
    ax.set_yticks(())
    ax.set_xticks(())
# phonemes
last_t = -1
for t0, grid in zip(T0, grids):
    for r in grid.realizations:
        for p, t in zip(r.phones, r.times):
            last_t = max(t0+t, last_t + 0.070)
            h = ax.text(last_t, -1, IPA[p], ha='center')
# Titles
fig.text(left_trf, y0-0.8, "TRFs")#, size=16)
fig.text(left_response, y0-0.8, "Predicted response components")#, size=16)
y0 -= 1.2

def plot_row(y0, name, color, label, norm=True, sharex=None, cmap='Blues', y_pred_norm=1e-20, xticks=False):
    # entropy
    x = load_predictor(name)
    if norm is True:
        norm = x.max()
    if x.ndim == 1:
        ax = fig.add_axes(y0, ylims, left, width, frameon=False, sharex=sharex)
        to_stem(ax, x / norm, color)
        kw = {}
    else:
        ax = fig.add_axes_rect(left, y0-0.3, width, 0.6, frameon=False, sharex=sharex)
        plot.Array(x, cmap=cmap, axes=[ax], yticklabels=False, xlabel=False, interpolation='none', xlim=xlim)
        if not xticks:
            ax.set_xticks(())
        ax.set_yticks(())
        kw = {'y': 0}
    if label:
        ax.set_ylabel(label, ha='left', **kw)
    # *
    fig.text(left_trf - x_sep / 2, y0, r'$\ast$', size=20, ha='center', va='center')
    # TRF
    key = Dataset.as_key(name)
    i = trfs.x.index(key)
    trf = trfs.h[i].sub(source=source_index)
    ax_trf = fig.add_axes(y0, ylims, left_trf, width_trf, frameon=False)
    plot_trf(trf*100, ax_trf, color)
    # =
    fig.text(left_response - x_sep / 2, y0, '=', size=20, ha='center', va='center')
    # response
    trf = trfs.h_scaled[i].sub(source=source_index)
    y_pred = convolve(trf, x)
    predicted_responses.append(y_pred)
    ax_pred = fig.add_axes(y0, ylims, left_response, width, frameon=False)
    plot_trf(y_pred*Y_SCALE, ax_pred, color)#, y_pred_norm)
    return ax, ax_trf, ax_pred


ax, _, _ = plot_row(y0, 'phone-entropy', 'orange', 'Cohort\nentropy')
y0 -= 0.6

# surprisal
plot_row(y0, 'phone-surprisal', 'sky blue', 'Phoneme\nsurprisal', sharex=ax)
y0 -= 0.4

# word/phone
plot_row(y0, 'phone-p0', 'black', 'Word\n', 2, sharex=ax)
y0 -= 0.4

plot_row(y0, 'phone-p12345', 'black', 'Phone\n', 2, sharex=ax)
y0 -= 0.5

# Spectrograms
spectrogram_color = (60/256, 120/256, 180/256)
spectrogram_color = (230/256, 120/256, 90/256)
spectrogram_color = 'k'
plot_row(y0, 'gammatone-8', spectrogram_color, 'Envelope\nspectrogram', 2, sharex=ax, cmap='binary')#, y_pred_norm=300)
y0 -= 0.7

# Onset spectrogram
ax, ax_trf, ax_pred = plot_row(y0, 'gammatone-edge-8', spectrogram_color, 'Onset\nspectrogram', 2, cmap='binary', y_pred_norm=1, xticks=True)
ax.set_xlabel('Experiment time [s]')
# x-axis labels
ax_trf.set_xlabel(r'$\tau$ [s]')
ax_trf.set_xticks([0, 0.500, 1.0])
ax_trf.tick_params(bottom=False, labelbottom=True)#, top=False)
for h in ax_trf.get_xticklabels():
    h.set_y(.25)
# ax.get_xticklabels()[-1].set_visible(False)
y0 -= 1.3

# Dipole STC
ax = fig.add_axes(y0, (-2.5*ylim, 2.5*ylim), left_response, width, frameon=False)
y_stc_plot = y_stc * Y_SCALE
y_stc_plot = filter_data(y_stc_plot, 0, 8)
plot_trf(y_stc_plot, ax, 'vermilion', linestyle='--')#, clip=True)
# predicted response
y_pred = sum(predicted_responses)
color = lch_to_rgb(50, 100, 27/360)
plot_trf(y_pred*Y_SCALE, ax, 'k')
ax.set_xlabel('Experiment time [s]')
ax.set_xticks(numpy.arange(0, y_pred.time.tstop, 1))
ax.tick_params(bottom=True, labelbottom=True, top=False)#, top=False)
# for h in ax.get_xticklabels():
#     h.set_y(.3)

fig.finalize()#True)
fig.save(DST / 'Methods TRFs.pdf')
fig.draw_outline()
# -

# ## Brain plot

ss = trfs.proportion_explained.source
brain = plot.brain.brain(ss, hemi='lh', surf='pial', views='lateral', w=600, h=500, cortex=CORTEX)
i = ss._array_index(source_index)
origin = ss.coordinates[i] * 1000 + [-15, 0, 0]#[-100, -20, -20]
axis = ss.normals[i]
from mayavi import mlab  # import after plotting brain so backend can be set
# mlab.points3d(*origin, color=(1,0,0), scale_factor=10, reset_zoom=False)
color = UNAMBIGUOUS_COLORS['vermilion']
mlab.quiver3d(*origin, *axis, color=color, figure=brain.brain_matrix[-1,-1]._f, reset_zoom=False, 
#               mode='2dthick_arrow', scale_factor=30, line_width=3,
              mode='arrow', scale_factor=30, scale_mode='vector',
#               mode='2darrow', scale_factor=15, line_width=3,
             )  # mode='arrow' for 3d arrow
brain.save_image(DST / 'Methods brain.png')
display(brain)
brain.close()

break

# # Predictors

dss = []
for path in PREDICTOR_DIR.glob('*~phone.pickle'):
    ds = load.unpickle(path)
    ds = ds.sub('any')
    ds[:, 'word'] = path.stem.split('~')[0]
    dss.append(ds)
value_ds = combine(dss)

apple_dss = []
for segment in range(1, 12):
    ds = load.unpickle(APPLESEED_PREDICTOR_DIR / f'{segment}~phone.pickle')
    ds = ds.sub('any')
    apple_dss.append(ds)
appleseed_value_ds = combine(apple_dss)

corr = test.Correlation('surprisal', 'entropy', ds=value_ds)
appleseed_corr = test.Correlation('surprisal', 'entropy', ds=appleseed_value_ds)
print(f"Burgundy: {corr}\nAppleseed: {appleseed_corr}")

p = plot.Scatter('entropy', 'surprisal', 'pos', ds=value_ds, colors='viridis', frame=None, alpha=0.2)

n_syllables = load.tsv(DATA_DIR / 'word n-syllables.txt')

n_syllables = dict(n_syllables.zip('word', 'n_syllables'))

time = UTS(0, 1/100, 100)
rows = []
for ds in dss:
    word = ds[0, 'word']
    row = [word, n_syllables[word.upper()]]
    for y in ['entropy', 'surprisal']:
        x = NDVar.zeros(time)
        for t, v in ds.zip('time', y):
            x[t] = v
        row.append(x)
    rows.append(row)
ndvar_ds = Dataset.from_caselist(['word', 'n_syllables', 'entropy', 'surprisal'], rows)

SMOOTH = 0.050
y_e = ndvar_ds['entropy'].smooth('time', SMOOTH)
y_s = ndvar_ds['surprisal'].smooth('time', SMOOTH)
p = plot.UTSStat([y_e, y_s], 'n_syllables', ds=ndvar_ds, axh=2, axw=3, frame='t')

# # Model tests

# +
GROUP = 'good2'
# GROUP = 'good2-data'

tests = {
    'surprisal': f"{MODEL_FULL} @ phone-surprisal",
    'entropy':   f"{MODEL_FULL} @ phone-entropy",
}
ress = {key: e.load_model_test(x, **WHOLEBRAIN, **TEST, group=GROUP) for key, x in tests.items()}

# +
from appleseed_jobs import FTP, FTP_5, STL, STL_5, e as appleseed


APPLESEED_EPOCH, kwargs = 'apple', STL
# APPLESEED_EPOCH, kwargs = 'seg1-2', FTP_5
MODEL_FULL_APPLESEED = "gte8 + phone-p0 + phone-p1_ + phone-surprisal + phone-entropy"
tests_apple = {
    'surprisal-appleseed': f"{MODEL_FULL_APPLESEED} @ phone-surprisal",
    'entropy-appleseed':   f"{MODEL_FULL_APPLESEED} @ phone-entropy",    
}
ress.update({key: appleseed.load_model_test(x, **kwargs, **TEST, epoch=APPLESEED_EPOCH) for key, x in tests_apple.items()})
# -

ResultCollection(ress)

# +
# @memory.cache
# def rel_det_test(x, experiment='b'):
#     "noisier (probably because of division)"
#     if experiment == 'a':
#         exp = appleseed
#         kwargs = FTP
#     elif experiment == 'b':
#         exp = e
#         kwargs = WHOLEBRAIN
#     else:
#         raise ValueError(experiment)
#     comp = exp._coerce_comparison(x, True)
#     ds = exp.load_trfs(-1, comp.x1, **kwargs)
#     ds0 = exp.load_trfs(-1, comp.x0, **kwargs)
#     assert numpy.all(ds['subject'] == ds0['subject'])

#     y = (ds['det'] - ds0['det']) / ds['det']
#     y = set_parc(y, 'aparc')
#     y = y.smooth('source', 0.005, 'gaussian')
#     return testnd.TTestOneSample(y, tfce=True, tail=1)


# ress = {key: rel_det_test(x) for key, x in tests.items()}
# ress.update({key: rel_det_test(x, 'a') for key, x in tests_apple.items()})

# +
# concatenating soure space messed up parc
source = set_parc(ress['entropy'].t.source, 'aparc')
# roi: posterior 2/3 of STG
rois = cssl.roi.mask_roi('STG301', source)
roi = rois[0] | rois[1]

roi_label = cssl.roi.mne_label('STG301', subjects_dir=e.get('mri-sdir'))

brain_vmax = ress['surprisal'].c1_mean.max()  #max(..., ress['surprisal-appleseed'].c1_mean.max())
percent = brain_vmax / 100
# -

# ## Full model predictive power

ress['surprisal'].c1_mean.max()

trfs = e.load_trfs(-1, MODEL_FULL, **WHOLEBRAIN, trfs=False)

trfs['r'].mean('case').max('source')

# ## Brain

# for determining cmap limits
for key, res in ress.items():
#     print(res.t.max(), res.masked_parameter_map().min())
    print(res.c1_mean.max(), res.difference.max(), res.masked_difference().min())

# +
# brain_cmap = plot.soft_threshold_colormap('lux-a', 2.5, 8)  # for t
# p = plot.ColorBar(brain_cmap, clipmin=0, w=1, h=1, width=0.05, ticks=[0, 2, 4, 6], label='$t$', background='black')
# p.save(DST / 'F3-brain cmap.pdf')

# brain_cmap = plot.soft_threshold_colormap('lux-a', 1.5e-5, 2e-4)  # for det
brain_cmap = plot.soft_threshold_colormap('lux-a', percent/10, percent)  # for det
ticks = {0: 0, percent/2: 0.5, percent: 1}
p = plot.ColorBar(brain_cmap, clipmin=0, w=2, h=1, width=0.05, ticks=ticks, label='∆ explained variability (%)', background=CORTEX[0])
# p.save(DST / 'Model brain cmap det.pdf')

# +


for key, res in ress.items():
    sp = plot.brain.SequencePlotter()
    sp.set_brain_args(surf='pial', cortex=CORTEX)
#     sp.set_brain_args(surf='smoothwm', cortex=('.8', '.6'))
    sp.set_parallel_view(scale=65)
    # sp.set_parallel_view(*brain_view)
#     y = res.t
#     y = res.masked_parameter_map(0.05)
#     y = res.difference
    y = res.masked_difference()

    # cmap based on maximum model fit (per experiment)
#     vmax = res.c1_mean.max()
#     brain_cmap = plot.soft_threshold_colormap('lux-a', vmax/1000, vmax/100)
    
    sp.add_ndvar(y, cmap=brain_cmap, alpha=0.8, smoothing_steps=15)#, lighting=True)
    # ROI outline
    load.update_subjects_dir(roi, y.source.subjects_dir)   
    sp.add_label(roi_label, color=(1., 1., 1.), borders=3, overlay=True, lighting=False)
    # table
    p = sp.plot_table(view='lateral', orientation='vertical', axw=1.2, axh=.8, dpi=300)
    p.save(DST / f'Model brain-det {key}.pdf')
#     break
# -

# ## ROI tests

# +
@memory.cache
def roi_det(x, roi='STG', experiment='b', norm='brain_max', group='okay-data'):
    if experiment.startswith('a'):
        exp = appleseed
        if experiment == 'a':
            kwargs = {**STL, 'epoch': 'apple'}
        elif experiment == 'a1':
            kwargs = {**STL_5, 'epoch': 'seg1-2'}
        elif experiment == 'a5':
            kwargs = {**STL_5, 'epoch': 'seg5-6'}
        elif experiment == 'a7':
            kwargs = {**STL_5, 'epoch': 'seg7-8'}
        elif experiment == 'a57':
            ds = roi_det(x, roi, 'a5', norm, group)
            ds7 = roi_det(x, roi, 'a7', norm, group)
            ds['det'] += ds7['det']
            ds['det'] /= 2
            ds[:, 'experiment'] = experiment
            return ds
    elif experiment == 'b':
        exp = e
        kwargs = {**STG, 'group': group}
    else:
        raise ValueError(experiment)
    comp = exp._coerce_comparison(x, True)
    ds = exp.load_trfs(-1, comp.x1, **kwargs)
    ds0 = exp.load_trfs(-1, comp.x0, **kwargs)
    assert numpy.all(ds['subject'] == ds0['subject'])

#     d_det = ds['det'] - ds0['det']
#     d_det /= ds['det']  # as proportion of the full model
#     ds['det'] = d_det
    source = set_parc(ds['det'].source, 'aparc')
    rois = cssl.roi.mask_roi(roi, source)
    dss = []
    for hemi, roi in zip(['lh', 'rh'], rois):
        hemi_ds = ds['subject',]
        det_1 = ds['det'].mean(source=roi)
        det_0 = ds0['det'].mean(source=roi)
        if norm == 'brain_max':
            norm_v = brain_vmax
        elif norm == 'experiment':
            norm_v = det_1
        else:
            raise ValueError(norm)
        hemi_ds['det'] = (det_1 - det_0) / norm_v
        hemi_ds[:, 'hemi'] = hemi
        hemi_ds[:, 'experiment'] = experiment
        dss.append(hemi_ds)
    return combine(dss)


# +
NORM = 'brain_max'
# NORM = 'experiment'
ROI = 'STG301'
APPLESEED_EPOCH = 'a'  # a, a5

ds_s = combine([
    roi_det(tests['surprisal'], ROI, norm=NORM, group=GROUP),
    roi_det(tests_apple['surprisal-appleseed'], ROI, APPLESEED_EPOCH, norm=NORM),
])
ds_s[:, 'measure'] = 'surprisal'
ds_e = combine([
    roi_det(tests['entropy'], ROI, norm=NORM, group=GROUP),
    roi_det(tests_apple['entropy-appleseed'], ROI, APPLESEED_EPOCH, norm=NORM),
])
ds_e[:, 'measure'] = 'entropy'
# subjects in both expeiments
freqs = table.frequencies('subject', ds=ds_e)
shared_subjects = set(freqs[freqs['n'] > 2, 'subject'])
print(f"Shared subjects: {shared_subjects}")
for ds in [ds_e, ds_s]:
    ds['shared'] = Factor(ds['subject'].isin(shared_subjects), labels={True: 'shared', False: 'unshared'})
    for subject in shared_subjects:
        index = ds.eval(f"(subject == {subject!r}) & (experiment == '{APPLESEED_EPOCH}')")
        ds[index, 'subject'] = f'{subject}b'
all_ds = combine([ds_e, ds_s])
# -

# ### Lateralization & Effect size

# +
# # fix outlier
# strange_index = abs(all_ds['det']) > 2 * all_ds['det'].std()
# print(all_ds[strange_index])
# other_values = all_ds[~strange_index, 'det']
# new_max = 2 * other_values.std()
# # new_max = other_values.max()
# print(f'--> {new_max:.4f} (new max {other_values.max():.4f})')
# all_ds[strange_index, 'det'] = new_max
# -

t = fmtxt.Table('lllll')
t.cells('experiment', 'measure', 'lh = rh', 'lh + rh', 'd')
interaction = all_ds.eval('experiment % measure')
for cell in interaction.cells:
    t.cells(*cell)
    dsi = all_ds[interaction == cell]
    t.cell(test.TTestRelated('det', 'hemi', 'lh', 'rh', 'subject', ds=dsi))
    dsi = dsi.aggregate('subject', drop='hemi')
#     print(dsi)
#     print(dsi['det'].mean(), dsi['det'].std(), scipy.stats.ttest_1samp(dsi['det'].x, 0))
    t.cell(test.TTestOneSample('det', 'subject', ds=dsi, tail=1))
    d = dsi['det'].mean() / dsi['det'].std()
    t.cell(fmtxt.stat(d))
p = plot.Boxplot('det', 'experiment % measure', match='subject', ds=all_ds, test=False, label_fliers=True, h=3, w=4)  # , top=.01
p.add_hline(0)
fmtxt.FloatingLayout([t, p])

# ### Plot

# +
import seaborn
from matplotlib.ticker import FuncFormatter

# y_res, y_lim = 2.5e-4, (-2e-5, 1.5e-4)  # as % total
# y_res, y_lim = .04, (-.005, .02)  # as % total
y_res, y_lim = .015, (-.002, .01)  # for brain_vmax

fig = AbsoluteLayoutFigure(y_res, h=3, w=2.)
formatter = FuncFormatter(lambda x, pos: '%g' % (100 * x))

y0s = [1.7, 0.7]
dss = [ds_e, ds_s]
colors = [UNAMBIGUOUS_COLORS[c] for c in ['orange', 'sky blue']]
axes = []
for y0, ds, color, is_last in zip(y0s, dss, colors, [False, True]):
    for experiment, left, is_right in zip(['b', APPLESEED_EPOCH], [0.2, 1.4], [False, True]):
        ax = fig.add_axes(y0, y_lim, left, 0.4)
        # format axes
        ax.spines['right'].set_visible(False)
        ax.spines['right'].set_position(('outward', 1))
        ax.yaxis.set_label_position('right')
#         ax.set_yticks([0.5e-4, 1.5e-4], minor=True)
        ax.tick_params(left=False, right=False, top=False, labelright=True, labelleft=False, labelbottom=is_last)
        ax.tick_params(axis='y', pad=4)
        ax.grid(True, 'major', 'y', color='0.8')
#         ax.grid(True, 'minor', 'y', color='0.9')
        for tick in ax.yaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.label.set_visible(False)
        if not is_right:
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.set_yticklabels(())
        # prepare data
        sds = ds.sub(ds["experiment"] == experiment)
        sds['is_rh'] = sds['hemi'] == 'rh'
        df = sds.as_dataframe()
        y = sds['det'].x
        palette = {'shared': (0, 0, 0), 'unshared': color}
        # finalize
        h = seaborn.swarmplot(x='is_rh', y='det', hue='shared', data=df, ax=ax, size=2, clip_on=False, palette=palette)
        # remove crap
        ax.get_legend().remove()
        ax.set_ylabel('')
        ax.set_xlabel('')
        # append
        axes.append(ax)
        if is_last:
            ax.set_xticklabels(['L', 'R'])
            if not is_right: 
                ax.set_ylabel("∆ explained variability (%)", y=-.005, ha='left', labelpad=8)

fig.finalize()
fig.save(DST / f'Models by-subject brain_vmax {APPLESEED_EPOCH} {GROUP}.pdf')
# -

# ### ANOVA

content = []
res = test.ANOVA('det', 'experiment * measure * hemi * subject(experiment)', ds=all_ds, title='Both')
content.append(res.table())
for ds in [ds_e, ds_s]:
    res = test.ANOVA('det', 'experiment * hemi * subject(experiment)', ds=ds, title=ds[0, 'measure'].capitalize())
    content.append(res.table())
p = plot.Barplot('det', 'experiment % measure', match='subject', ds=all_ds, w=3, corr=False)
content.append(p)
for hemi in ['lh', 'rh']:
    res = test.ANOVA('det', 'experiment * measure * subject(experiment)', sub=f"hemi == '{hemi}'", ds=all_ds, title=hemi.upper())
    content.append(res.table())
fmtxt.FloatingLayout(content)

# ### Log-scaled
# Variability explained with SNR differences is multiplicative (twice the SNR)

(all_ds['det'] * 100 + 1).min()

# +
all_ds['log_det'] = (all_ds['det'] * 100 + 1).log()

content = []
res = test.ANOVA('log_det', 'experiment * measure * hemi * subject(experiment)', ds=all_ds, title='Both')
content.append(res.table())
# for ds in [ds_e, ds_s]:
#     res = test.ANOVA('log_det', 'experiment * hemi * subject(experiment)', ds=ds, title=ds[0, 'measure'].capitalize())
#     content.append(res.table())
p = plot.Barplot('log_det', 'experiment % measure', match='subject', ds=all_ds, w=3, corr=False)
content.append(p)
for hemi in ['lh', 'rh']:
    res = test.ANOVA('log_det', 'experiment * measure * subject(experiment)', sub=f"hemi == '{hemi}'", ds=all_ds, title=hemi.upper())
    content.append(res.table())
fmtxt.FloatingLayout(content)
# -

# ### Test proportion:
# $$\frac {entropy} {surprisal}$$

ds_rm = table.repmeas('det', 'measure', 'subject', ds=all_ds)
ds_rm['prop'] = ds_rm['entropy'] / ds_rm['surprisal']  # + ds_rm['entropy'].x)
# display(ds_rm)
res = test.TTestIndependent('prop', 'experiment', ds=ds_rm)
p = plot.Barplot('prop', 'experiment', ds=ds_rm, w=3, title=res)
fmtxt.FloatingLayout([p, res.full])

# ### Power

import pingouin

ds = roi_det(tests['surprisal'], ROI, norm=NORM, group=GROUP)
ds = ds.sub("hemi == 'lh'")#.as_dataframe()
d = ds['det'].mean() / ds['det'].std()
print(f"{d=}")
power = pingouin.power_ttest(d=d, n=ds.n_cases, contrast='paired', alternative='greater')
print(f"n={ds.n_cases}, {power=}")

# ### Mono- vs multi-syllabic words

# +
GROUP = 'good2'
NORM = 'brain_max'
ROI = 'STG301'
base = 'gt8 + phone-0v12345 + cb-cohort-syllable'

dss = []
for measure in ['surprisal', 'entropy']:
    for syllabic in ['multi', 'mono']:
        ds = roi_det(f'{base} @ phone-{measure}-{syllabic}syllabic', ROI, norm=NORM, group=GROUP)
        ds[:, 'measure'] = measure
        ds[:, 'syllabic'] = syllabic
        dss.append(ds)
ds_syll = combine(dss)
ds_syll['shared'] = Factor(ds_syll['subject'].isin(shared_subjects), labels={True: 'shared', False: 'unshared'})
# -

ds_syll.head()

test.ANOVA('det', 'measure * syllabic * hemi * subject', ds=ds_syll)

test.pairwise('det', 'measure % syllabic', match='subject', ds=ds_syll, corr=False)

test.ttest('det', 'measure % syllabic', match='subject', ds=ds_syll, corr=False, tail=1)

p = plot.Barplot('det', 'measure % syllabic', match='subject', ds=ds_syll, corr=False, h=3, w=4)

# +
import seaborn
from matplotlib.ticker import FuncFormatter

# y_res, y_lim = 2.5e-4, (-2e-5, 1.5e-4)  # as % total
# y_res, y_lim = .04, (-.005, .02)  # as % total
y_res, y_lim = .015, (-.002, .01)  # for brain_vmax

fig = AbsoluteLayoutFigure(y_res, h=3, w=2.)
formatter = FuncFormatter(lambda x, pos: '%g' % (100 * x))

y0s = [1.7, 0.7]
colors = [UNAMBIGUOUS_COLORS[c] for c in ['orange', 'sky blue']]
axes = []
for y0, measure, color, is_last in zip(y0s, ['entropy', 'surprisal'], colors, [False, True]):
    for syllabic, left, is_right in zip(['mono', 'multi'], [0.2, 0.8], [False, True]):
        ax = fig.add_axes(y0, y_lim, left, 0.4)
        # format axes
        ax.spines['right'].set_visible(False)
        ax.spines['right'].set_position(('outward', 1))
        ax.yaxis.set_label_position('right')
#         ax.set_yticks([0.5e-4, 1.5e-4], minor=True)
        ax.tick_params(left=False, right=False, top=False, labelright=True, labelleft=False, labelbottom=is_last)
        ax.tick_params(axis='y', pad=4)
        ax.grid(True, 'major', 'y', color='0.8')
#         ax.grid(True, 'minor', 'y', color='0.9')
        for tick in ax.yaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)
            tick.label.set_visible(False)
        if is_right:
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.set_yticklabels(())
        # prepare data
        sds = ds_syll.sub((ds_syll["measure"] == measure) & (ds_syll['syllabic'] == syllabic))
        sds['is_rh'] = sds['hemi'] == 'rh'
        df = sds.as_dataframe()
        y = sds['det'].x
        palette = {'shared': (0, 0, 0), 'unshared': color}
        # finalize
        h = seaborn.swarmplot(x='is_rh', y='det', hue='shared', data=df, ax=ax, size=2, clip_on=False, palette=palette)
        # remove crap
        ax.get_legend().remove()
        if is_right:
            ax.set_ylabel('')
        else:
            ax.tick_params(labelright=False, labelleft=True)
            ax.set_ylabel(measure)
        ax.set_xlabel('')
        # append
        axes.append(ax)
        if is_last:
            ax.set_xticklabels(['L', 'R'])
            if is_right: 
                ax.set_ylabel("∆ explained variability (%)", y=-.005, ha='left', labelpad=8)
        else:
            ax.set_title(syllabic)

fig.finalize()
fig.save(DST / f'Models by-subject brain_vmax syllabic {GROUP}.pdf')
# -

# # TRFs

SCALE = 'original'
SCALE = None
trfs = e.load_trfs(-1, MODEL, scale=SCALE, **WHOLEBRAIN)

# ## Mask

res_surprisal = e.load_model_test(f'{MODEL_FULL} @ phone-surprisal', **WHOLEBRAIN, **TEST)
mask = set_parc(res_surprisal.p <= 0.05, 'aparc')

# ### Make symmetric
# (can be commented out)

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

# ## Time-course
# ### PCA

# PCA
masked_trf = trfs['phone_surprisal'].sub(source=mask)
roi_ds = cssl.roi.pca_roi_timecourse(masked_trf, trfs, hemi='both', tstart=0, tstop=1)
f = roi_ds['component'].max('source')
roi_ds['component'] /= f
roi_ds['source'] *= f

args = dict(ds=roi_ds, tstart=0, tstop=1, tfce=True)#, pmin=0.05)
ress_pca = {hemi: testnd.TTestOneSample('source', sub=f"hemi == {hemi!r}", **args) for hemi in ['lh', 'rh']}
ress_pca['hemi'] = testnd.TTestRelated('source', 'hemi', match='subject', **args)
fmtxt.FloatingLayout([res.clusters.sub("p<=0.05").sorted('tstart').as_table(title=hemi) for hemi, res in ress_pca.items()])

ress_pca['hemi'], ress_pca['hemi'].p.argmin()

# $10^{-12}$: p (pico)

# +
from eelbrain._colorspaces import lch_to_rgb
import matplotlib.ticker


colors = {
    'lh': lch_to_rgb(40, 70, 220/360),  # blue is like negative in brain map
#     'rh': lch_to_rgb(30, 10, 220/360),
#     'lh': lch_to_rgb(70, 80, 164/360),
    'rh': lch_to_rgb(10, 20, 164/360),
    'hemi': 'orange',
}
styles = {
    key: plot.Style(c,
#                   linestyle='--', 
                  masked={
#                       'linewidth': 0, 
#                       'saturation': 0.66,
                      'alpha': 0.33,
                  })#, }) 
    for key, c in colors.items()
}


y = resample(roi_ds['source'], 1000)
masks = {hemi: ress_pca[hemi].p > 0.05 for hemi in ['lh', 'rh']}
masks = {hemi: resample(mask.astype(float), 1000) > 0.5 for hemi, mask in masks.items()}

p = plot.UTSStat(
    y.sub(time=(-0.100, 1.001)), 
    'hemi', match='subject', ds=roi_ds, colors=styles, w=4, h=2, xlim=(-0.1, 1.0), clip=False, 
#     legend=False,
#     legend=(.52, .71),  # just for this plot
    legend=(.52, .75),  # higher to indicate covering both plots
    labels = {'lh': 'Left hemisphere', 'rh': 'Right hemisphere'},
    frame='none', 
    xlabel='Time (ms)',
#     frame='t', 
    ylabel='Normlized source current',
    mask=masks,
    #     ylabel='$$',
)
p.set_clusters(ress_pca['hemi'].clusters, y=-.0015, color=colors['hemi'], zorder=2)  # 1.9e-12

ax = p.figure.axes[0]
# x-axis
xticks = numpy.arange(-100, 1001, 100)
ax.set_xticks(xticks * 1e-3)
ax.set_xticklabels(['' if x % 200 else x for x in xticks])
ax.set_xlim(-0.1, 1)
# y-axis
yticks = numpy.arange(-0.002, 0.0031, 0.001)
ax.set_yticks(yticks)
ax.set_yticklabels(int(y*1000) for y in yticks)
ax.set_ylim(-0.002, 0.003)
ax.set_ylabel('First component amplitude', y=1.04, ha='right')
# grid
ax.tick_params('both', length=0, grid_color=(.8, .8, .8))
ax.grid(True, 'major', 'both', color='0.8', clip_on=False)
ax.patch.set_visible(False)


# ax.set_yticks([-.002, 0, .002])
# ax.set_yticklabels([-2, 0, 2])
spine = ax.spines['right'].set_visible(False)

rect = ax.get_position()
# spine.set_bounds(-0.002, 0.002)
p.save(DST / 'TRF pca-stc R1.pdf')
# -

p = plot.ColorList(colors, ['lh', 'rh', 'hemi'], {'lh': 'Left hemisphere', 'rh': 'Right hemisphere', 'hemi': 'Left ≠ right'}, shape='line', w=2)
# p.save(DST / 'TRF pca-stc legend hemi.pdf')
p = plot.ColorList(colors, ['lh', 'rh'], {'lh': 'Left hemisphere', 'rh': 'Right hemisphere'}, shape='line', w=2)
# p.save(DST / 'TRF pca-stc legend.pdf')

# ### Amplitude

# Amplitude
dss = []
for hemi in ('lh', 'rh'):
    ds = trfs['subject',]
    ds[:, 'hemi'] = hemi
    hemi_mask = mask.sub(source=hemi)
    ds['amplitude'] = trfs['phone_surprisal'].sub(source=hemi_mask).abs().sum('source')
    dss.append(ds)
amplitude_ds = combine(dss)

args = dict(ds=amplitude_ds, tstart=0, tstop=1, tfce=True)#, pmin=0.05)
res = testnd.TTestRelated('amplitude', 'hemi', match='subject', **args)
res

# +
p = plot.UTSStat(
    'amplitude.sub(time=(-0.100, 1.001))',
    'hemi', match='subject', ds=amplitude_ds, colors=styles, w=4, h=2, xlim=(-0.1, 1.0), clip=False, 
    legend=False,
#     legend=(.52, .61),
    labels = {'lh': 'Left hemisphere', 'rh': 'Right hemisphere'},
    frame='none', 
    xlabel='Time (ms)',
#     frame='t', 
    ylabel='Normlized source current',
)

ax = p.figure.axes[0]
# x-axis
xticks = numpy.arange(-100, 1001, 100)
ax.set_xticks(xticks * 1e-3)
ax.set_xticklabels(['' if x % 200 else x for x in xticks])
ax.set_xlim(-0.1, 1)
# y-axis
yticks = numpy.linspace(0, 0.25, 5)
ax.set_yticks(yticks)
ax.set_yticklabels((0, '', '', '', 1))
# ax.set_yticklabels([0, 1, 3])
ax.set_ylim(0.0, 0.25)
ax.set_ylabel('Normalized source current', y=1.04, ha='right')
# grid
ax.tick_params('both', length=0, grid_color=(.8, .8, .8))
ax.grid(True, 'major', 'both', color='0.8', clip_on=False)
ax.patch.set_visible(False)

brain_time_windows = [(0.000, 0.200), (0.250, 0.450)]
for tstart, tstop in brain_time_windows:
    ax.hlines(0.245, tstart, tstop, color='k')

spine = ax.spines['right'].set_visible(False)
ax.set_position(rect)
p.save(DST / 'TRF amplitude.pdf')
# -

# ## Brain

# +
cmap = plot.soft_threshold_colormap('polar-a', 0.1, 0.8)

component = roi_ds['component'].mean('case') * 4
component = complete_source_space(component)
sp = plot.brain.SequencePlotter()
sp.set_brain_args(surf='pial', mask=False, cortex=CORTEX)
sp.add_ndvar(component, cmap=cmap, smoothing_steps=20)
p = sp.plot_table(view='lateral', orientation='vertical', axw=1.8, axh=1.35, dpi=300, hemi_magnet=0.05)  # axw=1.2, axh=.8,
p.save(DST / 'TRF pca brain.pdf')
# -

cmap.vmax, cmap.vmin = 1, -1
ticks = {-1: 'Inward', 1: 'Outward'}
p = plot.ColorBar(cmap, w=2, h=1, width=0.05, ticks=ticks, label='Source current', background=CORTEX[0])
p.save(DST / 'TRF pca brain cmap.pdf')

# +


for key, res in ress.items():
    sp = plot.brain.SequencePlotter()
    sp.set_brain_args(surf='pial', cortex=('.8', '.6'))
#     sp.set_brain_args(surf='smoothwm', cortex=('.8', '.6'))
    sp.set_parallel_view(scale=65)
    # sp.set_parallel_view(*brain_view)
#     y = res.t
#     y = res.masked_parameter_map(0.05)
#     y = res.difference
    y = res.masked_difference()

    # cmap based on maximum model fit (per experiment)
#     vmax = res.c1_mean.max()
#     brain_cmap = plot.soft_threshold_colormap('lux-a', vmax/1000, vmax/100)
    
    sp.add_ndvar(y, cmap=brain_cmap, alpha=0.8, smoothing_steps=15)#, lighting=True)
    # ROI outline
    load.update_subjects_dir(roi, y.source.subjects_dir)   
    sp.add_ndvar_label(roi, (1., 1., 1.), borders=3, overlay=True, lighting=False)
    # table
    p = sp.plot_table(view='lateral', orientation='vertical', axw=1.2, axh=.8, dpi=300)
    p.save(DST / f'Model brain-det {key}.pdf')
# -
# ### Amplitude


# +
cmap = plot.soft_threshold_colormap('lux-a', 0.004, 0.012)  # v-max = 0.009
axw = 1.15
axh = axw / 1.8 * 1.35

sp = plot.brain.SequencePlotter()
sp.set_brain_args(surf='pial', mask=False, cortex=CORTEX)
for time_window in brain_time_windows:
    y = trfs['phone_surprisal'].sub(time=time_window).abs().mean('case').sum('time')
    y = y.smooth('source', 0.005, 'gaussian')
    start, stop = time_window
    sp.add_ndvar(y, cmap=cmap, smoothing_steps=20, label=f'{start*1000:.0f} - {stop*1000:.0f} ms')
margins = dict(top=0.3)
p = sp.plot_table(view='lateral', orientation='horizontal', axw=axw, axh=axh, dpi=300, hemi_magnet=0.05, margins=margins, mode='rgba')  # axw=1.2, axh=.8,
p.save(DST / 'TRF amplitude brain.pdf')
# -

# cmap.vmax, cmap.vmin = 1, 0
ticks = {0: '0', cmap.vmax: '1'}#, 1: 'Outward'}
p = plot.ColorBar(cmap, clipmin=0, w=2, h=1, width=0.05, ticks=ticks, label='Source current amplitude', background=CORTEX[0])
p.save(DST / 'TRF amplitude brain cmap.pdf')
