# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setup

# +
from itertools import chain
from eelbrain import *
from trftools.pipeline import ResultCollection

from basic_jobs import e, STG, TEMPORAL, WHOLEBRAIN, BACKWARD


TEST3 = {
    'smooth': 0.003,
    'metric': 'det',
}
TEST5 = {
    'smooth': 0.005,
    'metric': 'det',
}
TEST10 = {
    'smooth': 0.010,
    'metric': 'det',
}
# -

# # Preliminaries
# ## Backward model

ds = e.load_trfs(-1, 'gammatone-1', **BACKWARD, group='okay-data')

p = plot.Boxplot('z', ds=ds)

# ## First phoneme

display(e.show_comparison_terms('gt8 + phone-0v12345 +| cb-cohort-0v12345 > cb-cohort', True))

e.show_model_test({
    '0v1: > all': 'gt8 + phone-0v12345 +| cb-cohort-0v12345 > cb-cohort',
    '0v1: = all': 'gt8 + phone-0v12345 +| cb-cohort-0v12345 = cb-cohort',
    'all = 1:': 'gt8 + phone-0v12345 +| cb-cohort = cb-cohort-12345',
    '0v1: > 1:': 'gt8 + phone-0v12345 + cb-cohort-0v12345 | cb-cohort-0',
}, **TEMPORAL, **TEST5, xhemi=False, sig=False, brain_view='temporal', surf='pial')

e.show_model_test({
    '0v1: = all': 'gt8 + phone-0v12345 +| cb-cohort-0v12345 = cb-cohort',
    '0v1: > all': 'gt8 + phone-0v12345 +| cb-cohort-0v12345 > cb-cohort',
    '0v1: < all': 'gt8 + phone-0v12345 +| cb-cohort-0v12345 < cb-cohort',
    'all = 1:': 'gt8 + phone-0v12345 +| cb-cohort = cb-cohort-12345',
    'all > 1:': 'gt8 + phone-0v12345 +| cb-cohort > cb-cohort-12345',
    'all < 1:': 'gt8 + phone-0v12345 +| cb-cohort < cb-cohort-12345',
}, **STG, **TEST5, xhemi=False, sig=False, brain_view='temporal')

# ## N partitions

x = 'gt8 + phone-0v12345 + cohort'
e.show_model_test({'4=5': x}, parameter='partitions', compare_to=-5, **LTL, **TEST, sig=False)

# ## Word offset

e.show_model_test({
    'p0 = all': 'gt8 + phone-0v12345 +  cohort +| phone-offset',
}, **LANGUAGE, **TEST)

# # Auditory

e.show_model_test({
    'Envelope': 'gt8 + phone-0v12345 + cohort | gammatone-8',
    'Onsets':   'gt8 + phone-0v12345 + cohort | gammatone-edge-8',
    'Word onset':   'gt8 + phone-0v12345 + cohort | phone-p0',
}, **WB, **TEST5, surf='pial', cmap='lux-a',
    vmax=0.001,
)

e.show_model_test({
    'Envelope': 'gt8 + phone-0v12345 + cohort | gammatone-8',
    'Onsets':   'gt8 + phone-0v12345 + cohort | gammatone-edge-8',
    'Word onset':   'gt8 + phone-0v12345 + cohort | phone-p0',
}, **WB, **TEST3, surf='pial', cmap='lux-a',
    vmax=0.001,
)

# # Cohort model
# - surprisal
# - cohort entropy
# - phoneme entropy

comparisons_see = {
    'surprisal': 'gt8 + phone-0v12345 + cohort @ phone-surprisal',
    'entropy combined': 'gt8 + phone-0v12345 + cohort @ phone-entropy + phone-phoneme_entropy',
    'cohort-entropy': 'gt8 + phone-0v12345 + cohort @ phone-entropy',
    'phone-entropy': 'gt8 + phone-0v12345 + cohort @ phone-phoneme_entropy',
    'only cohort-entropy': 'gt8 + phone-0v12345 + phone-surprisal +@ phone-entropy',
    'only phone-entropy': 'gt8 + phone-0v12345 + phone-surprisal +@ phone-phoneme_entropy',
}

# ## Temporal lobe

e.show_model_test(comparisons_see, **TEMPORAL, **TEST3, surf='pial', cmap='lux-a', 
#     sig=False, 
    vmax=0.0001,
)

e.show_model_test(comparisons_see, **TEMPORAL, **TEST5, surf='pial', cmap='lux-a', 
#     sig=False, 
    vmax=0.0001,
)

# ## Whole brain

e.show_model_test(comparisons_see, **WHOLEBRAIN, **TEST3, surf='pial', cmap='lux-a', 
#     sig=False, 
    vmax=0.0001,
)

# ### 5 mm smoothing

e.show_model_test(comparisons_see, **WHOLEBRAIN, **TEST5, surf='pial', cmap='lux-a', 
#     sig=False, 
    vmax=0.0001)

# ## STG: Phoneme- vs cohort-entropy

e.show_model_test({
    'entropy': 'gt8 + phone-0v12345 + cohort | phone-entropy + phone-phoneme_entropy',
    'cohort-entropy': 'gt8 + phone-0v12345 + cohort | phone-entropy',
    'phone-entropy': 'gt8 + phone-0v12345 + cohort | phone-phoneme_entropy',
    'only cohort-entropy': 'gt8 + phone-0v12345 + phone-surprisal +| phone-entropy',
    'only phone-entropy': 'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy',
    'phone=cohort': 'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy = phone-entropy',
    'phone>cohort': 'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy > phone-entropy',    
}, **STG, **TEST5,
brain_view='temporal', surf='pial', cmap='lux-a')#, sig=False)

# ## ROI: Phoneme- vs cohort-entropy
# - Find region where entropy (combined) is significant
# - Use UV tests in this ROI

e.show_model_test({
    'phone-entropy': 'gt8 + phone-0v12345 + phone-surprisal +| phone-entropy + phone-phoneme_entropy',
}, **LTL5, **TEST)

# ## All subjects

comparisons2 = {
#     'surprisal': 'gt8 + phone-0v12345 + cohort | phone-surprisal',
    'entropy combined': 'gt8 + phone-0v12345 + cohort | phone-entropy + phone-phoneme_entropy',
    'cohort-entropy': 'gt8 + phone-0v12345 + cohort | phone-entropy',
    'phone-entropy': 'gt8 + phone-0v12345 + cohort | phone-phoneme_entropy',
    'only cohort-entropy': 'gt8 + phone-0v12345 + phone-surprisal +| phone-entropy',
    'only phone-entropy': 'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy',
}
e.show_model_test(comparisons2, **WHOLEBRAIN, **TEST5, surf='pial', cmap='lux-a', 
    sig=False, 
    vmax=0.0001,
)

# # Cohort-entropy only
# - surprisal
# - cohort-entropy

comparisons_se = {
    'surprisal': 'gt8 + phone-0v12345 + phone-entropy +| phone-surprisal',
    'entropy':   'gt8 + phone-0v12345 + phone-surprisal +| phone-entropy',
    'surprisal > entropy': 'gt8 + phone-0v12345 + phone-surprisal + phone-entropy | phone-surprisal = phone-entropy',
}

e.show_model_test(comparisons_se, **WHOLEBRAIN, **TEST5, surf='pial', cmap='lux-a', vmax=0.0001)

e.show_model_test(comparisons_se, **WHOLEBRAIN, **TEST5, surf='pial', cmap='polar-a', vmax=0.0001, sig=False, xhemi=False)

# ## Add phoneme entropy

comparisons = {
    'cohort-entropy':   'gt8 + phone-0v12345 + phone-surprisal +| phone-entropy',
    'phoneme-entropy':   'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy',
    'phoneme = cohort':   'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy = phone-entropy',
}
e.show_model_test(comparisons, **WB, **TEST5, surf='pial', cmap='lux-a', vmax=0.0001, sig=False)

# ## Entropy in STG

e.show_model_test({
    'cohort-entropy': 'gt8 + phone-0v12345 + phone-surprisal +| phone-entropy',
    'phoneme-entropy':   'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy',
    'phoneme = cohort':   'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy = phone-entropy',
}, **STG, **TEST5, brain_view='temporal', surf='pial', cmap='lux-a')#, sig=False)

e.show_model_test({
    'cohort-entropy': 'gt8 + phone-0v12345 + phone-surprisal +| phone-entropy',
    'phoneme-entropy':   'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy',
    'phoneme = cohort':   'gt8 + phone-0v12345 + phone-surprisal +| phone-phoneme_entropy = phone-entropy',
}, **LTL5, **TEST5, brain_view='temporal', surf='pial', cmap='lux-a')#, sig=False)

# ## Multi-syllabic words

base = 'gt8 + phone-0v12345 + cb-cohort-syllable'
e.show_model_test({
    'Surprisal-multi': f'{base} @ phone-surprisal-multisyllabic',
    'Entropy-multi': f'{base} @ phone-entropy-multisyllabic',
}, **STG, **TEST5, brain_view='temporal', surf='pial', cmap='lux-a')#, sig=False)

# # Word level

e.show_model_test({
    'Frequency': 'gt8 + phone-0v12345 + phone-surprisal + phone-phoneme_entropy +| phone-log10wf-p0',
    'UP-Freq': 'gt8 + phone-0v12345 + phone-surprisal + phone-phoneme_entropy +| phone-up + phone-log10wf-up',
}, **LANGUAGE, **TEST)
