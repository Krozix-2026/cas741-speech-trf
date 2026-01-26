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
from appleseed_jobs import e, FULL, LTL, FTP, FTP_5


TEST3 = {
    'smooth': 0.003,
    'metric': 'det',
}
TEST5 = {
    'smooth': 0.005,
    'metric': 'det',
}
# -

# # Surprisal & Entropy

comps = {
    'surprisal': "gte8 + phone-p0 + phone-p1_ + phone-entropy +| phone-surprisal",
    'entropy':   "gte8 + phone-p0 + phone-p1_ + phone-surprisal +| phone-entropy", 
    'surprisal = entropy': 'gte8 + phone-p0 + phone-p1_ + phone-surprisal + phone-entropy | phone-surprisal = phone-entropy',
}

e.show_model_test(comps, **FTP, **TEST5, surf='pial', cmap='lux-a', vmax=0.0001)

e.show_model_test(comps, **FTP, **TEST5, surf='pial', cmap='polar-a', sig=False, vmax=0.0001, xhemi=False)

# ## First two segments

e.show_model_test({
    'surprisal': f'{FULL} @ phone-surprisal',
    'entropy': f'{FULL} @ phone-entropy',
}, **FTP_5, epoch='seg1-2', **TEST5, surf='pial', cmap='lux-a', vmax=0.0001)

# # Phoneme entropy

comps = {
    '|surprisal': "gte8 + phone-p0 + phone-p1_ + burgundy | phone-surprisal",
    '|entropy': "gte8 + phone-p0 + phone-p1_ + burgundy | phone-entropy", 
    '|phoneme_entropy': "gte8 + phone-p0 + phone-p1_ + burgundy | phone-phoneme_entropy",
    'only entropy' = f"gte8 + phone-p0 + phone-p1_ + phone-surprisal +| phone-entropy"
}

# # Temporal

e.show_model_test(comps, **LTL, **TEST5, surf='pial', cmap='lux-a', vmax=0.0001)

e.show_model_test(comps, **LTL, **TEST3, surf='pial', cmap='lux-a', vmax=0.0001)

# # Tests

e.show_model_test(comps, **FTP, **TEST3, surf='pial', cmap='lux-a', vmax=0.0001, sig=False)

e.show_model_test(comps, **FTP, **TEST5, surf='pial', cmap='lux-a', vmax=0.0001, sig=False)
