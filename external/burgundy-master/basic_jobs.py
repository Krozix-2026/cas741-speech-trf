"""Jobs with cross-validation"""
import sys
sys.modules.pop('burgundy', None)
from burgundy import e


BASE = {
    'raw': 'ica1-20',
    'samplingrate': 100,
    'cv': True,
    'partitions': -5,
}
BACKWARD = {
    **BASE,
    'data': 'meg',
    'backward': True,
    'tstart': -0.500,
    'tstop': 0.000,
    'basis': 0,
}
WHOLEBRAIN = {
    **BASE,
    'inv': 'fixed-6-MNE-0',
    'mask': 'wholebrain',
    'tstart': -0.100,
    'tstop': 1.000,
    'selective_stopping': 1,
}
TEMPORAL = {**WHOLEBRAIN, 'mask': 'lateraltemporal'}
# TO_WB = {**TEMPORAL, 'mask': 'lateral_to_wholebrain'}
TO_WB = {**TEMPORAL, 'mask': 'stg_to_wholebrain'}
STG = {**WHOLEBRAIN, 'mask': 'superiortemporal'}


JOBS = [
    # Preliminaries
    # -------------
    # Backward model
    e.trf_job('gammatone-1', **BACKWARD, group='okay-data'),
    # Cohort effect on phone 0
    e.model_job('gt8 + phone-0v12345 +@ cb-cohort-0v12345 > cb-cohort', **TEMPORAL),
    e.model_job('gt8 + phone-0v12345 +@ cb-cohort > cb-cohort-12345', **TEMPORAL),
    # should word offset be controlled for?
    e.model_job('gt8 + phone-0v12345 + cb-cohort +@ phone-offset', **TEMPORAL),

    # Lexical processing
    # ------------------
    e.model_job('gt8 + phone-0v12345 + cb-cohort @ phone-entropy', **TO_WB, group='good2-data'),
    e.model_job('gt8 + phone-0v12345 + cb-cohort @ phone-surprisal', **TO_WB, group='good2-data'),

    # + Phoneme entropy
    # -----------------
    e.model_job('gt8 + phone-0v12345 + cohort @ phone-entropy', **STG, group='good2-data'),
    e.model_job('gt8 + phone-0v12345 + cohort @ phone-phoneme_entropy', **STG, group='good2-data'),
    # entropy measures
    e.model_job('gt8 + phone-0v12345 + phone-surprisal +@ phone-entropy', **WHOLEBRAIN),
    e.model_job('gt8 + phone-0v12345 + phone-surprisal +@ phone-phoneme_entropy', **WHOLEBRAIN),

    # Multisyllabic words
    # -------------------
    e.model_job('gt8 + phone-0v12345 + cb-cohort-syllable @ phone-entropy-multisyllabic', **STG),
    e.model_job('gt8 + phone-0v12345 + cb-cohort-syllable @ phone-entropy-monosyllabic', **STG),
    e.model_job('gt8 + phone-0v12345 + cb-cohort-syllable @ phone-surprisal-multisyllabic', **STG),
    e.model_job('gt8 + phone-0v12345 + cb-cohort-syllable @ phone-surprisal-monosyllabic', **STG),

    # Auditory processing
    # -------------------
    # e.model_job('gt8 + phone-0v12345 + cb-cohort @ phone-p0', **WHOLEBRAIN),
    # e.model_job('gt8 + phone-0v12345 + cb-cohort @ gammatone-8', **WHOLEBRAIN),
    # e.model_job('gt8 + phone-0v12345 + cb-cohort @ gammatone-edge-8', **WHOLEBRAIN),

    # UP
    # --
    # e.model_job('gt8 + phone-0v12345 + phone-surprisal + phone-phoneme_entropy +@ phone-up + phone-log10wf-up', **TO_LANGUAGE),
    # e.model_job('gt8 + phone-0v12345 + phone-surprisal + phone-phoneme_entropy +@ phone-log10wf-p0', **TO_LANGUAGE),
]
