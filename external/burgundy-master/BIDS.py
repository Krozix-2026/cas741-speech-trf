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

# # To BIDS

# +
from pathlib import Path
import shutil

from eelbrain import *
import mne
import mne_bids

from burgundy import e


ROOT = Path('/Volumes/Seagate BarracudaFastSSD/Burgundy-BIDS')
# shutil.rmtree(ROOT)
ROOT.mkdir(exist_ok=True)
# -

subjects = sorted(e.get_field_values('subject'))
{subject: f'{i:03}' for i, subject in enumerate(subjects, 1)}

subject_ids = {
 'R2621': '001',
 'R2623': '002',
 'R2626': '003',
 'R2628': '004',
 'R2631': '005',
 'R2633': '006',
 'R2635': '007',
 'R2638': '008',
 'R2640': '009',
 'R2641': '010',
 'R2644': '011',
 'R2647': '012',
 'R2648': '013',
 'R2649': '014',
 'R2651': '015',
 'R2653': '016',
 'R2658': '017',
 'R2659': '018',
}

# ## MEG

event_id = {
    'item': 162, 
    'item_post_probe': 163, 
    'no_probe': 166, 
    'yes_probe': 167,
}
for subject in e:
    s_id = subject_ids[subject]
    dst = mne_bids.BIDSPath(s_id, root=ROOT, task='words')
    if dst.directory.exists():
        continue

    raw = e.load_raw(raw='raw')#, preload=True)
    raw.info['line_freq'] = 60
    events = mne.find_events(raw)
    mne_bids.write_raw_bids(raw, dst, events_data=events, event_id=event_id)
#     break

# ## Empty-room

src = '/Volumes/Seagate BarracudaFastSSD/Burgundy/meg/R2627/R2627_emptyroom-raw.fif'
raw = mne.io.read_raw(src)
raw.info['line_freq']
date = raw.info['meas_date'].strftime('%Y%m%d')
dst = mne_bids.BIDSPath('emptyroom', session=date, task='noise', root=ROOT)
mne_bids.write_raw_bids(raw, dst)

# ## LOG files

for subject, s_id in subject_ids.items():
    src = Path(f'/Volumes/Seagate BarracudaFastSSD/Burgundy/meg/{subject}/{subject}-all_stims.log')
    dst_dir = ROOT / f'sub-{s_id}' / 'beh'
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir()
    dst = dst_dir / f'sub-{s_id}_task-words_beh.tsv'

    ds = load.tsv(src, delimiter='\t', skiprows=3, ignore_missing=True)[1:]
    ds[:, 'Subject'] = s_id
    for key in ['Uncertainty', 'Duration', 'ReqDur', 'Stim_Type']:
        ds[key].update_labels({'': 'n/a'})
    ds.save_txt(dst, nan='n/a')

# # Test

raw = mne.io.read_raw(ROOT / 'sub-001' / 'meg' / 'sub-001_task-words_meg.fif')


