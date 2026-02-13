from pathlib import Path
import mne

root = Path(r"C:\Dataset\Appleseed_BIDS_new")

for meg_dir in root.glob("sub-*/meg"):
    parts = sorted(meg_dir.glob("*_task-Appleseed_run*split-*_meg.fif"))
    if not parts:
        continue

    out = meg_dir / parts[0].name.replace("_split-01", "").replace("_split-02", "")
    # 保险：确保 out 名字确实是 ..._task-Appleseed_meg.fif
    out = meg_dir / out.name.replace("_split-01", "").replace("_split-02", "")

    if out.exists():
        print("exists:", out)
        continue

    print("merge:", meg_dir.parent.name, "n_parts=", len(parts))
    raws = [mne.io.read_raw_fif(p, preload=False) for p in parts]
    raw = mne.concatenate_raws(raws)
    raw.save(out, overwrite=False)
    print("wrote:", out)