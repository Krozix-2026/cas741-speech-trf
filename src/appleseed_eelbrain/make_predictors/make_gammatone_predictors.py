"""
Predictors based on gammatone spectrograms

Assumes that ``make_gammatone.py`` has been run to create the high resolution
spectrograms.
"""
from pathlib import Path
import os
import re
import eelbrain


# Define paths to data, and destination for predictors
DATA_ROOT = Path(os.environ.get("ALICE_ROOT", r"C:\Dataset\Appleseed")).expanduser()
STIMULUS_DIR = DATA_ROOT / "stimuli"
PREDICTOR_DIR = DATA_ROOT / "predictors"

# If the directory for predictors does not exist yet, create it
PREDICTOR_DIR.mkdir(exist_ok=True)

# ---- Robust: discover stimuli from existing gammatone pickles ----
# Expect files like: "1-gammatone1000.pickle", "11b-gammatone1000.pickle"
pickle_paths = sorted(STIMULUS_DIR.glob("*-gammatone1000.pickle"))

if not pickle_paths:
    raise FileNotFoundError(
        f"No '*-gammatone1000.pickle' found in {STIMULUS_DIR}. "
        f"Run make_gammatone.py first or check filenames."
    )

# Natural sort: 1,2,...,11,11b (instead of 1,10,11,11b,2...)
def natural_key(p: Path):
    stem = p.name.replace("-gammatone1000.pickle", "")
    # split into digit / non-digit chunks
    parts = re.split(r"(\d+)", stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key

pickle_paths = sorted(pickle_paths, key=natural_key)

# Loop through discovered stimuli keys
for p in pickle_paths:
    key = p.name.replace("-gammatone1000.pickle", "")  # e.g. "1", "11b"
    gt = eelbrain.load.unpickle(p)

    # Apply a log transform to approximate peripheral auditory processing
    gt_log = (gt + 1).log()
    # Apply the edge detector model to generate an acoustic onset spectrogram
    gt_on = eelbrain.edge_detector(gt_log, c=30)

    # Create and save 1 band versions of the two predictors (i.e., temporal envelope predictors)
    eelbrain.save.pickle(gt_log.sum("frequency"), PREDICTOR_DIR / f"{key}~gammatone-1.pickle")
    eelbrain.save.pickle(gt_on.sum("frequency"), PREDICTOR_DIR / f"{key}~gammatone-on-1.pickle")

    # Create and save 8 band versions of the two predictors (binning the frequency axis into 8 bands)
    x = gt_log.bin(nbins=8, func="sum", dim="frequency")
    eelbrain.save.pickle(x, PREDICTOR_DIR / f"{key}~gammatone-8.pickle")
    x = gt_on.bin(nbins=8, func="sum", dim="frequency")
    eelbrain.save.pickle(x, PREDICTOR_DIR / f"{key}~gammatone-on-8.pickle")

    # Create gammatone spectrograms with linear scale, only 8 bin versions
    x = gt.bin(nbins=8, func="sum", dim="frequency")
    eelbrain.save.pickle(x, PREDICTOR_DIR / f"{key}~gammatone-lin-8.pickle")

    # Powerlaw scale
    gt_pow = gt ** 0.6
    x = gt_pow.bin(nbins=8, func="sum", dim="frequency")
    eelbrain.save.pickle(x, PREDICTOR_DIR / f"{key}~gammatone-pow-8.pickle")

    print(f"Done: {key}")