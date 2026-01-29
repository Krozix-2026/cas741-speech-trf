"""
Generate predictors for word-level variables

See the `explore_word_predictors.py` notebook for more background
"""
from pathlib import Path
import os
import re
import eelbrain


# Define paths to source data, and destination for predictors
DATA_ROOT = Path(os.environ.get("ALICE_ROOT", r"C:\Dataset\Appleseed")).expanduser()
STIMULUS_DIR = DATA_ROOT / "stimuli"
PREDICTOR_DIR = DATA_ROOT / "predictors"
PREDICTOR_DIR.mkdir(exist_ok=True)

# Load the text file with word-by-word predictor variables
word_table = eelbrain.load.tsv(STIMULUS_DIR / "AliceChapterOne-EEG.csv")

# Add word frequency variable with expected inverse relationship to response
word_table["InvLogFreq"] = 17 - word_table["LogFreq"]


# ---- Robust: discover segments from the table (supports 11b etc.) ----
def natural_key(x):
    """Sort keys like 1,2,...,11,11b instead of 1,10,11,11b,2..."""
    s = str(x)
    parts = re.split(r"(\d+)", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

segments = sorted(set(word_table["Segment"]), key=natural_key)
if not segments:
    raise RuntimeError("No segments found in word_table['Segment'].")

# Loop through discovered segments
for segment in segments:
    # Subset rows for this segment (quote segment in case it's a string like "11b")
    segment_table = word_table.sub(f"Segment == {segment!r}")

    if len(segment_table) == 0:
        # should not happen because we took segments from the table,
        # but keep it safe
        continue

    # tstop: use last word offset (must exist)
    tstop = segment_table[-1, "offset"]

    # Initialize Dataset with word onset times; store stimulus duration in info
    data = eelbrain.Dataset({"time": segment_table["onset"]}, info={"tstop": tstop})

    # Predictor variables
    data["LogFreq"] = segment_table["InvLogFreq"]
    for key in ["NGRAM", "RNN", "CFG", "Position"]:
        data[key] = segment_table[key]

    # Boolean masks for lexical vs non-lexical words
    # (use bool() conversion to handle 0/1 or True/False robustly)
    is_lex = segment_table["IsLexical"].astype(bool) if hasattr(segment_table["IsLexical"], "astype") else (segment_table["IsLexical"] == True)
    data["lexical"] = is_lex
    data["nlexical"] = not is_lex

    # Save
    eelbrain.save.pickle(data, PREDICTOR_DIR / f"{segment}~word.pickle")
    print(f"Saved: {segment}~word.pickle   (N={len(segment_table)}, tstop={tstop})")