"""
Step 2 — DATA PREP
==================

Goal: pull a small, balanced training set out of the giant RadioML dataset.

We want the 10 modulations listed in CHOSEN_CLASSES below, 8,000 examples of each.
The signals file is ~19.5 GB, so we must NOT load it all into memory. Instead we:

  1. Load the small labels file fully (235 MB is fine) to find WHICH rows we want.
  2. Randomly pick PER_CLASS row-numbers per class.
  3. Memory-map the big signals file and read ONLY those rows off disk.
  4. Normalize each snippet, then split into train / test sets.
  5. Save the result to the `prepared/` folder so the next scripts can reuse it instantly.

Run it with:   .venv/bin/python scripts/data_prep.py
"""

import argparse
import ast
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Sample a balanced subset of RadioML.")
parser.add_argument("--per-class", type=int, default=None,
                    help="examples per class (overrides PER_CLASS below)")
parser.add_argument("--all-24", action="store_true",
                    help="use every class in the dataset, not just CHOSEN_CLASSES")
args = parser.parse_args()

# ----------------------------------------------------------------------------
# Settings you can tweak
# ----------------------------------------------------------------------------
# 10 modulations spanning the major families. Two deliberately-confusable
# "ladders" are included so the confusion matrix shows real structure:
#   PSK ladder:  BPSK -> QPSK -> 8PSK   (denser phase steps, look alike)
#   QAM ladder:  16QAM -> 64QAM -> 256QAM (denser constellations, look alike)
# plus distinct anchors (OOK, FM, AM, GMSK) that stay easy to tell apart.
# QPSK / 16 / 64 / 256QAM are the workhorses of real 5G data channels.
CHOSEN_CLASSES = [
    "OOK",                    # amplitude on/off  -- easy anchor
    "BPSK", "QPSK", "8PSK",   # PSK ladder        -- confusable cluster
    "16QAM", "64QAM", "256QAM",  # QAM ladder     -- confusable cluster
    "FM",                     # analog frequency  -- easy anchor
    "GMSK",                   # GSM/legacy cellular
    "AM-DSB-WC",              # broadcast AM      -- analog anchor
]
PER_CLASS = 8000                          # how many examples of each
TEST_FRACTION = 0.2                       # 20% held out for honest testing
SEED = 42                                 # fixed randomness => reproducible runs

if args.per_class is not None:            # let the command line win
    PER_CLASS = args.per_class

# Folder layout (this file lives in scripts/, data is in ../Data)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "Data"
OUT_DIR = HERE.parent / "prepared"
OUT_DIR.mkdir(exist_ok=True)

# Ground-truth signal-to-noise ratio (SNR) per example. This is the "how clean
# is this snippet" number, in decibels (-20 = buried in noise, +30 = crystal
# clear). It lives outside Data/ and lines up row-for-row with signals.npy. If
# it's present we carry it through so later scripts can check the model against
# real noise levels; if it's missing everything still works without it.
SNR_PATH = HERE.parent / "archive (1)" / "snrs.npy"

rng = np.random.default_rng(SEED)  # our own random-number generator, seeded

# ----------------------------------------------------------------------------
# 1. Read the class names (classes.txt is a Python line: `classes = [...]`)
# ----------------------------------------------------------------------------
text = (DATA_DIR / "classes.txt").read_text()
bracketed = text[text.index("["): text.rindex("]") + 1]  # just the [...] part
ALL_CLASSES = ast.literal_eval(bracketed)                 # safely parse to a list
print(f"Dataset has {len(ALL_CLASSES)} classes total.")

if args.all_24:                           # full paper replication: every class
    CHOSEN_CLASSES = list(ALL_CLASSES)
    print(f"--all-24: using all {len(CHOSEN_CLASSES)} classes.")

# Which COLUMN in the one-hot labels does each chosen class live in?
chosen_cols = [ALL_CLASSES.index(name) for name in CHOSEN_CLASSES]
print("Chosen classes and their label columns:",
      dict(zip(CHOSEN_CLASSES, chosen_cols)))

# ----------------------------------------------------------------------------
# 2. Load labels fully, find row numbers for each chosen class
# ----------------------------------------------------------------------------
labels = np.load(DATA_DIR / "labels.npy")          # shape (2555904, 24), one-hot
label_idx = np.argmax(labels, axis=1)              # turn one-hot into 0..23 per row
print("Loaded labels:", labels.shape)

picked_rows = []   # the row-numbers we will read from the big file
picked_y = []      # the NEW label (0..N-1) for our chosen-class problem
for new_label, col in enumerate(chosen_cols):
    rows_for_class = np.flatnonzero(label_idx == col)   # all rows of this modulation
    if len(rows_for_class) < PER_CLASS:
        raise ValueError(f"{CHOSEN_CLASSES[new_label]} only has {len(rows_for_class)} examples")
    chosen = rng.choice(rows_for_class, size=PER_CLASS, replace=False)  # random subsample
    picked_rows.append(chosen)
    picked_y.append(np.full(PER_CLASS, new_label))
    print(f"  {CHOSEN_CLASSES[new_label]:5s}: {len(rows_for_class):7d} available -> took {PER_CLASS}")

picked_rows = np.concatenate(picked_rows)
picked_y = np.concatenate(picked_y).astype(np.int64)

# Sorting the row-numbers makes reading from disk faster (more sequential).
order = np.argsort(picked_rows)
picked_rows = picked_rows[order]
picked_y = picked_y[order]

# ----------------------------------------------------------------------------
# 3. Memory-map the big signals file and read ONLY our rows
# ----------------------------------------------------------------------------
signals_mm = np.load(DATA_DIR / "signals.npy", mmap_mode="r")  # no data loaded yet
print("Signals memmap:", signals_mm.shape, signals_mm.dtype)

# Fancy-indexing a memmap reads just those rows from disk into a real array.
X = np.asarray(signals_mm[picked_rows], dtype=np.float32)      # shape (80000, 1024, 2)
print("Gathered signal subset into RAM:", X.shape,
      f"({X.nbytes / 1e6:.0f} MB)")

# Gather the matching SNR for each picked row (same order as X), if available.
# snrs.npy is (N, 1); we flatten to a plain 1-D array of one number per example.
picked_snr = None
if SNR_PATH.exists():
    snrs_mm = np.load(SNR_PATH, mmap_mode="r")
    picked_snr = np.asarray(snrs_mm[picked_rows], dtype=np.float32).reshape(-1)
    print(f"Gathered ground-truth SNR: {picked_snr.shape[0]} values, "
          f"range {picked_snr.min():.0f}..{picked_snr.max():.0f} dB")
else:
    print(f"(No SNR file at {SNR_PATH} -- skipping SNR, everything else still works.)")

# ----------------------------------------------------------------------------
# 4. Normalize each snippet to unit variance (the paper's recipe)
#    Each example is scaled on its own so loudness/volume doesn't matter,
#    only the *shape* of the signal does.
# ----------------------------------------------------------------------------
per_example_std = X.std(axis=(1, 2), keepdims=True)   # one std per snippet
per_example_std[per_example_std == 0] = 1.0           # avoid divide-by-zero
X = X / per_example_std

# ----------------------------------------------------------------------------
# 5. Train / test split (stratify keeps the classes balanced in both halves)
#    The split depends only on the seed, test_size and stratify labels, so it is
#    identical whether or not we also hand it the SNR array -- adding SNR does
#    NOT change which rows land in X_test. That keeps an already-trained model
#    valid after a re-run.
# ----------------------------------------------------------------------------
if picked_snr is not None:
    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        X, picked_y, picked_snr,
        test_size=TEST_FRACTION, random_state=SEED, stratify=picked_y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, picked_y, test_size=TEST_FRACTION, random_state=SEED, stratify=picked_y
    )
    snr_train = snr_test = None
print(f"Train set: {X_train.shape[0]} examples | Test set: {X_test.shape[0]} examples")

# Save everything for the train/evaluate scripts to load.
np.save(OUT_DIR / "X_train.npy", X_train)
np.save(OUT_DIR / "X_test.npy", X_test)
np.save(OUT_DIR / "y_train.npy", y_train)
np.save(OUT_DIR / "y_test.npy", y_test)
if snr_test is not None:
    np.save(OUT_DIR / "snr_train.npy", snr_train)
    np.save(OUT_DIR / "snr_test.npy", snr_test)
    print("Also saved per-example SNR: snr_train.npy, snr_test.npy")
(OUT_DIR / "classes.txt").write_text("\n".join(CHOSEN_CLASSES))
print(f"Saved prepared data to: {OUT_DIR}")
print("Done. Next step: train the model.")
