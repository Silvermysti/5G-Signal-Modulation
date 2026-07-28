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

import ast
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

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

# Folder layout (this file lives in scripts/, data is in ../Data)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "Data"
OUT_DIR = HERE.parent / "prepared"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)  # our own random-number generator, seeded

# ----------------------------------------------------------------------------
# 1. Read the class names (classes.txt is a Python line: `classes = [...]`)
# ----------------------------------------------------------------------------
text = (DATA_DIR / "classes.txt").read_text()
bracketed = text[text.index("["): text.rindex("]") + 1]  # just the [...] part
ALL_CLASSES = ast.literal_eval(bracketed)                 # safely parse to a list
print(f"Dataset has {len(ALL_CLASSES)} classes total.")

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
picked_y = []      # the NEW label (0,1,2) for our 3-class problem
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
X = np.asarray(signals_mm[picked_rows], dtype=np.float32)      # shape (15000, 1024, 2)
print("Gathered signal subset into RAM:", X.shape,
      f"({X.nbytes / 1e6:.0f} MB)")

# ----------------------------------------------------------------------------
# 4. Normalize each snippet to unit variance (the paper's recipe)
#    Each example is scaled on its own so loudness/volume doesn't matter,
#    only the *shape* of the signal does.
# ----------------------------------------------------------------------------
per_example_std = X.std(axis=(1, 2), keepdims=True)   # one std per snippet
per_example_std[per_example_std == 0] = 1.0           # avoid divide-by-zero
X = X / per_example_std

# ----------------------------------------------------------------------------
# 5. Train / test split (stratify keeps the 3 classes balanced in both halves)
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, picked_y, test_size=TEST_FRACTION, random_state=SEED, stratify=picked_y
)
print(f"Train set: {X_train.shape[0]} examples | Test set: {X_test.shape[0]} examples")

# Save everything for the train/evaluate scripts to load.
np.save(OUT_DIR / "X_train.npy", X_train)
np.save(OUT_DIR / "X_test.npy", X_test)
np.save(OUT_DIR / "y_train.npy", y_train)
np.save(OUT_DIR / "y_test.npy", y_test)
(OUT_DIR / "classes.txt").write_text("\n".join(CHOSEN_CLASSES))
print(f"Saved prepared data to: {OUT_DIR}")
print("Done. Next step: train the model.")
