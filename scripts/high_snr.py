"""
Step 8 — THE FAIR COMPARISON TO THE PAPER (accuracy at HIGH SNR)
================================================================

The paper's famous 98.3% (VGG) / 99.8% (ResNet) numbers are NOT an average over
all noise levels. They are the PEAK of the accuracy-vs-SNR curve -- measured only
on clean, high-SNR signals. Our own headline (66.3% / 67.5%) is the opposite: an
average over the ENTIRE range, from +30 dB down to -20 dB where the signal is
buried in noise and no classifier could win. Comparing those two is apples to
oranges.

This script scores each model the SAME way the paper reports: on the clean,
high-SNR slice of the test set. That is the honest side-by-side number.

"High SNR" here means SNR >= +18 dB (a common convention; the paper's curves have
flattened well before this point). We also print the single cleanest level and a
couple of other cut-offs so you can see the whole clean plateau.

Run it with:   .venv/bin/python scripts/high_snr.py
               .venv/bin/python scripts/high_snr.py --model vgg
               .venv/bin/python scripts/high_snr.py --high-snr 6
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hush TF startup noise

import argparse
from pathlib import Path

import numpy as np
import keras

parser = argparse.ArgumentParser(description="Score model(s) on the high-SNR slice.")
parser.add_argument("--model", choices=["vgg", "resnet", "both"], default="both",
                    help="which trained model(s) to score (default: both)")
parser.add_argument("--high-snr", type=float, default=18.0,
                    help="dB cut-off that counts as 'high SNR' (default: 18)")
args = parser.parse_args()

HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / "prepared"
MODEL_DIR = HERE.parent / "models"

# ----------------------------------------------------------------------------
# 1. Load the test set + ground-truth SNR (same idiom as gate.py)
# ----------------------------------------------------------------------------
X_test = np.load(PREP_DIR / "X_test.npy")
y_true = np.load(PREP_DIR / "y_test.npy")
classes = (PREP_DIR / "classes.txt").read_text().split()

snr_path = PREP_DIR / "snr_test.npy"
if not snr_path.exists():
    raise SystemExit(
        "Need prepared/snr_test.npy for this comparison. Re-run data_prep.py "
        "with archive (1)/snrs.npy present."
    )
snr_test = np.load(snr_path).reshape(-1)

print(f"Test set: {X_test.shape[0]} examples across {len(classes)} classes")
print(f"'High SNR' means SNR >= +{args.high_snr:.0f} dB\n")

models = ["vgg", "resnet"] if args.model == "both" else [args.model]

# A few cut-offs so the whole clean plateau is visible, worst-to-best.
cutoffs = [-20, 0, 6, args.high_snr]
cutoffs = sorted(set(cutoffs))
top_level = snr_test.max()

rows = []
for name in models:
    model = keras.models.load_model(MODEL_DIR / f"{name}.keras")
    probs = model.predict(X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    correct = (y_pred == y_true)

    overall = correct.mean()
    high = correct[snr_test >= args.high_snr].mean()
    top = correct[snr_test == top_level].mean()
    rows.append((name, overall, high, top, correct))

# ----------------------------------------------------------------------------
# 2. The headline table: all-SNR average vs high-SNR (the paper's number)
# ----------------------------------------------------------------------------
print("The comparison that matters")
print("-" * 72)
print(f"{'model':>8} | {'all-SNR avg':>12} | {'high-SNR (>=' + str(int(args.high_snr)) + 'dB)':>16} "
      f"| {'cleanest (' + str(int(top_level)) + 'dB)':>14}")
print("-" * 72)
for name, overall, high, top, _ in rows:
    print(f"{name.upper():>8} | {overall:>11.1%} | {high:>16.1%} | {top:>14.1%}")
print()
print(f"  Paper (24 classes, high SNR): VGG ~98.3%, ResNet ~99.8%.")
print(f"  The all-SNR average is dragged down by hopeless sub-0 dB signals;")
print(f"  on clean signals our small CPU models are far closer to the paper.\n")

# ----------------------------------------------------------------------------
# 3. Accuracy at each cut-off, so you can watch it climb up the clean plateau
# ----------------------------------------------------------------------------
print("Accuracy at rising SNR floors  (answer only signals at least this clean)")
print("-" * 72)
header = "  SNR >= | " + " | ".join(f"{name.upper():>8}" for name, *_ in rows)
print(header)
print("-" * 72)
for c in cutoffs:
    cells = []
    for _, _, _, _, correct in rows:
        mask = snr_test >= c
        cells.append(f"{correct[mask].mean():>8.1%}")
    print(f" {c:>5.0f}dB | " + " | ".join(cells))
print()
print("Done.")
