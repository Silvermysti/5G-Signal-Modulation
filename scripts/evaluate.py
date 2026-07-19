"""
Step 5 — EVALUATION
===================

Training gave us ONE number (81% accuracy). That tells us how often the model is
right, but not WHERE it goes wrong. This script digs into that.

What it produces:
  1. Overall accuracy on the untouched test set.
  2. A per-class report (precision / recall / f1 -- explained below).
  3. A CONFUSION MATRIX: a 3x3 grid showing, for each TRUE modulation, what the
     model actually GUESSED. The diagonal = correct. Everything off-diagonal
     = a mistake, and the pattern of mistakes is the interesting part.
  4. A heatmap image of that matrix saved to results/.
  5. A look at how CONFIDENT the model was when it was right vs. wrong.

Vocabulary:
  * precision = "when the model SAYS 'QPSK', how often is it actually QPSK?"
                (measures false alarms)
  * recall    = "of all the REAL QPSK signals, how many did it catch?"
                (measures misses)
  * f1        = a single score blending precision and recall.

Note on SNR: the paper reports accuracy as a CURVE vs signal-to-noise ratio,
because noise is the dominant factor. We can't reproduce that curve -- the
per-example SNR is not stored in our three data files (it lives in the original
RadioML HDF5). So our numbers mix clean and hopelessly-noisy snippets together.
That is why we score ~81% and not the paper's ~98% (which is their HIGH-SNR peak).

Run it with:   .venv/bin/python scripts/evaluate.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hush TF startup noise

from pathlib import Path

import numpy as np
import keras
import matplotlib
matplotlib.use("Agg")  # draw to a file, not a window (we have no desktop here)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / "prepared"
MODEL_PATH = HERE.parent / "models" / "vgg_3mod.keras"
RESULTS_DIR = HERE.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Load the trained model and the test set we never trained on
# ----------------------------------------------------------------------------
model = keras.models.load_model(MODEL_PATH)
X_test = np.load(PREP_DIR / "X_test.npy")
y_true = np.load(PREP_DIR / "y_test.npy")
classes = (PREP_DIR / "classes.txt").read_text().split()

print(f"Loaded model from: {MODEL_PATH}")
print(f"Test set: {X_test.shape[0]} examples across {len(classes)} classes {classes}\n")

# ----------------------------------------------------------------------------
# 2. Predict
# ----------------------------------------------------------------------------
# predict() returns, for each snippet, 3 probabilities that add up to 1
# (e.g. [0.02, 0.95, 0.03] = "95% sure it's class 1").
probs = model.predict(X_test, batch_size=256, verbose=0)
y_pred = np.argmax(probs, axis=1)       # the class with the highest probability
confidence = np.max(probs, axis=1)      # how sure the model was about that pick

accuracy = np.mean(y_pred == y_true)
print(f"Overall test accuracy: {accuracy:.3f}  ({np.sum(y_pred == y_true)}/{len(y_true)} correct)\n")

# ----------------------------------------------------------------------------
# 3. Per-class report
# ----------------------------------------------------------------------------
print("Per-class breakdown")
print("-" * 55)
print(classification_report(y_true, y_pred, target_names=classes, digits=3))

# ----------------------------------------------------------------------------
# 4. Confusion matrix
#    Rows    = what the signal TRULY was.
#    Columns = what the model GUESSED.
#    So cell [FM, QPSK] = "real FM signals that were mistaken for QPSK".
# ----------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100   # each row as % of that true class

header = "true \\ pred |" + "".join(f"{c:>8s}" for c in classes) + "   |  recall"
print("Confusion matrix (counts)")
print("-" * len(header))
print(header)
print("-" * len(header))
for i, name in enumerate(classes):
    row = "".join(f"{cm[i, j]:8d}" for j in range(len(classes)))
    print(f"{name:12s}|{row}   |  {cm_pct[i, i]:5.1f}%")
print()

# ----------------------------------------------------------------------------
# 5. Draw the heatmap
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

ax.set_xticks(range(len(classes)), classes)
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("Model's guess")
ax.set_ylabel("True modulation")
ax.set_title(f"Confusion matrix — overall accuracy {accuracy:.1%}")

# Write the number inside each square (white text on the dark squares).
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n({cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_pct[i, j] > 50 else "black", fontsize=9)

fig.colorbar(im, ax=ax, label="% of that true class")
fig.tight_layout()
out_png = RESULTS_DIR / "confusion_matrix.png"
fig.savefig(out_png, dpi=150)
print(f"Saved heatmap to: {out_png}\n")

# ----------------------------------------------------------------------------
# 6. Was the model confident when it was right -- and unsure when it was wrong?
#    A well-behaved model should be. If it is confidently WRONG a lot, that's a
#    warning sign.
# ----------------------------------------------------------------------------
correct_mask = y_pred == y_true
print("Confidence check")
print("-" * 55)
print(f"  Average confidence when CORRECT: {confidence[correct_mask].mean():.3f}")
print(f"  Average confidence when WRONG:   {confidence[~correct_mask].mean():.3f}")
overconfident = np.sum((~correct_mask) & (confidence > 0.9))
print(f"  Confidently wrong (>90% sure but mistaken): {overconfident} "
      f"of {np.sum(~correct_mask)} errors")
print("\nDone.")
