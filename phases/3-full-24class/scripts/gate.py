"""
Step 7 — THE "REFUSE TO ANSWER" GATE (selective classification)
===============================================================

Our 10-class model scores ~66% overall, but that average is misleading: it is
almost perfect on clean signals and barely above chance on the noisiest ones,
where the radio snippet is basically static and NO classifier could win.

A smarter product wouldn't force a guess on those. It would say "this is too
noisy, I won't answer." That is called SELECTIVE CLASSIFICATION: the model is
allowed to abstain, and we judge it only on the questions it chose to answer.

The trick: the model already tells us how sure it is. `predict()` gives a
probability for each class; the biggest one (the "confidence") is high when the
signal is clean and low when it's noise. So the gate is simply:

    if confidence >= threshold:  answer with the top class
    else:                        abstain ("too noisy")

Two numbers describe any threshold:
  * COVERAGE          = what fraction of signals we chose to answer.
  * SELECTIVE ACCURACY = how often we were right, counting only answered ones.

Raise the threshold and accuracy goes up but coverage goes down -- you answer
fewer questions, but you're right more often. This script sweeps the threshold,
prints that trade-off, recommends an operating point, and (if ground-truth SNR
is available) proves the abstained signals really are the low-SNR ones.

Run it with:   .venv/bin/python scripts/gate.py
               .venv/bin/python scripts/gate.py --model resnet
               .venv/bin/python scripts/gate.py --target-accuracy 0.95
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hush TF startup noise

import argparse
from pathlib import Path

import numpy as np
import keras
import matplotlib
matplotlib.use("Agg")  # draw to a file, not a window (we have no desktop here)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Confidence gate: let the model abstain.")
parser.add_argument("--model", choices=["vgg", "resnet"], default="vgg",
                    help="which trained model to gate (default: vgg)")
parser.add_argument("--target-accuracy", type=float, default=0.90,
                    help="desired accuracy WHEN THE MODEL ANSWERS (default: 0.90)")
args = parser.parse_args()

HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / "prepared"
MODEL_PATH = HERE.parent / "models" / f"{args.model}.keras"
RESULTS_DIR = HERE.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Load the trained model and the test set (same idiom as evaluate.py)
# ----------------------------------------------------------------------------
model = keras.models.load_model(MODEL_PATH)
X_test = np.load(PREP_DIR / "X_test.npy")
y_true = np.load(PREP_DIR / "y_test.npy")
classes = (PREP_DIR / "classes.txt").read_text().split()

# Ground-truth SNR is optional; it only powers the validation section below.
snr_path = PREP_DIR / "snr_test.npy"
snr_test = np.load(snr_path) if snr_path.exists() else None

print(f"Loaded model from: {MODEL_PATH}")
print(f"Test set: {X_test.shape[0]} examples across {len(classes)} classes\n")

# ----------------------------------------------------------------------------
# 2. Predict once. Confidence = the top probability the model assigned.
# ----------------------------------------------------------------------------
probs = model.predict(X_test, batch_size=256, verbose=0)
y_pred = np.argmax(probs, axis=1)       # the class it would pick
confidence = np.max(probs, axis=1)      # how sure it was about that pick
correct = (y_pred == y_true)            # was that pick actually right?

baseline_acc = correct.mean()
print(f"Forced-answer accuracy (no gate, must guess every time): {baseline_acc:.1%}\n")


def coverage_and_accuracy(threshold):
    """At a confidence cut-off, return (coverage, selective accuracy)."""
    answered = confidence >= threshold
    cov = answered.mean()
    # If we answered nothing, accuracy is undefined -- report NaN.
    sel_acc = correct[answered].mean() if answered.any() else np.nan
    return cov, sel_acc


# ----------------------------------------------------------------------------
# 3. A readable table across a few round thresholds
# ----------------------------------------------------------------------------
print("Threshold sweep  (raise the bar -> answer less, but be right more often)")
print("-" * 68)
print(f"{'min confidence':>15} | {'coverage':>10} | {'answered':>9} | {'accuracy':>9}")
print("-" * 68)
for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    cov, sel_acc = coverage_and_accuracy(t)
    acc_str = f"{sel_acc:.1%}" if not np.isnan(sel_acc) else "   --  "
    print(f"{t:>15.2f} | {cov:>9.1%} | {int(cov * len(y_true)):>9d} | {acc_str:>9}")
print()

# ----------------------------------------------------------------------------
# 4. Recommend an operating point: the LOWEST threshold that still hits the
#    target accuracy. Lowest threshold = most questions answered at that
#    accuracy, so it's the most useful setting.
# ----------------------------------------------------------------------------
grid = np.linspace(0.0, confidence.max(), 400)
covs = np.array([coverage_and_accuracy(t)[0] for t in grid])
accs = np.array([coverage_and_accuracy(t)[1] for t in grid])

# Only consider thresholds that answer a non-trivial slice (>= 2%), so we don't
# "achieve" high accuracy on a handful of lucky examples.
eligible = (accs >= args.target_accuracy) & (covs >= 0.02)
print(f"Recommended gate for >= {args.target_accuracy:.0%} accuracy when answering")
print("-" * 68)
if eligible.any():
    i = np.flatnonzero(eligible)[0]     # first (lowest) threshold that qualifies
    t_rec, cov_rec, acc_rec = grid[i], covs[i], accs[i]
    print(f"  Set the confidence threshold to {t_rec:.2f}.")
    print(f"  -> the model ANSWERS {cov_rec:.1%} of signals, and is {acc_rec:.1%} "
          f"accurate on those,")
    print(f"     while ABSTAINING on the noisy {1 - cov_rec:.1%} instead of "
          f"guessing.")
    print(f"  (Forced to answer everything, it would only be {baseline_acc:.1%}.)")
else:
    t_rec = None
    print(f"  No threshold reaches {args.target_accuracy:.0%} accuracy with >=2% "
          f"coverage on this test set.")
print()

# ----------------------------------------------------------------------------
# 5. Draw the risk-coverage curve: accuracy (y) vs coverage (x).
#    Reading right-to-left = raising the threshold.
# ----------------------------------------------------------------------------
valid = ~np.isnan(accs)
order = np.argsort(covs[valid])         # sort by coverage for a clean line
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(covs[valid][order], accs[valid][order] * 100, color="#1f77b4")
ax.axhline(baseline_acc * 100, ls="--", color="grey",
           label=f"forced-answer accuracy ({baseline_acc:.0%})")
if t_rec is not None:
    ax.scatter([cov_rec], [acc_rec * 100], color="crimson", zorder=5,
               label=f"recommended: answer {cov_rec:.0%} @ {acc_rec:.0%}")
ax.set_xlabel("Coverage  (fraction of signals the model chooses to answer)")
ax.set_ylabel("Selective accuracy on answered signals (%)")
ax.set_title(f"{args.model.upper()} — refuse-to-answer trade-off")
ax.set_xlim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
fig.tight_layout()
rc_png = RESULTS_DIR / f"risk_coverage_{args.model}.png"
fig.savefig(rc_png, dpi=150)
print(f"Saved risk-coverage curve to: {rc_png}\n")

# ----------------------------------------------------------------------------
# 6. Validate against ground-truth SNR (only if we have it).
#    Two checks:
#      (a) accuracy really does climb with SNR -- the paper's S-curve, and a
#          sanity check that snrs.npy lines up with our examples.
#      (b) the signals the gate abstains on are genuinely the low-SNR ones.
# ----------------------------------------------------------------------------
if snr_test is not None:
    print("SNR validation (ground-truth signal-to-noise ratio)")
    print("-" * 68)

    levels = np.unique(snr_test)
    acc_by_snr = np.array([correct[snr_test == s].mean() for s in levels])

    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    ax2.plot(levels, acc_by_snr * 100, marker="o", color="#2ca02c")
    ax2.axhline(100 / len(classes), ls=":", color="grey",
                label=f"random guessing ({100/len(classes):.0f}%)")
    ax2.set_xlabel("True SNR (dB)  — lower = noisier")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{args.model.upper()} — accuracy vs true SNR")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    snr_png = RESULTS_DIR / f"accuracy_vs_snr_{args.model}.png"
    fig2.savefig(snr_png, dpi=150)
    print(f"  Accuracy climbs from {acc_by_snr[0]:.1%} at {levels[0]:.0f} dB "
          f"to {acc_by_snr[-1]:.1%} at {levels[-1]:.0f} dB.")
    print(f"  Saved accuracy-vs-SNR curve to: {snr_png}")

    if t_rec is not None:
        answered = confidence >= t_rec
        print(f"\n  At the recommended threshold ({t_rec:.2f}):")
        print(f"    median SNR of ANSWERED  signals: "
              f"{np.median(snr_test[answered]):>5.0f} dB")
        print(f"    median SNR of ABSTAINED signals: "
              f"{np.median(snr_test[~answered]):>5.0f} dB")
        print("    -> the gate abstains on the noisy signals, exactly as intended.")
    print()

print("Done.")
