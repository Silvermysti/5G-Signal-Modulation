"""
Step 4 — TRAINING
=================

This is where the blank model from `model.py` actually LEARNS.

Big picture:
  1. Load the prepared data (the 12,000 train + 3,000 test snippets from Step 2).
  2. Build the VGG-style CNN from `model.py`.
  3. "Compile" it -- tell it HOW to learn (optimizer, loss, what to measure).
  4. "Fit" it -- show it the training data over and over for a few epochs.
  5. Save the trained model so Step 5 (evaluate) can grade it.

Vocabulary (first time each appears):
  * epoch      = one full pass through all 12,000 training examples.
  * batch      = a small group of examples processed together before the model
                 nudges its weights. We use 128 at a time (128 is a common,
                 memory-friendly size).
  * optimizer  = the rule that decides how to nudge the weights. "Adam" is the
                 paper's choice: a smart, self-adjusting version of gradient
                 descent (the basic "roll downhill toward lower error" method).
  * loss       = the number the model tries to make SMALL. It measures how wrong
                 the predictions are. We use cross-entropy (standard for picking
                 one class out of several).
  * validation = a slice of the TRAIN data we hold back each epoch to watch for
                 overfitting (memorizing instead of learning). This is separate
                 from the TEST set, which we keep untouched until Step 5.

Run it with:   .venv/bin/python scripts/train.py
               .venv/bin/python scripts/train.py --model resnet
On a CPU laptop this takes a few minutes.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hush TF startup noise

import argparse
from pathlib import Path

import numpy as np
import keras

# Import both model-builders (Step 3 and Step 6, same folder).
from model import build_vgg_cnn
from resnet_model import build_resnet

# Which architectures this script knows how to train.
BUILDERS = {"vgg": build_vgg_cnn, "resnet": build_resnet}

parser = argparse.ArgumentParser(description="Train a modulation classifier.")
parser.add_argument("--model", choices=BUILDERS, default="vgg",
                    help="which architecture to train (default: vgg)")
args = parser.parse_args()

# ----------------------------------------------------------------------------
# Settings you can tweak
# ----------------------------------------------------------------------------
BATCH_SIZE = 128          # examples processed together before each weight update
EPOCHS = 20               # full passes over the training data
VAL_FRACTION = 0.2        # slice of TRAIN held back each epoch to watch overfitting
SEED = 42                 # fixed randomness => repeatable runs

HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / "prepared"
MODEL_DIR = HERE.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / f"{args.model}.keras"   # where we save the trained model

# Make the run reproducible (same starting weights + shuffling every time).
keras.utils.set_random_seed(SEED)

# ----------------------------------------------------------------------------
# 1. Load the prepared data
# ----------------------------------------------------------------------------
X_train = np.load(PREP_DIR / "X_train.npy")   # (12000, 1024, 2) float32
y_train = np.load(PREP_DIR / "y_train.npy")   # (12000,) integers 0/1/2
X_test = np.load(PREP_DIR / "X_test.npy")     # (3000, 1024, 2)  -- untouched here
y_test = np.load(PREP_DIR / "y_test.npy")     # (3000,)

classes = (PREP_DIR / "classes.txt").read_text().split()
print("Classes:", classes)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ----------------------------------------------------------------------------
# 2. Build the model (blank -- either the VGG from Step 3 or the ResNet)
# ----------------------------------------------------------------------------
print(f"Architecture: {args.model}")
model = BUILDERS[args.model](input_shape=X_train.shape[1:], n_classes=len(classes))
model.summary()

# ----------------------------------------------------------------------------
# 3. Compile -- attach the learning rule, loss, and metric
# ----------------------------------------------------------------------------
# sparse_categorical_crossentropy: the "pick one class" loss for when labels are
# plain integers (0/1/2), which is exactly how data_prep saved them (not one-hot).
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ----------------------------------------------------------------------------
# 4. Callbacks -- little helpers that run during training
# ----------------------------------------------------------------------------
callbacks = [
    # Stop early if the validation loss stops improving for 5 epochs, and roll
    # back to the best weights. Saves time and guards against overfitting.
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    # Save the best model (by validation loss) to disk as we go.
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=0
    ),
]

# ----------------------------------------------------------------------------
# 5. Fit -- the actual training loop
# ----------------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=VAL_FRACTION,   # hold back 20% of TRAIN to watch each epoch
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,                       # one tidy line per epoch
)

# Make sure the final (best) model is saved even if EarlyStopping didn't trigger.
model.save(MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")

# A quick, honest score on the untouched TEST set (full grading is Step 5).
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Held-out TEST accuracy: {test_acc:.3f}  (loss {test_loss:.3f})")
print("Done. Next step: evaluate (confusion matrix + accuracy-vs-SNR).")
