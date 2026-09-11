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

MEMORY NOTE: passing a plain numpy array to model.fit()/evaluate() makes Keras
embed a SECOND full copy in memory (via Dataset.from_tensor_slices) on top of
the np.load() copy -- two 5.86 GB copies of X_train alive at once is why a
12.7 GB Colab VM ran out of RAM even though the file itself is smaller than
that. Fixed by loading with mmap_mode="r" (stays on disk, touched lazily) and
feeding Keras small batches through a PyDataset instead of a raw array, so at
most one batch (a few MB) is ever materialized.
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
# 1. Load the prepared data (memory-mapped: stays on disk until a batch reads it)
# ----------------------------------------------------------------------------
X_train_mm = np.load(PREP_DIR / "X_train.npy", mmap_mode="r")
y_train_mm = np.load(PREP_DIR / "y_train.npy", mmap_mode="r")
X_test_mm = np.load(PREP_DIR / "X_test.npy", mmap_mode="r")
y_test_mm = np.load(PREP_DIR / "y_test.npy", mmap_mode="r")

classes = (PREP_DIR / "classes.txt").read_text().split()
print("Classes:", classes)
print("Train:", X_train_mm.shape, "| Test:", X_test_mm.shape)


class NpyBatches(keras.utils.PyDataset):
    """Feeds Keras one batch at a time from a memory-mapped .npy array.

    Keeps VAL_FRACTION's holdout semantics identical to plain
    model.fit(X, y, validation_split=...): a contiguous slice of the END of
    the array, taken before any shuffling. Only the row order WITHIN the
    train portion is shuffled each epoch (shuffle=True) -- matches what
    validation_split + fit(shuffle=True) does with a raw array.

    ponytail: random-access batches read scattered rows from disk each epoch,
    which is slower than sequential I/O. Fine at this dataset size; if this
    ever becomes the bottleneck, sort each batch's indices before reading
    (mmap arrays reward ascending access) or convert to a shuffled TFRecord.
    """

    def __init__(self, X_mm, y_mm, indices, batch_size, shuffle, **kwargs):
        super().__init__(**kwargs)
        self.X_mm, self.y_mm = X_mm, y_mm
        self.indices = np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, i):
        batch_idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        return (
            np.asarray(self.X_mm[batch_idx], dtype="float32"),
            np.asarray(self.y_mm[batch_idx]),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


rng = np.random.default_rng(SEED)
n_train_total = X_train_mm.shape[0]
n_val = int(n_train_total * VAL_FRACTION)
# Same split point validation_split uses: last VAL_FRACTION untouched, rest shuffled.
train_idx = rng.permutation(n_train_total - n_val)
val_idx = np.arange(n_train_total - n_val, n_train_total)

# Taking a contiguous tail slice as validation only works if the saved data is
# already shuffled. If it happens to be class-sorted, that slice silently becomes
# "just the last few classes" and the model early-stops on a meaningless signal --
# a bug that costs a full training run to notice. Cheap to check, so check.
val_classes = np.unique(np.asarray(y_train_mm[val_idx]))
if len(val_classes) < len(classes):
    raise SystemExit(
        f"Validation slice covers only {len(val_classes)} of {len(classes)} classes "
        f"({val_classes.tolist()}).\nThe prepared data looks class-sorted rather than "
        f"shuffled -- re-run scripts/data_prep.py to regenerate it."
    )

train_ds = NpyBatches(X_train_mm, y_train_mm, train_idx, BATCH_SIZE, shuffle=True)
val_ds = NpyBatches(X_train_mm, y_train_mm, val_idx, BATCH_SIZE, shuffle=False)
test_ds = NpyBatches(X_test_mm, y_test_mm, np.arange(X_test_mm.shape[0]), BATCH_SIZE, shuffle=False)

# ----------------------------------------------------------------------------
# 2. Build the model (blank -- either the VGG from Step 3 or the ResNet)
# ----------------------------------------------------------------------------
print(f"Architecture: {args.model}")
model = BUILDERS[args.model](input_shape=X_train_mm.shape[1:], n_classes=len(classes))
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
    train_ds,
    validation_data=val_ds,          # same 20%-of-TRAIN holdout as before
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,                       # one tidy line per epoch
)

# Make sure the final (best) model is saved even if EarlyStopping didn't trigger.
model.save(MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")

# A quick, honest score on the untouched TEST set (full grading is Step 5).
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Held-out TEST accuracy: {test_acc:.3f}  (loss {test_loss:.3f})")
print("Done. Next step: evaluate (confusion matrix + accuracy-vs-SNR).")
