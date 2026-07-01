# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A radio **modulation classification** project built around the DeepSig **RadioML 2018.01A**
dataset. The task: given a 1024-sample raw I/Q radio snippet, predict which of 24 modulation
schemes produced it. There is **no application code yet** — currently the repo holds the
dataset, a pre-trained model, and the source research paper. New work means writing the
training/evaluation code from scratch (the user has chosen to learn deep learning by building
this).

Reference paper (in repo root): *Over the Air Deep Learning Based Radio Signal
Classification* — O'Shea, Roy, Clancy (2017/2018). This is the paper the dataset and the
pre-trained model come from; treat its model layouts as the spec.

## Repository layout

```
Over the Air Deep Learning Based Radio Signal Classification.pdf   the spec/paper
Data/
  signals.npy        ~19.5 GB  float32  shape (2555904, 1024, 2)   raw I/Q, 2 channels = I and Q
  labels.npy         ~235 MB   float32  shape (2555904, 24)        one-hot over the 24 classes
  classes.txt                  Python-literal list of the 24 class names (order matches labels)
  model_full_SNR.h5  3.6 MB    pre-trained Keras model (the paper's VGG-style 1D CNN)
  LICENSE.TXT                  CC BY-NC-SA 4.0 (DeepSig) — non-commercial, attribution required
```

## Critical facts about the data

- **`signals.npy` is ~19.5 GB and will NOT fit in RAM.** Always load with
  `np.load(path, mmap_mode='r')` (memory-map) and slice/batch from it. Never load it fully.
- Shape is `(N, 1024, 2)` — N examples, 1024 time-steps, 2 = I/Q. Keras Conv1D expects
  `(batch, 1024, 2)` directly; no transpose needed for that ordering.
- Labels are **one-hot** (not integer indices). Use categorical cross-entropy; convert with
  `np.argmax(labels, axis=1)` when an integer label is needed.
- `classes.txt` is not pure JSON — it's a Python assignment (`classes = [...]`). Parse with
  `ast.literal_eval` on the bracketed part, or read manually. Class order is significant and
  matches the label columns.
- **SNR (signal-to-noise ratio) is the dominant variable** in this dataset, but the SNR value
  for each example is **not stored in these three files** — the standard RadioML 2018.01A
  distribution carries it in the original HDF5. Accuracy must be reported *as a curve vs SNR*,
  not as a single number, per the paper. If SNR-stratified evaluation is needed, confirm where
  the per-example SNR lives before writing that code.

## Model architectures (from the paper, the spec to match)

VGG-style 1D CNN (`model_full_SNR.h5` is this; ~257k params):
```
Input (2 x 1024)
7 x [ Conv1D(64, kernel 3) -> MaxPool /2 ]   length 1024 -> 512 -> ... -> 8
Flatten
Dense(128, SELU) -> Dense(128, SELU) -> Dense(24, Softmax)
```
Conv layers use ReLU + BatchNorm; dense layers use SELU + AlphaDropout in the paper.

ResNet variant (~236k params, the paper's best model): 6 residual stacks of small Conv1D
units with skip connections, then 3 dense layers (SELU/SELU/Softmax). Build only if asked.

Training recipe from the paper: Adam optimizer, categorical cross-entropy, inputs normalized
to unit variance.

## Current work: 3-modulation laptop build

The user is recreating the paper's **VGG-style CNN as a learning exercise**, scoped down to run
on a CPU laptop. Agreed parameters:

- **Framework: TensorFlow / Keras** (finalized — chosen to match the existing `.h5` and for
  beginner-friendliness; PyTorch was considered and deferred).
- **3 classes only: `OOK`, `QPSK`, `FM`** (picked to be easy to tell apart for an encouraging
  first result). A harder follow-up trio discussed: `BPSK`, `QPSK`, `8PSK`.
- **~5,000 examples per class** (~15k total) — fits in RAM, trains on CPU in minutes.
- Data prep plan: load `labels.npy` fully (235 MB is fine), find row indices for the 3 chosen
  classes via `np.argmax`, subsample, then gather those rows from the `signals.npy` memmap
  (each example is a contiguous ~8 KB block, so scattered fancy-indexing is fast enough).
- **Reminder:** per-example SNR is absent from these files, so the model trains across all noise
  levels mixed together (including unclassifiable low-SNR snippets). Expect ~70–90% accuracy on
  the easy trio, NOT the paper's ~98% peak. This is expected, not a bug.

Build the work as separate small, commented scripts (one per step: data prep, model, train,
evaluate) so the user can read and run them individually.

## Environment / gotchas

- Target machine: **12 CPU cores, ~15 GB RAM, no GPU**, Python 3.12, `pip` 24 and `venv`
  available. Install TensorFlow into a project **virtualenv** (`python3 -m venv`), not system-wide.
- **No `h5py`, no TensorFlow/Keras, and no PyTorch are installed yet.** Loading
  `model_full_SNR.h5` or building any model requires installing these first (TF bundles `h5py`).
- `numpy` is available system-wide (Python 3, `/usr/lib/python3/dist-packages`).
- Not a git repository. The user's global rules require pushing to GitHub after significant
  changes and **never** adding Claude as a commit co-author — honor both if git is initialized.

## Working style for this repo

The user is a self-described beginner learning deep learning through this project. Explain
concepts in plain English, teach the "why," and prefer the simpler VGG CNN over ResNet as the
starting point. See the user's global CLAUDE.md for the full mentoring guidelines.
