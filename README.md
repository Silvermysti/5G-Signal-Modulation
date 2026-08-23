# Radio modulation classification with a 1D CNN

Given a raw 1024-sample I/Q radio snippet, which modulation scheme produced it?

This is a scaled-down replication of the VGG-style convolutional network from
O'Shea, Roy & Clancy, *[Over the Air Deep Learning Based Radio Signal
Classification](https://arxiv.org/abs/1712.04578)* (IEEE J-STSP, 2018), built as a
hands-on way to learn deep learning. The paper classifies 24 modulations using
2.55M examples on a V100 GPU; this trains on **10 modulations** and **80,000
examples** in about fifteen minutes on a CPU laptop. (It began as an easier
**3-class** warm-up — OOK/QPSK/FM at 81.1% — kept below as context.)

| | |
|---|---|
| **Test accuracy (10 classes)** | **66.3%** on 16,000 held-out examples (chance = 10%) |
| **With a "refuse to answer" gate** | **90% accurate while still answering 68%** of signals |
| **Trustworthy anchors** | FM, AM, OOK, 16QAM, 64QAM all score **>96% precision** |
| **Warm-up (3 classes)** | **81.1%** — 99.4% on clean signals, ~42% on the noisiest third |
| **Model** | VGG-style 1D CNN, 159,818 parameters |
| **Hardware** | 12 CPU cores, no GPU |

The 66% average hides *structure*. The network is near-perfect on physically
distinct modulations and vague only on look-alike families — and when noise
destroys a signal it funnels the junk into one "garbage-can" class (8PSK) rather
than spreading the error around. Reading that pattern out of the confusion matrix
is the most interesting part of the project — see the
[full report](report/README.md).

![Confusion matrix](results/confusion_matrix_vgg.png)

## The ten classes

Chosen to span the major modulation families, with two deliberately *confusable*
"ladders" so the confusion matrix shows real structure, plus distinct anchors that
stay easy to tell apart. QPSK / 16 / 64 / 256QAM are the workhorses of real 5G
data channels.

| Class | Family | Role |
|---|---|---|
| `OOK` | amplitude (on/off) | easy anchor |
| `BPSK`, `QPSK`, `8PSK` | phase-shift keying | **PSK ladder** (look alike) |
| `16QAM`, `64QAM`, `256QAM` | quadrature-amplitude | **QAM ladder** (look alike) |
| `FM` | analog frequency | easy anchor |
| `GMSK` | GSM / legacy cellular | — |
| `AM-DSB-WC` | analog amplitude | easy anchor |

Following the paper's central thesis, the network sees **raw I/Q samples only** —
no Fourier transform, no hand-crafted features. It learns its own.

## Knowing when not to answer

Forcing a guess on a signal that noise has already destroyed is a mistake. The
model reports how confident it is, so `scripts/gate.py` lets it **abstain** when
that confidence is low — "too noisy, I won't answer." Tuned to stay 90% accurate
*when it answers*, it still responds to **68% of signals** and quietly drops the
rest (vs 66% if forced to guess on everything).

That the gate really is filtering by noise — not luck — is provable: the dataset's
ground-truth SNR (signal-to-noise ratio) shows the answered signals have a median
of **+14 dB** and the abstained ones **−12 dB**. Plotting accuracy against true SNR
reproduces the paper's classic S-curve.

![Accuracy vs SNR](results/accuracy_vs_snr_vgg.png)

## Architecture

```
Input  1024 x 2  (I/Q)
7 x [ Conv1D(64, kernel 3, ReLU) -> BatchNorm -> MaxPool /2 ]   1024 -> 512 -> ... -> 8
Flatten                                                          512
Dense(128, SELU) -> AlphaDropout
Dense(128, SELU) -> AlphaDropout
Dense(10, Softmax)
```

Layer-for-layer the paper's Table III, with the output head narrowed from 24
classes to 10. Trained with Adam and cross-entropy, stopping when validation loss
stops improving — the paper's stated recipe. Only the final layer's width changes
with the number of classes; the script sizes it automatically.

## Repository layout

```
scripts/
  data_prep.py     memory-maps the 19.5 GB dataset, pulls 8,000 examples per
                   class, normalizes to unit variance, 80/20 stratified split
  model.py         the VGG CNN (run directly to print the architecture)
  resnet_model.py  the ResNet variant with skip connections
  train.py         training with early stopping + best-model checkpointing
  evaluate.py      accuracy, precision/recall, confusion matrix, confidence check
  gate.py          refuse-to-answer gate: risk-coverage + accuracy-vs-SNR curves
  high_snr.py      scores VGG and ResNet on the clean, high-SNR slice (paper-style)
prepared/          the sampled + split arrays produced by data_prep.py
                   (X_train/X_test/y_train/y_test.npy, snr_*.npy, classes.txt)
models/            trained weights — vgg.keras, resnet.keras
report/            write-ups, one folder per phase (interactive HTML + markdown)
  README.md          index linking both phase reports
  warmup-3class/     Phase 1 — the 3-class warm-up (81.1%)
  scaleup-10class/   Phase 2 — 10 classes + refuse-to-answer gate
results/           generated figures
Data/              the raw RadioML dataset (gitignored — see Setup)
```

## Setup

Requires Python 3.12 and the DeepSig RadioML 2018.01A dataset.

```bash
python3 -m venv .venv
.venv/bin/pip install tensorflow scikit-learn matplotlib
```

Place the dataset in `Data/` (gitignored — it is ~19.5 GB and non-commercially
licensed):

```
Data/
  signals.npy        (2555904, 1024, 2) float32   raw I/Q
  labels.npy         (2555904, 24)      float32   one-hot
  classes.txt        the 24 class names, in label-column order
```

`signals.npy` will not fit in RAM. `data_prep.py` memory-maps it and reads only
the rows it needs.

## Running

```bash
.venv/bin/python scripts/data_prep.py   # ~20 seconds (pulls 80,000 examples)
.venv/bin/python scripts/train.py       # ~15 minutes on CPU (10 classes)
.venv/bin/python scripts/evaluate.py    # metrics + confusion matrix
.venv/bin/python scripts/gate.py        # refuse-to-answer gate + SNR curves
```

Add `--model resnet` to `train.py`/`evaluate.py`/`gate.py` to run the residual
network instead. The class list and per-class count live at the top of
`data_prep.py`; `gate.py` takes `--target-accuracy` to tune how strict it is.

## Architectures built

- **VGG CNN** — 66.3% on 10 classes (81.1% on the easier 3-class warm-up)
- **ResNet** — skip-connection variant; **67.5% on the same 10 classes** (and 82.1%
  vs 81.1% on the 3-class task). Only ~1% better, because the bottleneck is data
  quality (noise), not the model — both hit the same ceiling.

Scored the way the paper reports — on **clean, high-SNR signals only** (≥ +18 dB) —
both models reach **~94%** (`scripts/high_snr.py`). Our headline 66–67% looks far
below the paper's 98.3% / 99.8% only because it averages in the hopeless sub-0 dB
signals the paper excludes from its peak number. The lesson we reproduced:
**architecture matters less than signal quality.** A cleverer network cannot recover
information that noise destroyed before capture — on clean signals VGG and ResNet tie.

## Not built (yet)

The paper covers three methods; this repo implements the two deep-learning ones:

- **XGBoost baseline** on higher-order moments — the classical-features comparison.
- **Over-the-air capture and transfer learning** — requires software-defined radio
  hardware.

## Credits

Dataset: [DeepSig RadioML 2018.01A](https://www.deepsig.ai/datasets), licensed
CC BY-NC-SA 4.0 — non-commercial use, attribution required.

Paper: T. J. O'Shea, T. Roy, and T. C. Clancy, "Over the Air Deep Learning Based
Radio Signal Classification," *IEEE Journal of Selected Topics in Signal
Processing*, vol. 12, no. 1, pp. 168–179, 2018.
