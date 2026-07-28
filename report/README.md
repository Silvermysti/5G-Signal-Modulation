# Project report — radio modulation CNN

A scaled-down replication of the VGG-style CNN from O'Shea, Roy & Clancy,
*Over the Air Deep Learning Based Radio Signal Classification* (IEEE J-STSP, 2018),
built to run on a CPU laptop. The project ran in two stages:

1. **Warm-up — 3 easy classes** (OOK / QPSK / FM): get an encouraging first result
   and understand *why* accuracy behaves the way it does. → **81.1%**
2. **Scale-up — 10 classes** spanning the major modulation families, with two
   deliberately confusable "ladders". → **66.3%** (chance = 10%)

**Full report:** [`report.html`](report.html) — open it in a browser (GitHub shows
HTML as source, so download it or use a local preview). It has the interactive
charts; this page is the summary.

## Headline results

| Experiment | Classes | Test accuracy | Chance | Examples | Train time |
|---|---|---|---|---|---|
| Warm-up | 3 | **81.1%** | 33% | 15,000 | ~5 min |
| Scale-up | 10 | **66.3%** | 10% | 80,000 | ~15 min |

Both on a VGG-style 1D CNN (~160k parameters), 12 CPU cores, no GPU. Only the
final layer's width changes with the number of classes.

---

## Experiment 2 — scaling to 10 classes

### The classes

Chosen to span the major families, with two confusable ladders (signals that look
more alike as you climb) plus distinct anchors. QPSK / 16 / 64 / 256QAM are the
workhorses of real 5G data channels.

| Class | Family | Role |
|---|---|---|
| `OOK` | amplitude on/off | easy anchor |
| `BPSK`, `QPSK`, `8PSK` | phase-shift keying | **PSK ladder** |
| `16QAM`, `64QAM`, `256QAM` | quadrature-amplitude | **QAM ladder** |
| `FM` | analog frequency | easy anchor |
| `GMSK` | GSM / legacy cellular | — |
| `AM-DSB-WC` | analog amplitude | easy anchor |

### Confusion matrix

![VGG 10-class confusion matrix](../results/confusion_matrix_vgg.png)

The 66% average hides a clear *structure* — three findings worth reading off the grid:

**1. The "8PSK garbage can."** 8PSK has poor precision (0.335) but high recall
(0.796): almost every other class dumps a slice of its errors into the 8PSK column
(BPSK→8PSK 384, 16QAM→8PSK 373, GMSK→8PSK 328, OOK→8PSK 306, FM→8PSK 284…).
**2,532 of the 5,390 total errors (47%) are "guessed 8PSK, was something else."**
When noise destroys a signal, the model funnels the unclassifiable junk into one
class — the same behaviour the OOK↔FM pair showed in the 3-class run, now
concentrated in a single "when unsure, say 8PSK" sink.

**2. QPSK ⇄ 256QAM.** 256QAM was called QPSK 581 times and QPSK called 256QAM 341
times — 922 errors on one pair. Notably the confusion crosses *families* (PSK vs
QAM); the QAMs barely confuse each other. A tidy "the ladders blur internally"
prediction did **not** hold — you only find this by running it.

**3. The anchors are rock-solid.** When the model commits to a distinct modulation
it is almost never wrong:

| Class | Precision | Recall | F1 | | Class | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| FM | **1.000** | 0.701 | 0.824 | | 64QAM | 0.970 | 0.795 | 0.874 |
| AM-DSB-WC | 0.997 | 0.733 | 0.844 | | 256QAM | 0.683 | 0.461 | 0.551 |
| 16QAM | 0.990 | 0.579 | 0.731 | | GMSK | 0.576 | 0.664 | 0.617 |
| OOK | 0.967 | 0.649 | 0.777 | | BPSK | 0.598 | 0.652 | 0.624 |
| 8PSK | 0.335 | 0.796 | 0.471 | | QPSK | 0.504 | 0.601 | 0.548 |

The model fails *gracefully*: near-perfect on distinct signals, vague only on
look-alikes. It also **knows when it is unsure** — average confidence 0.904 when
correct vs 0.305 when wrong, with only 2.9% of errors made confidently. That
calibration is exactly what a "refuse to answer when too noisy" gate would need.

---

## Experiment 1 — the 3-class warm-up

Three physically distinct modulations (OOK / QPSK / FM), 5,000 examples each.

| Metric | Value |
|---|---|
| Test accuracy (3,000 held-out) | **81.1%** |
| Accuracy on the cleaner 60% of signals | **99.4%** |
| Accuracy on the noisiest 30% | ~42% (chance = 33%) |

That flat 81% is an average of two regimes: near-perfect where the signal is clean,
near-chance where noise destroyed the modulation before capture. That is an
information limit, not a model weakness — and it reproduces the S-curve the paper
reports.

### Recovering accuracy-vs-SNR without SNR labels

The paper reports accuracy as a curve against signal-to-noise ratio, but our three
data files don't store per-example SNR. We estimated it blind from the signal alone
using two independent statistics — **spectral flatness** (noise spreads energy
evenly across frequencies) and **lag-1 autocorrelation** (noise samples are
uncorrelated). The two agree at **r = −0.996**.

Bucketing the 3-class test set by that estimate, worst noise first:

| Bucket | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 37.7% | 42.0% | 46.7% | 88.3% | 100% | 99.7% | 98.7% | 99.3% | 98.7% | 99.7% |

Sanity check: `0.6 × 99.4% + 0.1 × 88.3% + 0.3 × 42.1% = 81.1%` — the overall score.

### VGG vs ResNet (3-class)

We also built the paper's ResNet (skip connections, [`../results/confusion_matrix_resnet.png`](../results/confusion_matrix_resnet.png)).
It reached **82.1%** vs VGG's 81.1% — only **+1%**, because the bottleneck is data
quality (noise), not architecture. Skip connections make training easier by letting
each layer learn a small adjustment instead of a full transformation, but they
cannot recover information noise already destroyed. Both models hit the same ceiling.

---

## Against the paper

| Paper element | Status |
|---|---|
| Raw I/Q input, no expert features | replicated |
| VGG CNN layout (Table III) | replicated layer-for-layer (output head 3 or 10 wide) |
| ResNet layout (Table IV) | replicated with functional API |
| Training recipe (Adam, cross-entropy, early stop) | replicated |
| Accuracy-as-a-curve vs SNR | approximated via blind noise proxies |
| High-SNR accuracy ~98.3% (VGG), 99.8% (ResNet) | matched shape (99.4% at high SNR on 3-class) |
| Multi-class confusion structure | reproduced (10-class ladders + garbage-can sink) |
| XGBoost baseline | studied only, not built |
| Over-the-air testing, transfer learning | out of scope (needs SDR hardware) |

## Reproducing

```bash
.venv/bin/python scripts/data_prep.py   # pull + normalize 80k examples (~20s)
.venv/bin/python scripts/train.py       # train VGG, ~15 min on CPU (10 classes)
.venv/bin/python scripts/evaluate.py    # metrics + confusion matrix
```

The class list and per-class count live at the top of `scripts/data_prep.py`; add
`--model resnet` to train/evaluate to use the residual network instead.

Dataset: DeepSig RadioML 2018.01A, CC BY-NC-SA 4.0 (non-commercial, attribution
required). The dataset and trained model are gitignored — see the repo root
`CLAUDE.md` for the expected `Data/` layout.
