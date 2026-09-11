# Full replication report — all 24 modulations (Phase 3 of 3)

The project's final stage: the complete paper replication. All **24 modulations**
DeepSig's RadioML 2018.01A ships, trained on a free **Google Colab T4 GPU** (the
laptop CPU that handled Phases 1–2 would take hours at this scale).

> **Phase 1:** the 3-class warm-up (81.1%) is in [`../warmup-3class/`](../warmup-3class/README.md).
> **Phase 2:** the 10-class run + refuse-to-answer gate is in [`../scaleup-10class/`](../scaleup-10class/README.md).

## Headline results

| Metric | VGG | ResNet |
|---|---|---|
| Test accuracy, all SNR (96,000 held-out) | 39.5% | **48.7%** |
| High-SNR accuracy (≥ +18 dB, paper-style) | 57.9% | **75.8%** |
| Classes with **zero** recall (never predicted) | **4 of 24** | 0 of 24 |

Chance is 1-in-24 (4.2%), so both models are far above guessing — but the honest
comparison to the paper's ~98.3% (VGG) / ~99.8% (ResNet) shows a real gap this time,
not just an SNR-averaging illusion. At 24 classes, **architecture stops being a minor
detail and starts being the difference between learning a class at all or not.**

## What changed at 24 classes

Phases 1 and 2 taught one clear lesson: VGG and ResNet land within ~1% of each
other, because noise — not architecture — was the ceiling. **That lesson breaks at 24
classes.** ResNet beats VGG by **9.2 points** (48.7% vs 39.5%), and the gap isn't just
quantitative — it's qualitative:

**VGG completely gives up on four classes.** `64APSK`, `128QAM`, `128APSK`, and
`AM-DSB-WC`'s sibling `AM-DSB-SC` all score **precision = 0.000, recall = 0.000** — the
model never once predicts them, for any of their 4,000 test examples. ResNet learns
all 24; its weakest class (`128APSK`) still gets 11.7% recall — far from great, but
never zero.

## The garbage-can effect, concentrated

Where do the four collapsed classes' 16,000 examples go? Overwhelmingly into three
other classes that become VGG's dumping ground:

| Sink class | Times predicted | True count | Over-prediction |
|---|---|---|---|
| `8ASK` | 14,938 | 4,000 | 3.7× |
| `AM-SSB-SC` | 11,061 | 4,000 | 2.8× |
| `16PSK` | 10,413 | 4,000 | 2.6× |

Together these three classes absorb **36,412 of 96,000 predictions (38%)** despite
being only 12.5% of the true examples. This is the same "garbage-can" behaviour found
in Phase 2 (where 8PSK alone absorbed the noise-destroyed signals) — except at 24
classes, with four genuinely hard classes instead of one noisy-signal problem, VGG's
garbage can spreads across a small cluster of look-alike classes instead of a single
sink.

**Why these four, specifically:** `64APSK`, `128QAM`, and `128APSK` are extremely
dense, high-order constellations — visually almost indistinguishable from each other
and from their neighbours even without noise. `AM-DSB-SC` differs from `AM-DSB-WC`
(an "easy anchor" in Phase 2) only in whether the carrier is suppressed — a subtle
distinction. These are the four hardest pairwise-confusable classes in the whole set,
and VGG's plain feed-forward stack couldn't learn to separate them in the available
training time. ResNet's skip connections make that optimization easier — consistent
with the original ResNet paper's finding that skip connections mainly help *training*,
not raw capacity — and here that translates directly into "learns vs. doesn't."

## Confusion matrices

**VGG** — note the four solid-zero rows/columns and the wide 8ASK/AM-SSB-SC/16PSK sink:

![VGG 24-class confusion matrix](../../results/confusion_matrix_vgg_24class.png)

**ResNet** — same four classes are still the weakest, but every one is learned:

![ResNet 24-class confusion matrix](../../results/confusion_matrix_resnet_24class.png)

## Against the paper

| Paper element | Status |
|---|---|
| All 24 classes (full replication) | done — VGG 39.5% / ResNet 48.7% all-SNR |
| High-SNR accuracy ~98.3% (VGG) / 99.8% (ResNet) | not matched — 57.9% / 75.8% (see below) |
| Multi-class confusion structure | reproduced + extended — collapsed-class sink |
| ResNet's real value | reproduced and **strengthened**: the gap widens with class count |

**Why we don't match the paper's high-SNR numbers here** (unlike Phase 2, where 10-class
high-SNR nearly matched): the paper trains on ~106,000 examples per class; we used
**20,000/class** (480,000 total, chosen for a ~25-minute Colab budget). With 24 genuinely
confusable classes — several of which need fine detail to separate — that's not enough
data to fully resolve the hardest ones. This is an honest, expected limitation, not a bug:
more data and a longer training budget would very likely close much of this gap,
following the same "architecture matters less than data quality/quantity" theme as the
rest of the project — just with *quantity* joining *noise* as a constraint at this scale.

## Next

- **More data per class** — closer to the paper's ~106k/class, to see how much of the
  gap on the four hardest classes closes.
- **A refuse-to-answer gate for 24 classes** — `gate.py` already generalizes; the
  collapsed-class problem may show up as low confidence rather than confident wrong
  answers.
- **Why these specific four** — a closer look at whether it's constellation density,
  training-order effects, or something else that makes VGG give up on exactly these
  classes.

## Reproducing

Data prep now supports the full class list directly:

```bash
.venv/bin/python scripts/data_prep.py --all-24 --per-class 20000   # ~480k examples
.venv/bin/python scripts/train.py --model vgg      # best run on a GPU (Colab notebook: colab/train_24class.ipynb)
.venv/bin/python scripts/train.py --model resnet
.venv/bin/python scripts/evaluate.py --model vgg
.venv/bin/python scripts/evaluate.py --model resnet
.venv/bin/python scripts/high_snr.py               # both models, high-SNR comparison
```

On a 12-core CPU laptop the 24-class/480k run is impractically slow; `colab/train_24class.ipynb`
runs the same scripts on a free Colab GPU in ~25 minutes total for both models.

Dataset: DeepSig RadioML 2018.01A, CC BY-NC-SA 4.0 (non-commercial, attribution
required). The dataset and trained models are gitignored — see
the repo root `README.md` (Setup section) for the expected `Data/` layout.
