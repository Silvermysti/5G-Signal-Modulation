# Project phases

A replication of the VGG-style CNN (and its ResNet variant) from O'Shea, Roy & Clancy,
*Over the Air Deep Learning Based Radio Signal Classification* (IEEE J-STSP, 2018),
built to learn deep learning hands-on — starting on a CPU laptop, finishing on a free
GPU. The work ran in three phases, and each one is preserved here in full.

| Phase | Scope | Headline | Report |
|---|---|---|---|
| **1 — Warm-up** | 3 easy classes (OOK / QPSK / FM) | **81.1%**; found the two-regime (clean vs noise) split | [`1-warmup-3class/`](1-warmup-3class/README.md) · [html](1-warmup-3class/report.html) |
| **2 — Scale-up** | 10 classes + refuse-to-answer gate | **66.3%** → **90% while answering 68%** gated | [`2-scaleup-10class/`](2-scaleup-10class/README.md) · [html](2-scaleup-10class/report.html) |
| **3 — Full replication** | all 24 classes, on a Colab GPU | VGG **39.5%** / ResNet **48.7%** — architecture finally matters | [`3-full-24class/`](3-full-24class/README.md) · [html](3-full-24class/report.html) |

## What's in each phase folder

```
N-phase-name/
  scripts/      the code EXACTLY as it was when that phase's numbers were produced
  results/      the figures that run generated
  README.md     markdown write-up
  report.html   standalone interactive report (open in a browser)
```

The `scripts/` folders are **frozen snapshots pulled from git history**, not copies of
the current code — which is the point. Phase 1 has five scripts; Phase 2 has seven
(the refuse-to-answer gate and high-SNR scoring were written during it); Phase 3 has
eight. The toolset visibly grows as the problem got harder. Each phase's README names
the exact commit it was taken from.

Current, actively-maintained code lives in the repo-root [`scripts/`](../scripts/) —
that's where later fixes (the memory and validation-split bugs) landed, and where any
further work continues.

## Reading order

**Start with Phase 1** for the setup and the core "accuracy is limited by noise, not
the model" finding, then **Phase 2** for scaling to more classes and turning the
model's own confidence into an abstain-when-unsure feature — validated against
ground-truth SNR — then **Phase 3** for the full 24-class replication, where that
"architecture doesn't matter much" lesson from Phases 1–2 breaks down.

Dataset: DeepSig RadioML 2018.01A, CC BY-NC-SA 4.0 (non-commercial, attribution
required). See the repo root `README.md` for setup and data layout.
