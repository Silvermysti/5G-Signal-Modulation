# Project reports

A replication of the VGG-style CNN (and its ResNet variant) from O'Shea, Roy & Clancy,
*Over the Air Deep Learning Based Radio Signal Classification* (IEEE J-STSP, 2018),
built to learn deep learning hands-on — starting on a CPU laptop, finishing on a free
GPU. The work ran in three phases — each has its own report, so the folder reflects
the growing scope.

| Phase | Scope | Headline | Report |
|---|---|---|---|
| **1 — Warm-up** | 3 easy classes (OOK / QPSK / FM) | **81.1%**; found the two-regime (clean vs noise) split | [`warmup-3class/`](warmup-3class/README.md) · [html](warmup-3class/report.html) |
| **2 — Scale-up** | 10 classes + refuse-to-answer gate | **66.3%** → **90% while answering 68%** gated | [`scaleup-10class/`](scaleup-10class/README.md) · [html](scaleup-10class/report.html) |
| **3 — Full replication** | all 24 classes, on a Colab GPU | VGG **39.5%** / ResNet **48.7%** — architecture finally matters | [`full-24class/`](full-24class/README.md) · [html](full-24class/report.html) |

Each `report.html` is a standalone interactive report (open in a browser — GitHub
shows HTML as source, so download it or use a local preview); each `README.md` is the
markdown summary. Generated figures live in the repo's [`results/`](../results/) folder.

**Start with Phase 1** for the setup and the core "accuracy is limited by noise, not
the model" finding, then **Phase 2** for scaling to more classes and turning the
model's own confidence into an abstain-when-unsure feature — validated against
ground-truth SNR — then **Phase 3** for the full 24-class replication, where that
"architecture doesn't matter much" lesson from Phases 1–2 breaks down.

Dataset: DeepSig RadioML 2018.01A, CC BY-NC-SA 4.0 (non-commercial, attribution
required). See the repo root `README.md` for setup and data layout.
