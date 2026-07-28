# Project reports

A scaled-down replication of the VGG-style CNN from O'Shea, Roy & Clancy,
*Over the Air Deep Learning Based Radio Signal Classification* (IEEE J-STSP, 2018),
built to run on a CPU laptop and to learn deep learning hands-on. The work ran in
two phases — each has its own report, so the folder reflects the growing scope.

| Phase | Scope | Headline | Report |
|---|---|---|---|
| **1 — Warm-up** | 3 easy classes (OOK / QPSK / FM) | **81.1%**; found the two-regime (clean vs noise) split | [`warmup-3class/`](warmup-3class/README.md) · [html](warmup-3class/report.html) |
| **2 — Scale-up** | 10 classes + refuse-to-answer gate | **66.3%** → **90% while answering 68%** gated | [`scaleup-10class/`](scaleup-10class/README.md) · [html](scaleup-10class/report.html) |

Each `report.html` is a standalone interactive report (open in a browser — GitHub
shows HTML as source, so download it or use a local preview); each `README.md` is the
markdown summary. Generated figures live in the repo's [`results/`](../results/) folder.

**Start with Phase 1** for the setup and the core "accuracy is limited by noise, not
the model" finding, then **Phase 2** for scaling to more classes and turning the
model's own confidence into an abstain-when-unsure feature — validated against
ground-truth SNR.

Dataset: DeepSig RadioML 2018.01A, CC BY-NC-SA 4.0 (non-commercial, attribution
required). See the repo root `README.md` and `CLAUDE.md` for setup and data layout.
