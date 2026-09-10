# Banking77 calibration note (S2.5)

Method calibration only: the same T1 (TF-IDF+LogReg) / T2 (frozen embeddings+LR) code paths score ~84-88% (TF-IDF+LR) to ~94% (BERT full) / ~88% (USE 10-shot) on Banking77 in the literature, and no banking data, labels, or weights enter the Twitter heads.

## Anchors (Casanueva et al. 2020, Efficient Intent Detection with Dual Sentence Encoders)

| setting | model | accuracy |
|---|---|---|
| full (10k train) | fine-tuned BERT | ~93.7% |
| 10-shot | USE / ConveRT dual encoder | ~85.2% |
| full | TF-IDF + LR baseline | ~84-88% |

Our Twitter result for context (pseudo-labels, NOT comparable as a claim):
T1 TF-IDF+LR macro-F1 0.787 on golden-200; T2 frozen MiniLM+LR 0.650
(-> stay on T1). The Banking77 numbers calibrate the METHOD (same code),
never the weights: `grep -ri banking77 src/` is empty by test.

## Sandbox

- local_run: False
- reason: dataset not vendored; literature anchors only (no download in repro)
