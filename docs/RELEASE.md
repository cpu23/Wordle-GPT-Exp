# v1.0.0 — completed research release

**Status: completed.** This version freezes the experiment as a reproducible
portfolio research artifact. No further training or expansion of the benchmark
is planned as part of this project. The experimental package is preserved;
this is a presentation and inference release, not a training-system refactor.

[Release and downloadable model](https://github.com/cpu23/wordle-gpt-family/releases/tag/v1.0.0)
· [Research landing page](../README.md)
· [Runnable inference example](INFERENCE.md)
· [Model card](MODEL_CARD.md)

## Final result

On **719 known dictionary words**, five folds and three seeds, mechanics-first
soft distillation reached **95.64% ± 0.80 pp** win rate, versus **81.92% ± 0.14 pp**
for hard SFT with the same full-word decoder. The benchmark holds out source
secrets, not vocabulary. The best policy scores complete legal five-letter
words; it does not generate guesses with token-greedy decoding.

The release contains one variant-B 7.2M checkpoint, seed 0 / fold 1. Its own
held-out result is **135/144 wins (93.75%)**. Do not attribute the aggregate
five-fold benchmark to this one checkpoint.

## Changes

- Results-first README with three source-backed research figures: distillation,
  state scaling, and the same-checkpoint decoder discovery.
- Training-command catalogue moved to `docs/TRAINING.md`.
- `python -m wordle_gpt.demo --secret colon`: selected neural policy, automatic
  version-pinned model download, guesses, feedback and remaining turns.
- GitHub Release bundle includes weights, architecture, ordered tokenizer,
  dictionary and provenance/checksum manifest; training checkpoints remain
  excluded from source control.
- Frozen CPU inference environment, reproducible example and model card.
- GitHub Actions runs the existing unittest suite; MIT licence for code and
  released weights.

## Preserved research record

`docs/EXPERIMENTS.md`, `docs/ARTICLE.md`, datasets, historical run reports,
failed experiments and original training/evaluation code are retained. Earlier
results use different decoders, models or panels and are labeled separately
in the landing-page figures rather than being substituted for the final result.

Known limits are intentional: a fixed 719-word universe, synthetic teacher
states, no solver fallback, occasional repeated guesses, and no claim of
performance on the full NYT answer list or unseen words.

## Release asset

`wordle-gpt-7.2m-soft-v1.0.0.zip` — 26,567,702 bytes.

```text
SHA-256 b1d07d64f061f23c472deb58c20c4d92a49d20fccc4513002341c4f2e49aa674
```

See [INFERENCE.md](INFERENCE.md) for the exact environment and observed game
trace. Release source is pinned by the `v1.0.0` Git tag.
