# Model Card: wordle-gpt-7.2m-soft

The selected soft-distilled Wordle policy for the completed `v1.0.0` research
release. Download the checkpoint, tokenizer and dictionary together from
[GitHub Releases](https://github.com/cpu23/wordle-gpt-family/releases/tag/v1.0.0)
as `wordle-gpt-7.2m-soft-v1.0.0.zip`. This card documents its provenance,
measured performance and limitations.

| Field | Value |
| :--- | :--- |
| Version | `1.0.0` (`v1.0.0` release tag) |
| Variant | B — soft distillation from the mechanics-pretrained 7.2M checkpoint |
| Parameters | 7,162,403 |
| Architecture | decoder-only Transformer: 4 layers, 384 hidden, 12 heads, MLP 1536, context 96, 35 tokens |
| Decoder | word-argmax: argmax over joint five-letter sequence log-probability across all 719 dictionary words |
| Training data | synthetic trajectories only; soft targets from the in-repo exhaustive classical solver |
| Dictionary | the repository's 719-word list (`resources/words.txt`) |
| Licence | MIT, © 2026 Wordle GPT contributors (repository `LICENSE`); first-party weights, no third-party pretrained weights |

## Bundle contents

`dist/wordle-gpt-7.2m-soft-v1.0.0.zip` (26,567,702 bytes,
SHA-256 `b1d07d64f061f23c472deb58c20c4d92a49d20fccc4513002341c4f2e49aa674`)
extracts to one flat directory:

| File | Bytes | SHA-256 |
| :--- | ---: | :--- |
| `model.pt` | 28,663,113 | `176d118fb9b0dc72f4ec853007f0cd1704b3e676b0e4c0d0ca07b31b06a9835f` |
| `tokenizer.json` | 548 | `5697cfdc4e8d561180a8fae56dca6fed1731f5cc608dadd0d3fe8583a2282d93` |
| `words.txt` | 4,314 | `3f4682f8bedf30744c51133af3f5dcd6436d6dcfccf5323a448f48bb0928d04d` |
| `manifest.json` | 3,340 | `e755012b69d990dfa982191e318f84e492f95d8f8a2199ff6a76035e3c0c5e4d` |

* `model.pt` contains only `model_state_dict`, `model_config`, `vocabulary_size`,
  and a format tag — no optimizer, resume, teacher, or training state.
* `tokenizer.json` stores the ordered 35-token vocabulary; the demo refuses to
  load a bundle whose tokens differ from the runtime vocabulary.
* `words.txt` is the original 719-word dictionary used for training and every
  benchmark; the demo plays and scores only whole words from this list.
* `manifest.json` records architecture, provenance, held-out scores, and the
  SHA-256 checksum of every other bundle file. The demo verifies those
  checksums on every load and rejects mismatches.

## Provenance

| Field | Value |
| :--- | :--- |
| Benchmark run | `runs/soft-distillation-cv5-word-argmax` |
| Selected checkpoint | `runs/soft-distillation-cv5-word-argmax/B/seed-0/fold-1/B-T0.25/checkpoints/best.pt` |
| Checkpoint SHA-256 | `dda34516ccc7787d9a8c9d37b94b12f97104891d021b0c3bf26ca0c70e6e662f` |
| Training step / epoch | 11,482 / epoch 2 |
| Distillation temperature | 0.25 |
| Seed / fold | 0 / 1 |
| Initialization | `runs/scaling-cv5-1m/seed-0/fold-1/7.2m/mechanics/checkpoints/best.pt` (`32866aa7…2ce71d92`) |
| Teacher dataset | `data/soft-teacher-cv1m` (manifest `9e262fc7…8042cc`) |
| Supervised states | 698,112 (95% expert distillation, 5% mechanics replay) |

The teacher is the in-repo exhaustive classical solver: it ranks every legal
guess by exact expected surviving answers, and the soft targets are
`softmax(-log(cost / min cost) / T)` over 128 sampled candidate actions per
observable state. The student is a 7.2M-parameter decoder-only Transformer
pretrained on game mechanics, then distilled at `T = 0.25`.

Checkpoints are selected on training-panel validation only: maximum
word-argmax wins, then minimum average attempts, average guesses, exhaustive
expected-survivor regret, and teacher–student KL
(`B-T0.25/best.json`). No held-out test games enter checkpoint selection.
Source-secret splits are strict: fold-1 test secrets are never training
sources, but they remain possible answers (the decoder scores the full
dictionary, as in the original expert SFT).

## Evaluation

All numbers below use the word-argmax decoder at `T = 0.25`, six guesses per
game, and the 719-word dictionary.

**Released checkpoint, its own held-out fold (seed 0, fold 1):** 144 games,
135 wins — **93.75%**, average 3.785 guesses on wins, 3.924 attempts per game,
0 invalid guesses. Source:
`B/seed-0/fold-1/held-out-evaluation.json` (also in the bundle manifest).

| Run | Win rate | Wins | Average attempts |
| :--- | ---: | ---: | ---: |
| Selected checkpoint, held-out fold 1 | 93.75% | 135 / 144 | 3.924 |
| Variant B, seed 0, all five folds | 94.71% | 681 / 719 | 3.890 |
| **Variant B aggregate, 3 seeds × 5 folds** | **95.64%** ± 0.80 pp | 687.7 / 719 | 3.949 |
| Hard SFT baseline aggregate | 81.92% ± 0.14 pp | 589.0 / 719 | 3.897 |
| Variant C (soft from hard-SFT init) aggregate | 81.69% ± 0.40 pp | 587.3 / 719 | 3.983 |

Read the rows together carefully: the headline 95.64% is the three-seed,
five-fold aggregate, and the seed-0 94.71% row uses five fold-specific
checkpoints. The released zip contains **one** checkpoint (seed 0, fold 1),
whose own held-out result is the 93.75% row. Full per-fold, per-seed results:
`runs/soft-distillation-cv5-word-argmax/benchmark-complete.json`.

## Running the demo

The demo runs on CPU with the frozen inference requirements
(`requirements-inference.txt`: Python 3.11, `torch==2.6.0+cpu`, `numpy==2.2.6`):

```bash
python -m pip install -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m wordle_gpt.demo --secret colon                                   # downloads the release asset once
python -m wordle_gpt.demo --bundle dist/wordle-gpt-7.2m-soft-v1.0.0        # offline, extracted bundle
python -m wordle_gpt.demo --bundle dist/wordle-gpt-7.2m-soft-v1.0.0.zip    # offline, bundle archive
python -m wordle_gpt.demo --interactive                                    # you supply feedback
```

The downloaded asset is cached under the platform cache directory
(`$XDG_CACHE_HOME/wordle-gpt-demo` on Linux) and verified on every load. The
demo prints each guess, its feedback, the turns left, and the number of
dictionary words still consistent with the feedback, then the terminal
outcome; exit code 0 means solved, 1 not solved, 2 input or bundle error.

Verified example from this build (`--bundle … --secret colon`):

```text
Turn 1/6: stare -> XXXXX | turns left 5 | candidate words 46
Turn 2/6: cling -> GYXYX | turns left 4 | candidate words 1
Turn 3/6: colon -> GGGGG | turns left 3 | candidate words 1
Solved 'colon' in 3 guesses.
```

Inference uses no search and no teacher: each turn is a single greedy
whole-word argmax over the 719 dictionary words. Nothing beyond the released
weights, tokenizer, and word list is required.

## Limitations

* **Fixed dictionary.** The model is trained and evaluated on the 719-word list
  in the bundle; play outside it, different word lengths, or different feedback
  conventions is out of scope. Off-dictionary secrets are rejected by the demo.
* **Greedy decoder.** Word-argmax is deterministic and can repeat a guess until
  the guess budget runs out (for example, 'straw' lost as
  `stare → stair → strap → strap → strap → strap` in the seed-0 held-out
  games). It has no exploration, no self-consistency, and no abstention.
* **Narrow training distribution.** All training states come from an in-repo
  classical solver's play; the model can inherit that teacher's blind spots and
  is expected to degrade off-distribution.
* **Cost per turn.** Scoring every dictionary word costs one batched forward
  pass per turn (fast for 719 CPU words, but scales with dictionary size).
* **No calibration.** The policy returns a single word; it exposes no
  confidence or risk estimate, and it does not model the possibility of
  dishonest feedback.
* **Research artifact.** This is a small research model for demonstration and
  benchmark reproduction, not a production or competitive solver. Not
  affiliated with, endorsed by, or derived from The New York Times; "Wordle" is
  used descriptively.

## Licence and attribution

The code, the released weights, and this bundle are covered by the repository's
MIT licence. The weights are first-party: trained from scratch on synthetic
trajectories generated in this repository, with no third-party pretrained
weights or fine-tuned derivatives. The 719-word list is the repository's own
runtime input (`resources/words.txt`), shipped unchanged in the bundle.
