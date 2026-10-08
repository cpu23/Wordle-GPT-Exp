# Training guide

This is the long-form companion to the short *Training and reproduction* section of the README. It keeps the full command catalogue — data builders, mechanics pretraining, expert SFT with replay, soft-policy distillation, anchored DPO, the GRPO variants, and evaluation — together with the architecture, checkpoint-selection, and training-token context needed to read those commands.

All commands run from the repository root. Every module below is invoked as `python -m …`; either activate a virtual environment with `requirements.txt` installed, or prefix with `uv run --with-requirements requirements.txt`. Training scripts default to `--device cuda`; CPU works for smoke-sized runs only. Inference-only users do not need this guide: the released checkpoint runs through `requirements-inference.txt` and `wordle_gpt.demo` (see [INFERENCE.md](INFERENCE.md)).

## Model configurations

Four sizes were trained. The 7.2M model is the one behind every v1.0.0 result; the other three trace the capacity and scaling story in [EXPERIMENTS.md](EXPERIMENTS.md).

| | Base | Scaled | **Released (7.2M)** | Capacity probe |
| :--- | ---: | ---: | ---: | ---: |
| Parameters | 814,627 | 3,202,083 | **7,162,403** | 12,695,587 |
| Layers | 4 | 4 | **4** | 4 |
| Embedding size | 128 | 256 | **384** | 512 |
| Attention heads | 4 | 8 | **12** | 16 |
| MLP hidden dim | 512 | 1024 | **1536** | 2048 |
| Context length | 96 | 96 | **96** | 96 |

All sizes share one tokenizer: 26 letters + 3 feedback digits (`0` gray, `1` yellow, `2` green) + 6 control tokens (`<G>` guess, `<F>` feedback, `<E>` end, `<M>` mechanics, `<S>` secret, `<P>` policy), and games are serialized as text:

```text
policy     <P> <G>could<F>22010<G>colon<E>
mechanics  <M><S>colon<G>could<F>22010<E>
```

## Training stages at a glance

1. **Mechanics pretraining** — learn exact feedback on synthetic (secret, guess) pairs; loss reaches ~5×10⁻⁴.
2. **Expert SFT with replay** — imitate the classical minimum-expected-survivors solver, with 5% mechanics replay to prevent catastrophic forgetting.
3. **Soft policy distillation** (arms B and C) — match the solver's full action distribution at temperature `T`; this produced the released model.
4. **Anchored DPO** — a research detour: standard DPO collapses gameplay, and an added SFT anchor term rescues it (see the [benchmark results](../runs/dpo-cv5-7.2m/aggregate.json)).
5. **GRPO variants** — four reward designs for on-rollout improvement; all are development history, not part of the released policy.

Selection and evaluation discipline, applied by every trainer:

- **Modes.** `development` uses fixed 575/72/72 train/validation/test secrets; `benchmark` uses all 719 secrets in five folds (each seed holds every secret out exactly once). Folds filter expert histories and mechanics examples by source secret; manifests report the exact number omitted.
- **Validation only.** Checkpoints are selected on validation data. The fixed 72 test secrets require an explicit `--split test` and are reserved for final development comparisons.
- **Criteria.** Mechanics and SFT select on validation loss; soft distillation selects on whole-word-argmax validation win rate over pinned panels; GRPO trainers keep a gameplay-ranked `best.pt` and, where applicable, a separate `best-continuation.pt`.

## Data builders

### Cross-validation modes and universal pools

```bash
python -m wordle_gpt.evaluation.cross_validation \
  --mode development --output data/wordle-development.json

# Recreate the final benchmark modes and universal pool
python -m wordle_gpt.evaluation.cross_validation \
  --mode benchmark --output data/wordle-cv5.json --model-seeds 0 1 2
python -m wordle_gpt.datasets.dataset_expert \
  --output-dir data/wordle-v2-diverse-cv --include-test
python -m wordle_gpt.datasets.dataset_mechanics_cv

# Materialize strict folds, then run 5 folds × 3 seeds × both nested sizes
python -m wordle_gpt.evaluation.benchmark_cv --prepare-only
python -m wordle_gpt.evaluation.benchmark_cv --skip-prepare
```

`dataset_expert` generated 200,000 unique observable histories (`--total-states`, default 200,000; nested prefixes at 10K/50K/100K/200K), each relabeled with the clever minimum-expected-survivors action regardless of the policy that reached it. `--include-test` admits the fixed test secrets **only** when building the universal CV pool. `dataset_mechanics_cv` builds the 100,000-example mechanics corpus used by the benchmark.

### 500K and 1M pools

```bash
python -m wordle_gpt.datasets.build_500k_pools dev
python -m wordle_gpt.datasets.build_500k_pools cv
python -m wordle_gpt.datasets.build_1m_pools
```

`build_500k_pools` reproduces the original generation stream through state 200,000, then accepts a state whose remaining-answer-set fingerprint has already appeared with probability 0.5 (novel answer sets are always accepted), so the nested 200K prefix stays field-identical while the 200K–500K tail is biased toward logical novelty — the diversity trick that made the later 1M corpus trainable. `build_1m_pools` extends this to 1,000,000 records and verifies that the first 500,000 records are field-identical to the 500K pool.

Each 1M pool exceeds GitHub's 100 MB per-file limit, so it is committed as `examples.jsonl.gz.part-0` + `examples.jsonl.gz.part-1`. Reassemble and verify against the manifest SHA-256, then re-verify every materialized fold on the exact fields the training loader reads:

```bash
python -m wordle_gpt.datasets.join_pools          # reassemble + verify examples.jsonl.gz
python -m wordle_gpt.experiments.seed_1m_runs cv  # re-verify materialized folds
```

### Consistency and v2 datasets

```bash
python -m wordle_gpt.datasets.dataset_consistency  # 148,272 balanced candidate-consistency examples
python -m wordle_gpt.datasets.dataset_v2           # masked v2 dataset with unobservable targets
```

## Mechanics pretraining and expert SFT

### Original nested baseline (814,627 parameters)

```bash
python -m wordle_gpt.training.train_nested \
  --sizes 100000 \
  --steps 10000 \
  --checkpoints 0 100 500 1900 5000 10000

python -m wordle_gpt.training.train_nested \
  --sizes 500000 \
  --steps 10000 \
  --checkpoints 0 100 500 1900 3400 5000 10000
```

`--sizes` selects nested state prefixes; `--checkpoints` are the step marks saved per size. Nested training stores its metrics in-tree, which is why the run artifacts in `runs/` carry per-checkpoint `metrics.json` files.

### v2 experiments (A / B / C / D)

```bash
python -m wordle_gpt.experiments.experiments_v2 --experiment a --patience 4 --max-epochs 100

python -m wordle_gpt.experiments.experiments_v2 --experiment b-mechanics --patience 4 --max-epochs 100

python -m wordle_gpt.experiments.experiments_v2 --experiment b-expert \
  --initial-checkpoint runs/v2-experiment-b-mechanics/checkpoints/best.pt \
  --patience 4 --max-epochs 100

# Experiment C: V1 trajectory pretraining → expert SFT
# Experiment D: mechanics → candidate consistency → expert SFT
```

Each stage stops after `--patience` consecutive validation checks without an improvement of at least `1e-4`; every strict improvement overwrites that stage's `best.pt`.

### Scaled models (3.2M / 7.2M / 12.7M)

Architecture flags are passed directly:

```bash
python -m wordle_gpt.experiments.experiments_v2 \
  --experiment b-mechanics \
  --output-dir runs/v2-3m/b-mechanics \
  --context-length 96 --embedding-size 256 \
  --num-layers 4 --num-heads 8 --mlp-size 1024
```

`python -m wordle_gpt.experiments.scaling_experiment` trains the full size ladder over the 1M-state development corpus (defaults: `--data-dir data/wordle-dev-1m`, `--runs-dir runs/scaling-dev-1m`, up to 100 epochs); `--print-parameters` reports parameter counts per size.

### Expert/mechanics replay ratios

Replay is the fix for catastrophic forgetting: pure expert SFT raises mechanics validation loss from 0.000515 to 10.392833 (≈20,000×), while interleaving mechanics batches holds it near zero.

```bash
python -m wordle_gpt.experiments.experiments_replay \
  --output-dir runs/v2-replay/e3-expert90-mechanics10 \
  --initial-checkpoint runs/v2-experiment-b-mechanics/checkpoints/best.pt \
  --expert-ratio 0.9 --mechanics-ratio 0.1 --consistency-ratio 0

python -m wordle_gpt.experiments.experiments_replay \
  --output-dir runs/v2-replay/consistency-expert80-mechanics10-consistency10 \
  --initial-checkpoint runs/v2-experiment-d-consistency/checkpoints/best.pt \
  --expert-ratio 0.8 --mechanics-ratio 0.1 --consistency-ratio 0.1
```

The three ratios must sum to 1. Expert batches are sampled with replacement from the same seed, and epoch boundaries are based on cumulative supervised expert tokens; each objective gets its own deterministic random stream, and replay batches are spread evenly through the expert updates. The final benchmark used 5% mechanics replay.

## Soft classical-policy distillation

Train against the classical solver's probability distribution over 128 legal guesses instead of a single expert move.

```bash
python -m wordle_gpt.distillation.soft_teacher \
  --output-dir data/soft-teacher-1m --workers 8 \
  --mode data/wordle-development.json

python -m wordle_gpt.distillation.train_soft_distillation
python -m wordle_gpt.distillation.compare_soft_distillation
python -m wordle_gpt.distillation.benchmark_soft_distillation
```

- **Teacher.** `soft_teacher` caches, per state, the 128 solver-ranked candidates with their expected-survivor costs and ranks (raw arrays, not baked soft targets) and writes temperature-distribution statistics for `0.25 0.50 1.00` (defaults; `--stats-only` reports without writing, `--expected-count` sanity-checks corpus size). The training script recomputes `softmax(−log(cost / min cost) / T)` from the cached costs at each run's temperature. Skip teacher construction if the verified dataset already exists.
- **Training.** `train_soft_distillation` trains each temperature as a separate configuration (`B-T0.25`, `B-T0.50`, `C-T0.25`, …), comparing mechanics-pretrained (**B**, LR 3e-4) and hard-SFT-initialized (**C**, LR 1e-5 or 3e-6) students with 5% mechanics replay (`{'expert': 0.95, 'mechanics': 0.05, 'consistency': 0.0}`). Checkpoints are selected by maximum word-argmax wins on the pinned validation panel, tie-broken by fewer attempts, lower exhaustive expected-survivor regret, and lower teacher-to-student KL. `--variants`, `--temperatures`, `--fold`, `--c-lr`, `--mechanics-checkpoint`, `--hard-checkpoint` are all explicit.
- **Evaluation.** Guesses are selected by full-word probability over all 719 dictionary words, never token-greedy decoding: on one checkpoint that distinction is 72/72 versus 2/72 wins (README finding 3). The final fixed benchmark used `T=0.25`, five folds, and three seeds.
- **Resume.** Repeat training with `--resume`. The resumable trainer saves every 100 updates (`--save-every`) and on graceful stop; legacy token-greedy run states are not compatible with the word-argmax trainer.

The final artifact in `runs/soft-distillation-cv5-word-argmax/` is produced by `benchmark_soft_distillation` with its defaults (output dir, `data/soft-teacher-cv1m`, `data/wordle-cv5.json`, baseline checkpoints under `runs/scaling-cv5-1m`, the five folds and three seeds recorded in the mode file, `--resume` to continue an interrupted sweep). To evaluate existing hard-SFT checkpoints without training, run it with `--baseline-only --resume`.

## Anchored DPO (research history)

Standard DPO raised measured preference accuracy while crashing gameplay to 0% raw wins with 72/72 invalid guesses (see the [failure records](../runs/dpo-dev/beta-0.05/metrics.jsonl)); adding an SFT anchor to the chosen sequence rescued it. The rescue experiment lives in `wordle_gpt.dpo`:

```bash
python -m wordle_gpt.dpo.run_dpo_development   # standard DPO run on the development split
python -m wordle_gpt.dpo.run_dpo_rescue        # anchored sweep: λ ∈ {0, 0.1, 0.5, 1.0}, λ=1.0 at LR 1e-6 and 3e-6
python -m wordle_gpt.dpo.benchmark_dpo_cv      # benchmark sweep of the rescue
```

The rescue starts from `runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt` and writes to `runs/dpo-rescue-anchor/`. Lower-level pieces are `wordle_gpt.dpo.build_preferences` (builds chosen/rejected pairs from rollouts) and `wordle_gpt.dpo.train_dpo` (single run; takes explicit `--train-preferences`, `--validation-preferences`, `--mechanics-data`, `--base-checkpoint`, `--beta`). Results: [development summary](../runs/dpo-rescue-anchor/development-summary.md) · [benchmark](../runs/dpo-cv5-7.2m/aggregate.json).

## GRPO variants (research history)

Every GRPO trainer reports validation gameplay and policy diagnostics without touching test secrets. None of these runs produced the released policy; they are all reproducible from the original 7.2M SFT checkpoint.

### One-guess expected-information GRPO

Optimize expected candidate reduction across all consistent dictionary answers, with a small solve bonus:

```bash
python -m wordle_gpt.grpo.grpo_information_states
python -m wordle_gpt.grpo.train_information_grpo
```

Defaults: original 7.2M SFT, 1,000 updates, LR `3e-6`, KL `0.10`. Each state uses 48 policy proposals and 16 random guesses (`--groups-per-update`, state caps via `--train-cap 256 --validation-cap 32`). This mixed-proposal objective is a ranking surrogate, **not unbiased on-policy PPO**. Held-out secrets are excluded as sources but remain in the reward's answer universe.

### Token-level continuation GRPO

Train eight continuations per sampled mid-game state. Reward favors solving in fewer total guesses; only newly generated letters receive policy loss.

```bash
python -m wordle_gpt.grpo.grpo_corpus
python -m wordle_gpt.grpo.train_grpo_games \
  --state-corpus data/grpo-continuations/train.jsonl \
  --validation-corpus data/grpo-continuations/validation.jsonl \
  --updates 1000 --kl-beta 0.10 --lr 1e-6 \
  --output-dir runs/grpo-token-lr-1e6
```

For the learning-rate comparison, repeat with `--lr 3e-6` and `1e-5` in separate output directories. Skip corpus creation if the verified corpus exists. The objective uses per-token PPO ratios and equal rollout weighting. Fixed training/validation panels track continuation reward; `best-continuation.pt` is separate from the gameplay-ranked `best.pt`. Earlier whole-trajectory-ratio results are historical, not results of this objective.

### Full-trajectory GRPO

Train eight complete games per sampled secret, starting from the original 7.2M SFT:

```bash
python -m wordle_gpt.grpo.train_grpo_games --updates 10000 --output-dir runs/grpo-token-games-dev
```

- Reward: `7 - guesses_used` on a solve, otherwise `0`; no intermediate reward.
- Defaults: LR `1e-6`, frozen-SFT KL `0.10`, token-level clipping, and equal rollout weighting.
- Resume: `--resume-checkpoint <path>` with unchanged settings and a larger `--updates` total. Old trajectory-ratio checkpoints are incompatible.

### One-step GRPO

Optimize one guess at a time from the original 7.2M SFT, using actual secret feedback:

```bash
python -m wordle_gpt.grpo.train_grpo --updates 10000 --output-dir runs/grpo-dev-dense
```

Each state receives eight distinct legal guesses. Reward is `ln(candidates_before / candidates_after) + 5 * solved`; defaults are LR `1e-6` and frozen-SFT KL `0.02`. Resume is not supported; use a new output directory per run.

## Evaluation

```bash
# 5-fold cross-validation benchmark (final benchmark; also drives the SFT baseline arms)
python -m wordle_gpt.evaluation.benchmark_cv --skip-prepare

# Per-checkpoint gameplay on one split
python -m wordle_gpt.evaluation.evaluate_v2 PATH_TO_CHECKPOINT --split validation --details

# Whole-word vs token-greedy decoder audit (README finding 3)
python -m wordle_gpt.evaluation.audit_checkpoint_decoders
```

`benchmark_cv` writes per-seed combined artifacts and paired per-secret comparisons (A-loses/B-wins, A-wins/B-loses, both-win, both-lose, per-secret guess deltas). `evaluate_v2` defaults to `--split validation`; the fixed 72 test secrets need an explicit `--split test`. `audit_checkpoint_decoders` re-evaluates checkpoints under three decoder policies (prefix-masked token-greedy, conditional token-greedy, full-word argmax) and writes one JSON per checkpoint to `runs/checkpoint-decoder-audit/`. `wordle_gpt.evaluation.action_regret` is a library module used by the information-reward diagnostics, not a CLI.

## Checkpoint training tokens

Counts at the selected **SFT `best.pt`**, including its mechanics initialization. SFT includes mechanics replay; counts are supervised, non-padding target tokens processed, including repeats — not unique data or input/context tokens. Later distillation/GRPO training is excluded.

| Checkpoint | Mechanics pretraining | SFT + replay | Total |
| :--- | ---: | ---: | ---: |
| [815K replay baseline](../runs/v2-replay/e2-expert95-mechanics5/best.json) | [3,606,240](../runs/v2-experiment-b-mechanics/best.json) | Not recorded | Not recorded |
| [3.2M scaling baseline](../runs/scaling-dev-1m/seed-0/fold-1/3.2m/best.json) | [1,200,000](../runs/scaling-dev-1m/seed-0/fold-1/3.2m/mechanics/best.json) | 33,091,200 | **34,291,200** |
| [7.2M current SFT](../runs/scaling-dev-1m/seed-0/fold-1/7.2m/best.json) | [3,998,720](../runs/scaling-dev-1m/seed-0/fold-1/7.2m/mechanics/best.json) | 37,818,880 | **41,817,600** |

Scaling counts are for development seed 0, fold 1; other folds and checkpoints have different training lengths. The 815K checkpoint predates the SFT token counter.

## Reproducibility notes

- **Seeds.** Model seeds `0 1 2` drive both the benchmark sweeps and the CV mode files; dataset generation uses fixed seeds (`20260815`, `20260823`, `20260824`, `20260923`, `20260924` appear in the builders above).
- **Artifacts.** Every run writes `metrics.json`, checkpoints, and logs under `runs/<run-name>/`; `runs/**/checkpoints/` is gitignored, so retrain to regenerate a checkpoint while its metrics stay reviewable. `data/` is regenerable from pools plus the mode file and is also gitignored — except the committed pool archives, which are enough to rebuild it.
- **Failures are documented.** Runs that went nowhere (DPO collapse, GRPO reward hacking, token-greedy decoding collapse) are kept in `runs/` and written up in [EXPERIMENTS.md](EXPERIMENTS.md) — read them before re-running any of the sweeps above.
