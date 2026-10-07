# Wordle GPT

A compact decoder-only Transformer exploring whether a neural network can learn both the rules and an effective strategy for Wordle purely from game trajectories as text.

Rather than relying on an external game engine or search algorithm at runtime, Wordle GPT treats the game as a sequence-prediction problem: tracking board state, respecting feedback constraints, and generating informative guesses through autoregressive token generation.

---

## Core Questions

1. **Rule learning:** Can a small transformer learn valid 5-letter dictionary words and exact Wordle feedback mechanics without hardcoded rules?
2. **Strategy acquisition:** Can the model learn to narrow down possible candidate words and pick high-information guesses purely by imitating expert play?
3. **Catastrophic forgetting:** Does learning strategic play erase the model's understanding of basic game rules, and can multi-task experience replay prevent that regression?

---

## Model Architecture

The model is a standard pre-norm decoder-only Transformer with causal multi-head self-attention, GELU feed-forward blocks, and learned positional embeddings.

Current distillation and GRPO experiments use the **7.2M-parameter model**. The smaller models below are earlier baselines.

| Parameter | Original Baseline | Earlier Scaled Model | Current Model |
| :--- | :--- | :--- | :--- |
| **Parameters** | ~815,000 | 3,202,083 | **7,162,403** |
| **Embedding Size** | 128 | 256 | 384 |
| **Layers** | 4 | 4 | 4 |
| **Attention Heads** | 4 | 8 | 12 |
| **MLP Hidden Dim** | 512 | 1024 | 1536 |
| **Context Length** | 96 tokens | 96 tokens | 96 tokens |
| **Vocabulary** | 35 tokens | 35 tokens | 35 tokens |

The 35-token vocabulary consists of:
- 26 lowercase English letters (`a`–`z`)
- 3 feedback digits: `0` (gray / miss), `1` (yellow / wrong position), `2` (green / exact hit)
- 6 structural control tokens: `<G>` (guess), `<F>` (feedback), `<E>` (end of game), `<M>` (mechanics task), `<S>` (secret word), `<P>` (policy task)

### Checkpoint training tokens

Counts at the selected **SFT `best.pt`**, including its mechanics initialization. SFT includes mechanics replay; counts are supervised, non-padding target tokens processed, including repeats—not unique data or input/context tokens. Later distillation/GRPO training is excluded.

| Checkpoint | Mechanics pretraining | SFT + replay | Total |
| :--- | ---: | ---: | ---: |
| [815K replay baseline](runs/v2-replay/e2-expert95-mechanics5/best.json) | [3,606,240](runs/v2-experiment-b-mechanics/best.json) | Not recorded | Not recorded |
| [3.2M scaling baseline](runs/scaling-dev-1m/seed-0/fold-1/3.2m/best.json) | [1,200,000](runs/scaling-dev-1m/seed-0/fold-1/3.2m/mechanics/best.json) | 33,091,200 | **34,291,200** |
| [7.2M current SFT](runs/scaling-dev-1m/seed-0/fold-1/7.2m/best.json) | [3,998,720](runs/scaling-dev-1m/seed-0/fold-1/7.2m/mechanics/best.json) | 37,818,880 | **41,817,600** |

Scaling counts are for development seed 0, fold 1; other folds/checkpoints have different training lengths. The 815K checkpoint predates the SFT token counter.

---

## Sequence Representation

Games are serialized as token sequences alternating between guesses and feedback:

```text
<G>could<F>22010<G>colon<F>22222<E>
```

### Multi-Task Objectives

- **Mechanics (`<M>`):** Given a secret word and a guess, predict the 5-digit feedback pattern:
  `<M><S>colon<G>could<F>22010<E>`
- **Policy (`<P>`):** Given the observed game history, predict the next strategic guess:
  `<P><G>could<F>22010<G>colon<E>`

Training with multi-task experience replay (blending mechanics examples during policy fine-tuning) prevents catastrophic forgetting of game rules while policy performance improves.

---

## Key Findings

- **Mechanics are learned rapidly:** The model achieves near-perfect prediction of Wordle feedback patterns within the first few training epochs.
- **Replay stabilizes strategy:** Fine-tuning exclusively on expert moves causes the model to forget game rules. A 5%–10% mechanics replay ratio maintains rule fidelity without degrading gameplay quality.
- **State scaling drives performance:** Scaling training from 10,000 to 1,000,000 unique game states improves held-out 5-fold cross-validation win rates from ~70% to **>98%**.
- **Constrained decoding:** Filtering output logits at generation time to valid 5-letter dictionary words eliminates rare spelling failures and yields consistent wins.

*For full daily training logs, loss curves, gradient norms, and ablation studies, see [EXPERIMENTS.md](docs/EXPERIMENTS.md).*

---

## Quickstart

### Setup

Requires Python 3.11+. Install dependencies using `uv` or standard `pip`. Run all commands below from the repository root:

```bash
uv run --with-requirements requirements.txt python -m unittest discover -s tests
# or with standard pip:
pip install -r requirements.txt
python -m unittest discover -s tests
```

### Training

Train the original small baseline on nested state datasets (not the current 7.2M model):

```bash
uv run --with-requirements requirements.txt \
  python -m wordle_gpt.training.train_nested \
  --sizes 100000 \
  --steps 10000 \
  --checkpoints 0 100 500 1000 5000 10000
```

### Soft classical-policy distillation

Train against a classical solver's probability distribution over 128 legal guesses, rather than a single expert move.

```bash
uv run --with-requirements requirements.txt python -m wordle_gpt.distillation.soft_teacher \
  --output-dir data/soft-teacher-1m --workers 8 \
  --mode data/wordle-development.json
uv run --with-requirements requirements.txt python -m wordle_gpt.distillation.train_soft_distillation
uv run --with-requirements requirements.txt python -m wordle_gpt.distillation.compare_soft_distillation
uv run --with-requirements requirements.txt python -m wordle_gpt.distillation.benchmark_soft_distillation
```

- **Training:** Compare mechanics-pretrained (B) and hard-SFT-initialized (C) models at temperatures `0.25`, `0.50`, and `1.00`, with 5% mechanics replay.
- **Evaluation:** Select guesses by full-word probability over all 719 words, not token-greedy decoding. The fixed benchmark uses `T=0.25`, five folds, and three seeds, subject to a development gate.
- **Resume:** Repeat training with `--resume`. Runs save every 100 updates and on graceful stop; legacy token-greedy runs are incompatible.

Skip teacher construction if the verified dataset exists. Use `python -m wordle_gpt.distillation.benchmark_soft_distillation --baseline-only --resume` to evaluate existing hard-SFT checkpoints without training.

### One-guess expected-information GRPO

Optimize expected candidate reduction across all consistent dictionary answers, with a small solve bonus:

```bash
uv run --with-requirements requirements.txt python -m wordle_gpt.grpo.grpo_information_states
uv run --with-requirements requirements.txt python -m wordle_gpt.grpo.train_information_grpo
```

Defaults: original 7.2M SFT, 1,000 updates, LR `3e-6`, KL `0.10`. Each state uses 48 policy proposals and 16 random guesses. This mixed-proposal objective is a ranking surrogate, **not unbiased on-policy PPO**. Held-out secrets are excluded as sources, but remain in the reward's answer universe.

### Token-level continuation GRPO

Train eight continuations per sampled mid-game state. Reward favors solving in fewer total guesses; only newly generated letters receive policy loss.

```bash
uv run --with-requirements requirements.txt python -m wordle_gpt.grpo.grpo_corpus
uv run --with-requirements requirements.txt python -m wordle_gpt.grpo.train_grpo_games \
  --state-corpus data/grpo-continuations/train.jsonl \
  --validation-corpus data/grpo-continuations/validation.jsonl \
  --updates 1000 --kl-beta 0.10 --lr 1e-6 \
  --output-dir runs/grpo-token-lr-1e6
```

For the learning-rate comparison, repeat with `--lr 3e-6` and `1e-5`, using separate output directories. Skip corpus creation if the verified corpus exists.

The objective uses per-token PPO ratios and equal rollout weighting. Fixed training/validation panels track continuation reward; `best-continuation.pt` is separate from the gameplay-ranked `best.pt`. Earlier whole-trajectory-ratio results are historical, not results of this objective.

### Full-trajectory GRPO

Train eight complete games per sampled secret, starting from the original 7.2M SFT:

```bash
uv run --with-requirements requirements.txt python -m wordle_gpt.grpo.train_grpo_games \
  --updates 10000 --output-dir runs/grpo-token-games-dev
```

- **Reward:** `7 - guesses_used` on a solve, otherwise `0`; no intermediate reward.
- **Defaults:** LR `1e-6`, frozen-SFT KL `0.10`, token-level clipping, and equal rollout weighting.
- **Resume:** Use `--resume-checkpoint <path>` with unchanged settings and a larger `--updates` total. Old trajectory-ratio checkpoints are incompatible.

### One-step GRPO

Optimize one guess at a time from the original 7.2M SFT, using actual secret feedback:

```bash
uv run --with-requirements requirements.txt python -m wordle_gpt.grpo.train_grpo \
  --updates 10000 --output-dir runs/grpo-dev-dense
```

Each state receives eight distinct legal guesses. Reward is `ln(candidates_before / candidates_after) + 5 * solved`; defaults are LR `1e-6` and frozen-SFT KL `0.02`. Resume is not supported; use a new output directory per run.

GRPO runs report validation gameplay and policy diagnostics without evaluating test secrets. See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for run histories and results.

### Evaluation

Run the 5-fold cross-validation benchmark across multiple seeds:

```bash
uv run --with-requirements requirements.txt python -m wordle_gpt.evaluation.benchmark_cv --skip-prepare
```

---

## Repository Structure

- `wordle_gpt/` — production package, grouped into `core/`, `datasets/`, `training/`, `evaluation/`, `experiments/`, `dpo/`, `grpo/`, and `distillation/`.
- `tests/` — test suite.
- `resources/words.txt` and `resources/sample-trajectories.jsonl.gz` — runtime inputs.
- `data/` and `runs/` — generated datasets and run artifacts.
- `docs/` — [ARTICLE.md](docs/ARTICLE.md) and [EXPERIMENTS.md](docs/EXPERIMENTS.md).
