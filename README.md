# Wordle GPT

**Can a small Transformer learn Wordle rules and strategy from recorded games?**

This was my first Transformer project, inspired by Andrej Karpathy's videos. I used it to learn model design, training from scratch and reinforcement learning. Wordle gave me cheap synthetic data and deterministic rules, so feedback and game outcomes were easy to check.

This project trains a model on guesses and feedback stored as text. The best model scores complete legal words and selects the word with the highest probability.

**Completed · [v1.0.0](https://github.com/cpu23/wordle-gpt-family/releases/tag/v1.0.0) · [MIT licence](LICENSE)**

## Final result

**95.64% win rate** with a 7.2M-parameter model trained on a solver's word-choice probabilities. This method is called **soft distillation**. Hard supervised fine-tuning (**SFT**) trains on one chosen word per example.

| Training method | Starting model | Win rate |
| :--- | :--- | ---: |
| **Soft distillation (B)** | Trained on rules | **95.64%** |
| Hard SFT | Trained on rules | 81.92% |
| Soft distillation (C) | Trained with hard SFT | 81.69% |

The benchmark uses **719 known dictionary words**, five test groups and three training seeds. Each word is tested once per seed. All methods score complete legal words and have zero invalid guesses.

[Benchmark data](runs/soft-distillation-cv5-word-argmax/benchmark-complete.json) · [Model card](docs/MODEL_CARD.md)

## Architecture

| Parameters | Layers | Embedding size | Attention heads | Context | Vocabulary |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7,162,403 | 4 | 384 | 12 | 96 tokens | 35 tokens |

The model reads previous guesses and feedback. Its vocabulary contains 26 letters, three feedback digits and six control tokens.

## Three findings

### 1 · Soft distillation raises the win rate by 13.7 points

<a href="runs/soft-distillation-cv5-word-argmax/benchmark-complete.json"><img src="docs/figures/fig1-sft-vs-distillation.svg" alt="Win rates: hard SFT 81.92%, soft distillation B 95.64%, soft distillation C 81.69%." width="100%"></a>

B starts from a model trained on rules and reaches **95.64%**. C starts from hard SFT and reaches **81.69%**. Their learning rates are 0.0003 and 0.00001, respectively.

### 2 · More training states improve play

<a href="docs/EXPERIMENTS.md"><img src="docs/figures/fig2-state-scaling.svg" alt="Win rate rises from 34.59% at 100K training states to 73.76% at 1M states." width="100%"></a>

An earlier 3.2M model improved from **34.59% to 73.76%** as training grew from 100K to 1M game states. That model chose guesses one letter at a time.

### 3 · A decoder change turns 2 wins into 72

<a href="runs/soft-distillation-resumable/diagnostics/soft_best.json"><img src="docs/figures/fig3-decoder.svg" alt="The same trained model wins 2 of 72 games with token-greedy decoding and 72 of 72 with full-word scoring." width="100%"></a>

The same trained model won **2/72** games when it chose the most likely letter at each step. It won **72/72** when it scored each complete legal word. The weights stayed the same. The final benchmark and demo use full-word scoring.

## Run the model

From the repository root, with Python 3.11:

```bash
python -m pip install -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m wordle_gpt.demo --secret colon
```

The first run downloads the model, tokenizer and word list. Each turn shows the guess, feedback and remaining turns. Use `--interactive` to enter feedback from your own board.

The released model wins **135/144 games (93.75%)** on its test group. [Inference guide](docs/INFERENCE.md)

## Research record

- [Training guide](docs/TRAINING.md) — commands and model settings.
- [Experiment log](docs/EXPERIMENTS.md) — runs, results and failures.
- [Model card](docs/MODEL_CARD.md) — training source, scores and file hashes.
- [Release notes](docs/RELEASE.md) — the completed version.

Original code, experiment logs and research artifacts are preserved.
