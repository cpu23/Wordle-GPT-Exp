# Teaching a Small Transformer to Play Wordle

*From learning the rules to a 95.64% win rate.*

**Author:** Harris Oldroyd  
**Website:** [harrisoldroyd.com](https://harrisoldroyd.com)

Can a small Transformer learn Wordle rules and strategy from recorded games?

I trained models from 814,627 to 12.7 million parameters on up to one million game states. The work went through supervised training, preference training and reinforcement learning. The best result came from teaching a 7.2M model the solver's probabilities over several possible guesses, then scoring complete words during play.

That model reached a **95.64% win rate** across five test groups and three training seeds. The benchmark covers **719 known dictionary words**. Each word is tested once per seed, with six guesses per game.

The path to that result mattered as much as the final score. Training on strategy caused the model to forget the rules. Preference scores rose while gameplay collapsed. Reinforcement learning raised training rewards without a reliable gain on held-out games. Then a decoder change turned two wins into 72 with the same weights.

## 1. A small language for the game

I stored guesses and feedback as text, using 35 tokens:

- 26 letters: `a` to `z`.
- Three feedback digits: `0` for gray, `1` for yellow and `2` for green.
- Six control tokens: `<G>`, `<F>`, `<E>`, `<M>`, `<S>` and `<P>`.

The control tokens mark guesses, feedback, the end of a record, rules, secrets and policy prompts. A policy prompt asks for the next guess.

```text
Game record:
<P><G>could<F>22010<G>colon<F>22222<E>

Rule-training record:
<M><S>colon<G>could<F>22010<E>
```

Rule training gives the model both the secret and the guess. During play, it receives only previous guesses and feedback. The game engine supplies feedback after each move.

All models use four decoder-only Transformer layers and a 96-token context. I changed the layer width and number of attention heads to test model size.

| Parameters | Layer width | Attention heads |
| ---: | ---: | ---: |
| 814,627 | 128 | 4 |
| 3,202,083 | 256 | 8 |
| 7,162,403 | 384 | 12 |
| 12,695,587 | 512 | 16 |

## 2. Learning the rules, then forgetting them

I first checked that the model could memorize 32 game records. After 1,000 training steps, it reproduced a 19-token continuation exactly.

The next dataset mixed expert, simple and random moves. Training loss kept falling, but validation loss reached its lowest value at epoch four: **0.7759**. By epoch 20, it had risen to **1.2170**. The model was overfitting.

I then separated two tasks: predict feedback from a secret and guess, and predict an expert's next move from the visible board.

Training on rules worked well. Validation loss fell to **0.000515**. But further training on expert moves raised rule loss to **10.392833**, about 20,000 times higher. The same failure appeared across three seeds.

I kept rule examples in the training stream. With 95% expert batches and 5% rule batches, rule loss stayed at **0.007317**. This model also won **72/72** games on the small test split.

That gave me a practical training recipe: keep a small share of rule examples while teaching strategy.

[Rule retention and replay results](EXPERIMENTS.md#2026-08-14--experiment-e-multi-task-replay)

## 3. More states improved play

A 72-word test was useful for development. I expanded the benchmark to five groups covering all 719 words, repeated across three training seeds.

For each group, I removed training records whose source secret belonged to the test group. The dictionary stayed fixed. Checkpoint selection used a separate validation split.

I generated histories with several policies, including random and deliberately poor moves. A classical solver then assigned an expert guess to each state. It scored guesses by the expected number of answers left after feedback; lower was better.

For the larger datasets, I also selected states by their remaining-answer sets. This added more distinct decision problems instead of mostly adding similar histories.

<img src="figures/fig2-state-scaling.svg" alt="Win rate rises from 34.59% with 100K training states to 73.76% with 1M states." width="100%">

These runs used the **3.2M model**, choosing one letter at a time.

| Training states | Distinct remaining-answer sets | Win rate |
| ---: | ---: | ---: |
| 100,000 | 37,992 | 34.59% |
| 200,000 | 65,779 | 52.48% |
| 500,000 | 167,621 | 66.25% |
| 1,000,000 | 303,212 | 73.76% |

More states improved play at each step. The gain became smaller as the dataset grew.

[State-scaling results](EXPERIMENTS.md#2026-08-21--1m-logical-state-scaling-with-solver-top-actions)

## 4. Legal words helped, but losses remained

The model sometimes produced a five-letter string outside the dictionary. I added a decoder that allowed only prefixes of legal words. It still chose the most likely next letter at each step.

With one million training states, this raised the win rate from **73.76% to 77.05%** and removed invalid guesses.

Legal spelling helped. The model still lost games while choosing valid words. I next tested whether a larger model could make better choices.

## 5. A larger model reached about 80%

I held the one-million-state dataset fixed and compared model sizes. These results use the legal-prefix decoder.

| Model | Win rate |
| ---: | ---: |
| 3.2M | 76.59% |
| 7.2M | 80.44% |
| 12.7M | 80.39% |

The 7.2M model improved on 3.2M. Increasing it to 12.7M gave almost the same win rate. I kept the 7.2M model and changed the training method.

## 6. Better preference scores, worse gameplay

Hard supervised fine-tuning, or **SFT**, trains on one chosen guess per state. I wanted to teach the model the difference between stronger and weaker guesses.

I built **439,483 preference pairs** from the solver's scores and tried Direct Preference Optimization, or **DPO**. This method trains the model to prefer one guess over another.

In the run with a learning rate of `0.00001` and `β = 0.05`, validation preference accuracy rose from **75.56% to 86.97%**. Gameplay collapsed.

| Training passes | Unrestricted wins | Legal-prefix wins |
| ---: | ---: | ---: |
| 0 — SFT | 57/72 | 59/72 |
| 1 | 0/72 | 25/72 |
| 3 | 0/72 | 12/72 |

After one pass, all 72 unrestricted games ended with an invalid guess.

The probability records showed why the preference score was misleading. The model reduced the probability of both preferred and rejected words. Rejected words fell faster, so the preference margin improved even as valid guesses became less likely.

I added the supervised loss for the preferred word and reduced the learning rate. This kept a direct training signal for valid expert guesses.

Across five groups and three seeds, this anchored DPO method reached **81.46%**, compared with **80.44%** for SFT. It was a small improvement, about seven extra wins per seed.

[DPO failure records](../runs/dpo-dev/beta-0.05/metrics.jsonl) · [Anchored DPO benchmark](../runs/dpo-cv5-7.2m/aggregate.json)

## 7. GRPO did not give a reliable gain

At this point, I planned to train on game outcomes with Group Relative Policy Optimization, or **GRPO**. It samples several actions or games from one starting state, compares their rewards and updates the model toward the better outcomes.

I ran that plan from the original 7.2M SFT model.

First, I rewarded single guesses for reducing the answer set and solving the game. After 10,000 updates, legal-prefix validation wins fell from **59/72 to 57/72**.

Next, I sampled complete games and rewarded wins and fewer turns. Training wins were already above **99.97%**. Training reward rose, but validation wins fell from **59/72 to 49/72** after 10,000 updates.

I then started rollouts from recorded states, including later turns, to give the model harder training problems. I corrected the objective to clip probability ratios per generated token and give each rollout equal weight. I tested three learning rates for 1,000 updates each.

All three increased training reward. All three ended with lower held-out continuation reward. The selected gameplay checkpoint remained the original SFT model.

A final single-guess run used exact expected information gain. Its best gameplay checkpoint won **60/72**, one more game than SFT. By update 1,000, it had fallen to **54/72**.

The GRPO runs did not justify a larger benchmark. I stopped and returned to the solver's training targets.

[GRPO runs and objective correction](EXPERIMENTS.md#2026-09-23--dense-10000-update-grpo-run)

## 8. Teach probabilities over several guesses

The solver can give several guesses similar scores. Hard SFT selects one and trains the model to reproduce it. I changed the target to a probability distribution over **128 legal guesses per state**.

This method is **soft distillation**. Better solver scores receive more probability. A temperature controls how strongly the distribution favors the best scores. The final runs used **0.25** and kept 5% rule replay.

I tested two starting points:

- **B:** a model trained on rules, with a learning rate of `0.0003`.
- **C:** a model already trained with hard SFT, with a learning rate of `0.00001`.

The next problem appeared when I tried to play games with the soft-trained model.

## 9. The decoder changed the result

On one development checkpoint, legal-prefix decoding won only **2/72** games. Scoring each complete legal word with the same model won **72/72**.

<img src="figures/fig3-decoder.svg" alt="The same checkpoint wins 2 of 72 games with token-greedy decoding and 72 of 72 with full-word scoring." width="100%">

The weights stayed the same. The decoder changed.

Choosing the most likely next letter commits to a prefix before the later letters are scored. The most likely first letter can lead to a lower-probability complete word.

I instead scored all 719 words by adding the model's log-probability for each of their five letters. The decoder selected the complete word with the highest score. Each letter probability depends on the board history and the earlier letters in that word.

An independent direct-forward check reproduced all 72 game trajectories and found zero word-choice disagreements on the checked panel.

I made full-word scoring the decoder for checkpoint selection, the final benchmark and the demo.

[Same-checkpoint comparison](../runs/soft-distillation-resumable/diagnostics/soft_best.json) · [Independent check](../runs/soft-distillation-resumable/diagnostics/independent-verification.json)

## 10. The final benchmark

I evaluated hard SFT and both soft-distillation variants with the same full-word decoder, five test groups and three seeds. Each method played **2,157 held-out games**.

<img src="figures/fig1-sft-vs-distillation.svg" alt="Final win rates: hard SFT 81.92%, soft distillation B 95.64%, soft distillation C 81.69%." width="100%">

| Training method | Win rate | Standard deviation across seeds |
| :--- | ---: | ---: |
| Hard SFT | 81.92% | 0.14 points |
| **Soft distillation B** | **95.64%** | **0.80 points** |
| Soft distillation C | 81.69% | 0.40 points |

All three produced zero invalid guesses. B improved on hard SFT by **13.72 percentage points**, about **99 extra wins per seed**. C stayed close to the baseline. B and C used different starting weights and learning rates.

The final result combines three choices: varied training states, soft targets from the solver and complete-word scoring. More parameters gave a smaller gain. The reinforcement-learning runs gave no reliable improvement over SFT.

The decoder finding also changed how I read the earlier failures. A gameplay score depends on both the trained weights and the rule used to select an action. Here, changing that rule had a larger effect than any training change on the same checkpoint.

[Full benchmark data](../runs/soft-distillation-cv5-word-argmax/benchmark-complete.json)

## Run the finished model

The project is complete and released as [v1.0.0](https://github.com/cpu23/wordle-gpt-family/releases/tag/v1.0.0). The release contains one 7.2M B checkpoint, its tokenizer and the 719-word list. That checkpoint won **135/144 games (93.75%)** on its own test group.

From the repository root, with Python 3.11:

```bash
python -m pip install -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m wordle_gpt.demo --secret colon
```

The first run downloads the model. The demo prints each guess, feedback and remaining turns. Use `--interactive` to enter feedback from your own board.

The model card records the checkpoint and file hashes. The training guide, logs and failed runs remain in the repository.

[Inference guide](INFERENCE.md) · [Model card](MODEL_CARD.md) · [Training guide](TRAINING.md) · [Experiment log](EXPERIMENTS.md)
