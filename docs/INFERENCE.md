# Try the released policy

A CPU-only demo of the completed research project. No training data, teacher
cache, GPU or training run is needed. The first game downloads the selected
checkpoint, tokenizer and 719-word dictionary from the
[v1.0.0 release](https://github.com/cpu23/wordle-gpt-family/releases/tag/v1.0.0).

## Reference environment

Verified on Linux x86_64 with Python **3.11.15**, PyTorch **2.6.0+cpu** and
NumPy **2.2.6**. All Python dependencies are pinned in
[`requirements-inference.txt`](../requirements-inference.txt).

```bash
git clone --branch v1.0.0 https://github.com/cpu23/wordle-gpt-family.git
cd wordle-gpt-family
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m wordle_gpt.demo --secret colon
```

After setup, the last line is the only command needed to play. The selected
model is variant B, seed 0, fold 1; it is **one** checkpoint, not the ensemble
of fold-specific checkpoints used to measure the final benchmark.

## Observed game

The download and cache-path messages depend on your machine. The game output
in the frozen reference environment is:

```text
Model: variant B | 7162403 parameters | 719 dictionary words
Decoder: word-argmax (argmax joint five-letter log-probability)
Feedback: G=green (correct spot), Y=yellow (wrong spot), X=gray (absent).
Secret: colon
Turn 1/6: stare -> XXXXX | turns left 5 | candidate words 46
Turn 2/6: cling -> GYXYX | turns left 4 | candidate words 1
Turn 3/6: colon -> GGGGG | turns left 3 | candidate words 1
Solved 'colon' in 3 guesses.
```

Each turn scores the complete five-letter sequence for **all 719 legal words**
and selects the highest joint probability. It does not construct a guess by
choosing the most likely next token five times. Wordle feedback comes from the
game engine; candidate counts are diagnostic only and do not filter or override
the model's choice. The chosen secret is not supplied to the neural policy.

## Play against your own board

```bash
python -m wordle_gpt.demo --interactive
```

Supply five feedback letters after each guess: `g` for green, `y` for yellow,
`x` for gray. Use a secret from the bundled dictionary; a live NYT game can use
words outside this research vocabulary. A blank response stops the demo.

## Offline and integrity checks

Download
[`wordle-gpt-7.2m-soft-v1.0.0.zip`](https://github.com/cpu23/wordle-gpt-family/releases/download/v1.0.0/wordle-gpt-7.2m-soft-v1.0.0.zip)
once and run:

```bash
sha256sum wordle-gpt-7.2m-soft-v1.0.0.zip
python -m wordle_gpt.demo --bundle wordle-gpt-7.2m-soft-v1.0.0.zip --secret colon
```

Expected archive SHA-256:

```text
b1d07d64f061f23c472deb58c20c4d92a49d20fccc4513002341c4f2e49aa674
```

`--bundle` also accepts an extracted directory. The demo checks file hashes
against the bundle manifest, checks token order, and loads the model with
`weights_only=True`. The default download uses HTTPS. Compare the archive hash
above when obtaining the bundle from another source; manifest checks alone
are integrity checks, not authentication of arbitrary third-party bundles.

The default cache is `$XDG_CACHE_HOME/wordle-gpt-demo` on Linux (or
`~/.cache/wordle-gpt-demo` when unset). Override it with `--cache-dir PATH` or
`WORDLE_GPT_CACHE`. Subsequent games use the local cached bundle.

Exit status: `0` solved, `1` exhausted the guess budget, `2` input/bundle error.
The model can lose and repeat guesses; the demo deliberately preserves the
measured policy rather than adding a solver fallback.

## Evidence and reuse

- [Model card](MODEL_CARD.md): provenance, selected checkpoint's **135/144**
  held-out wins, aggregate benchmark distinction, checksums and limitations.
- [Training guide](TRAINING.md): optional reproduction of the experiments.
- [MIT licence](../LICENSE): code and released model weights.

The archive is a GitHub Release asset, not a committed checkpoint. Historical
training logs, failed experiments and run reports remain in the source tree.
