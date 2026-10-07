from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from wordle_gpt.core.model import WordleGPT, checkpoint_model_config
from wordle_gpt.core.tokenizer import FEEDBACK_TO_SYMBOL, FEEDBACK_TOKEN, GUESS_TOKEN
from wordle_gpt.core.tokenizer_v2 import POLICY_TOKEN, VOCABULARY_SIZE, decode, encode
from wordle_gpt.training.train import generate_constrained_guess, generate_tokens
from wordle_gpt.core.wordle import DEFAULT_WORDS, load_words, score_guess

DEFAULT_SPLITS_PATH = Path("data/wordle-100k/secret-splits.json")
DEFAULT_MAX_GUESSES = 6
LETTER_TOKEN_LIMIT = 26
DECODE_MODES = ("raw", "constrained", "word-argmax")


@dataclass(frozen=True)
class GameResult:
    secret: str
    guesses: tuple[str, ...]
    won: bool
    invalid_guesses: int


@dataclass(frozen=True)
class GameplaySummary:
    checkpoint: str
    games: int
    wins: int
    win_rate: float
    average_attempts: float
    average_guesses: float
    invalid_guesses: int
    decode: str
    results: tuple[GameResult, ...]


def _feedback_symbols(feedback: str) -> str:
    return "".join(FEEDBACK_TO_SYMBOL[mark] for mark in feedback)


def load_v2_model(checkpoint_path: str | Path, device: str) -> WordleGPT:
    """Restore one v2 checkpoint for deterministic greedy evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("vocabulary_size") != VOCABULARY_SIZE:
        raise ValueError("checkpoint does not use the current v2 vocabulary")
    model = WordleGPT(
        **checkpoint_model_config(checkpoint, vocab_size=VOCABULARY_SIZE)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class WordArgmaxPolicy:
    """Full-dictionary joint-word MAP; cache only while model weights are fixed."""

    def __init__(self, model, words):
        self.model = model
        self.words = tuple(words)
        self.tokens = torch.tensor(
            [encode(word) for word in self.words],
            device=next(model.parameters()).device,
        )
        self.cache = {}

    @torch.inference_mode()
    def __call__(self, prefix):
        from wordle_gpt.distillation.soft_policy import candidate_sequence_logps

        key = tuple(prefix)
        if key not in self.cache:
            prompts = torch.tensor([prefix], device=self.tokens.device)
            lengths = torch.tensor([len(prefix)], device=self.tokens.device)
            scores = candidate_sequence_logps(
                self.model, prompts, lengths, self.tokens.unsqueeze(0)
            )
            self.cache[key] = encode(self.words[int(scores[0].argmax())])
        return self.cache[key]


def play_secret(
    model: WordleGPT,
    secret: str,
    allowed_words: frozenset[str],
    *,
    max_guesses: int = DEFAULT_MAX_GUESSES,
    constrained: bool = False,
    policy=None,
) -> GameResult:
    """Play one game greedily; an invalid generated word ends the game.

    With ``constrained=True`` every guess is token-masked so that the
    generated five-letter word is guaranteed to be an allowed word.
    """
    prefix = encode(POLICY_TOKEN + GUESS_TOKEN)
    guesses: list[str] = []
    for _ in range(max_guesses):
        if policy is not None:
            generated = prefix + policy(prefix)
        elif constrained:
            generated = prefix + generate_constrained_guess(
                model, prefix, allowed_words
            )
        else:
            generated = generate_tokens(model, prefix, max_new_tokens=5)
        guess_ids = generated[-5:]
        if any(not 0 <= token_id < LETTER_TOKEN_LIMIT for token_id in guess_ids):
            guesses.append(decode(guess_ids))
            return GameResult(secret, tuple(guesses), False, 1)
        guess = decode(guess_ids)
        guesses.append(guess)
        if guess not in allowed_words:
            return GameResult(secret, tuple(guesses), False, 1)
        if guess == secret:
            return GameResult(secret, tuple(guesses), True, 0)
        feedback = _feedback_symbols(score_guess(secret, guess))
        prefix = generated + encode(FEEDBACK_TOKEN + feedback + GUESS_TOKEN)
    return GameResult(secret, tuple(guesses), False, 0)


def evaluate_model(
    model: WordleGPT,
    secrets: Sequence[str],
    allowed_words: Sequence[str],
    *,
    checkpoint: str = "in-memory",
    decode: str = "raw",
) -> GameplaySummary:
    """Play every supplied secret with an already-loaded model."""
    if decode not in DECODE_MODES:
        raise ValueError(f"unknown decode mode: {decode!r}")
    constrained = decode == "constrained"
    allowed = frozenset(allowed_words)
    policy = WordArgmaxPolicy(model, allowed_words) if decode == "word-argmax" else None
    results = tuple(
        play_secret(model, secret, allowed, constrained=constrained, policy=policy)
        for secret in secrets
    )
    wins = sum(result.won for result in results)
    return GameplaySummary(
        checkpoint=checkpoint,
        games=len(results),
        wins=wins,
        win_rate=wins / len(results),
        average_guesses=(
            sum(len(result.guesses) for result in results if result.won) / wins
            if wins
            else 0.0
        ),
        average_attempts=sum(len(result.guesses) for result in results) / len(results),
        invalid_guesses=sum(result.invalid_guesses for result in results),
        decode=decode,
        results=results,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    secrets: Sequence[str],
    allowed_words: Sequence[str],
    *,
    device: str,
    decode: str = "raw",
) -> GameplaySummary:
    """Restore a checkpoint and play every supplied secret."""
    model = load_v2_model(checkpoint_path, device)
    return evaluate_model(
        model,
        secrets,
        allowed_words,
        checkpoint=str(checkpoint_path),
        decode=decode,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedily play a v2 checkpoint on a held-out secret split."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument(
        "--decode",
        choices=DECODE_MODES,
        default="raw",
        help="Raw tokens, legal-prefix token greedy, or full-dictionary joint-word argmax.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--details", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    summary = evaluate_checkpoint(
        args.checkpoint,
        splits[args.split],
        load_words(args.words),
        device=device,
        decode=args.decode,
    )
    payload = asdict(summary)
    if not args.details:
        payload.pop("results")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
