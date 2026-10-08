"""Play Wordle on CPU with the released 7.2M soft-distilled policy.

The demo downloads the fixed ``v1.0.0`` GitHub release asset
``wordle-gpt-7.2m-soft-v1.0.0.zip`` into a local cache, verifies the bundle
checksums, and plays one game against a chosen secret. Guesses are scored with
the same whole-word decoder as the released benchmark: each turn it picks the
dictionary word with the highest joint five-letter log-probability under the
checkpoint ("word-argmax"). There is no search, teacher, or token-by-token
generation at inference.

Examples::

    python -m wordle_gpt.demo                    # secret defaults to "colon"
    python -m wordle_gpt.demo --secret colon
    python -m wordle_gpt.demo --bundle dist/wordle-gpt-7.2m-soft-v1.0.0
    python -m wordle_gpt.demo --bundle dist/wordle-gpt-7.2m-soft-v1.0.0.zip
    python -m wordle_gpt.demo --interactive      # you supply the feedback
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

import torch

from wordle_gpt.core.model import MODEL_CONFIG_KEYS, WordleGPT
from wordle_gpt.core.tokenizer import FEEDBACK_TOKEN, FEEDBACK_TO_SYMBOL, GUESS_TOKEN
from wordle_gpt.core.tokenizer_v2 import (
    POLICY_TOKEN,
    TOKENS,
    VOCABULARY_SIZE,
    decode,
    encode,
)
from wordle_gpt.core.wordle import GREEN, filter_answers, load_words, score_guess
from wordle_gpt.evaluation.evaluate_v2 import WordArgmaxPolicy, load_v2_model

RELEASE_REPOSITORY = "cpu23/wordle-gpt-family"
RELEASE_TAG = "v1.0.0"
ASSET_NAME = "wordle-gpt-7.2m-soft-v1.0.0.zip"
RELEASE_URL = (
    f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
)
BUNDLE_FORMAT = "wordle-gpt-bundle/1"
TOKENIZER_FORMAT = "wordle-gpt-tokenizer/1"
REQUIRED_FILES = ("manifest.json", "model.pt", "tokenizer.json", "words.txt")
CHECKED_FILES = ("model.pt", "tokenizer.json", "words.txt")
DEFAULT_SECRET = "colon"
DEFAULT_MAX_GUESSES = 6
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120
USER_AGENT = f"wordle-gpt-demo/1.0 ({RELEASE_REPOSITORY})"


@dataclass(frozen=True)
class Bundle:
    """One verified release bundle: model, dictionary, and provenance."""

    path: Path
    model: WordleGPT
    words: tuple[str, ...]
    manifest: Mapping[str, object]


def default_cache_dir() -> Path:
    """Return the platform cache directory used for the release asset."""
    override = os.environ.get("WORDLE_GPT_CACHE")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "wordle-gpt-demo"


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract ``archive`` into ``destination``, refusing unsafe members.

    Members must be plain relative paths (no absolute paths, drive letters,
    backslashes, ``..`` traversal, or symlinks) and the total uncompressed
    size must stay under ``MAX_ARCHIVE_BYTES``.
    """
    with zipfile.ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        if not members:
            raise ValueError(f"release archive is empty: {archive}")
        total_bytes = 0
        for member in members:
            name = member.filename
            posix = PurePosixPath(name)
            if (
                not name
                or posix.is_absolute()
                or name.startswith(("\\", "/"))
                or "\\" in name
                or ":" in name
                or ".." in posix.parts
            ):
                raise ValueError(f"release archive contains an unsafe path: {name!r}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"release archive contains a symlink: {name!r}")
            total_bytes += member.file_size
            if total_bytes > MAX_ARCHIVE_BYTES:
                raise ValueError("release archive exceeds the uncompressed size limit")
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            if member.is_dir():
                continue
            target = destination / PurePosixPath(member.filename)
            if not _is_within(target, destination):
                raise ValueError(f"release archive escapes the extraction root: {member.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def download_asset(url: str, destination: Path) -> Path:
    """Stream ``url`` to ``destination`` atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with partial.open("wb") as sink:
                shutil.copyfileobj(response, sink)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"could not download the release asset ({error.code} {error.reason}); "
            f"check that release {RELEASE_TAG} of {RELEASE_REPOSITORY} is published, "
            "or download the asset manually and pass --bundle"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"could not reach {url} ({error.reason}); download the release asset "
            "manually and pass --bundle for offline use"
        ) from error
    if not zipfile.is_zipfile(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file is not a zip archive: {url}")
    os.replace(partial, destination)
    print(f"Downloaded {destination.stat().st_size} bytes to {destination}")
    return destination


def _load_manifest(directory: Path) -> Mapping[str, object]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"bundle is missing manifest.json: {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"bundle manifest.json is not valid JSON: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"bundle manifest.json has an unsupported format: {directory}")
    return manifest


def verify_bundle_files(directory: Path) -> Mapping[str, object]:
    """Check every bundled file against the checksums recorded in the manifest."""
    manifest = _load_manifest(directory)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("bundle manifest.json is missing the files section")
    for name in CHECKED_FILES:
        record = files.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError(f"bundle manifest.json has no checksum for {name}")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"bundle is missing {name}: {directory}")
        digest = _hash_file(path)
        if digest != record["sha256"]:
            raise ValueError(
                f"checksum mismatch for {name}: expected {record['sha256']}, got {digest}"
            )
    return manifest


def load_bundle(directory: Path) -> Bundle:
    """Verify and load one extracted bundle for CPU inference."""
    manifest = verify_bundle_files(directory)
    tokenizer = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    if not isinstance(tokenizer, dict) or tokenizer.get("format") != TOKENIZER_FORMAT:
        raise ValueError("bundle tokenizer.json has an unsupported format")
    tokens = tokenizer.get("tokens")
    if not isinstance(tokens, list) or tuple(tokens) != TOKENS:
        raise ValueError(
            "bundle tokenizer does not match this runtime's tokenizer; "
            "the bundle was built for a different token vocabulary"
        )
    words = load_words(directory / "words.txt")
    model = load_v2_model(directory / "model.pt", "cpu")
    expected = manifest.get("model")
    if isinstance(expected, dict) and isinstance(expected.get("architecture"), dict):
        architecture = expected["architecture"]
        if set(architecture) != set(MODEL_CONFIG_KEYS) or dict(model.config) != architecture:
            raise ValueError("bundle model.pt does not match the manifest architecture")
    return Bundle(path=directory, model=model, words=words, manifest=manifest)


def resolve_bundle(path: Path, cache_dir: Path) -> Path:
    """Accept an extracted bundle directory or a bundle zip archive."""
    if path.is_dir():
        return path
    if path.is_file() and zipfile.is_zipfile(path):
        directory = cache_dir / f"{path.stem}-{_hash_file(path)[:12]}"
        if not all((directory / name).is_file() for name in REQUIRED_FILES):
            staging = cache_dir / f"{directory.name}.tmp"
            if staging.exists():
                shutil.rmtree(staging)
            safe_extract_zip(path, staging)
            verify_bundle_files(staging)
            if directory.exists():
                shutil.rmtree(directory)
            os.replace(staging, directory)
        return directory
    raise ValueError(f"bundle path is neither a directory nor a zip archive: {path}")


def ensure_bundle(cache_dir: Path, asset_url: str) -> Path:
    """Return a verified extracted bundle, downloading the release asset if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    directory = cache_dir / ASSET_NAME.removesuffix(".zip")
    if all((directory / name).is_file() for name in REQUIRED_FILES):
        try:
            verify_bundle_files(directory)
            return directory
        except ValueError as error:
            print(f"Discarding invalid cached bundle: {error}", file=sys.stderr)
            shutil.rmtree(directory, ignore_errors=True)
    archive = cache_dir / ASSET_NAME
    for attempt in (1, 2):
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            download_asset(asset_url, archive)
        staging = cache_dir / f".{directory.name}.extracting"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            safe_extract_zip(archive, staging)
            verify_bundle_files(staging)
        except ValueError as error:
            shutil.rmtree(staging, ignore_errors=True)
            archive.unlink(missing_ok=True)
            if attempt == 2:
                raise
            print(f"Retrying download: {error}", file=sys.stderr)
            continue
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(staging, directory)
        return directory
    raise RuntimeError("could not prepare the release bundle")


def _read_feedback(guess: str) -> str | None:
    while True:
        try:
            raw = input(f"Feedback for {guess!r} (5 letters, g/y/x; blank to quit): ")
        except EOFError:
            print()
            return None
        answer = raw.strip().upper()
        if not answer:
            return None
        if len(answer) == 5 and all(mark in "GYX" for mark in answer):
            return answer
        print("Enter exactly five letters from g (green), y (yellow), x (gray).")


def play_game(
    model: WordleGPT,
    words: Sequence[str],
    secret: str | None,
    *,
    max_guesses: int = DEFAULT_MAX_GUESSES,
    interactive: bool = False,
) -> int:
    """Play one game greedily with whole-word scoring; return the exit code."""
    policy = WordArgmaxPolicy(model, words)
    prefix = encode(POLICY_TOKEN + GUESS_TOKEN)
    possible: tuple[str, ...] = tuple(words)
    print("Feedback: G=green (correct spot), Y=yellow (wrong spot), X=gray (absent).")
    if not interactive:
        print(f"Secret: {secret}")
    for turn in range(1, max_guesses + 1):
        guess_ids = policy(prefix)
        guess = decode(guess_ids)
        if interactive:
            feedback = _read_feedback(guess)
            if feedback is None:
                print("Stopped by player.")
                return 2
        else:
            feedback = score_guess(secret, guess)
        symbols = "".join(FEEDBACK_TO_SYMBOL[mark] for mark in feedback)
        possible = filter_answers(possible, guess, feedback)
        turns_left = max_guesses - turn
        print(
            f"Turn {turn}/{max_guesses}: {guess} -> {feedback} | "
            f"turns left {turns_left} | candidate words {len(possible)}"
        )
        if feedback == GREEN * 5:
            target = "" if interactive else f" '{secret}'"
            print(f"Solved{target} in {turn} guesses.")
            return 0
        prefix = prefix + guess_ids + encode(FEEDBACK_TOKEN + symbols + GUESS_TOKEN)
    if interactive:
        print(f"Not solved within {max_guesses} guesses.")
    else:
        print(f"Failed: '{secret}' was not solved within {max_guesses} guesses.")
    return 1


def _print_bundle(bundle: Bundle) -> None:
    provenance = bundle.manifest.get("provenance")
    variant = provenance.get("variant") if isinstance(provenance, dict) else None
    parameters = sum(parameter.numel() for parameter in bundle.model.parameters())
    label = f"variant {variant} | " if variant else ""
    print(f"Bundle: {bundle.path}")
    print(f"Model: {label}{parameters} parameters | {len(bundle.words)} dictionary words")
    print("Decoder: word-argmax (argmax joint five-letter log-probability)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wordle_gpt.demo",
        description=(
            "Play one Wordle game with the released 7.2M soft-distilled policy. "
            f"Without --bundle the {RELEASE_TAG} release asset is downloaded once "
            "into the cache directory."
        ),
    )
    parser.add_argument(
        "--secret",
        default=DEFAULT_SECRET,
        help=f"secret word to play against (default: {DEFAULT_SECRET!r}); ignored with --interactive",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="extracted release bundle directory or bundle zip archive (offline use)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="download and bundle-extraction cache (default: platform cache directory)",
    )
    parser.add_argument(
        "--asset-url",
        default=RELEASE_URL,
        help=f"release asset URL (default: the fixed {RELEASE_TAG} GitHub release asset)",
    )
    parser.add_argument(
        "--max-guesses",
        type=int,
        default=DEFAULT_MAX_GUESSES,
        help=f"guesses per game, 1-{DEFAULT_MAX_GUESSES} (default: %(default)s)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="you supply feedback for each guess instead of using a known secret",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.max_guesses <= DEFAULT_MAX_GUESSES:
        parser.error(f"--max-guesses must be between 1 and {DEFAULT_MAX_GUESSES}")
    if args.asset_url != RELEASE_URL and args.bundle is None:
        print(f"Note: using a non-default release asset: {args.asset_url}")
    try:
        cache_dir = (args.cache_dir or default_cache_dir()).expanduser()
        directory = (
            resolve_bundle(args.bundle, cache_dir)
            if args.bundle is not None
            else ensure_bundle(cache_dir, args.asset_url)
        )
        bundle = load_bundle(directory)
        _print_bundle(bundle)
        if args.interactive:
            return play_game(
                bundle.model,
                bundle.words,
                None,
                max_guesses=args.max_guesses,
                interactive=True,
            )
        secret = args.secret.strip().lower()
        if secret not in bundle.words:
            raise ValueError(
                f"secret {secret!r} is not in the bundled dictionary "
                f"({len(bundle.words)} words)"
            )
        return play_game(
            bundle.model,
            bundle.words,
            secret,
            max_guesses=args.max_guesses,
        )
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
