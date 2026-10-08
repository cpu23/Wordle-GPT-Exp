import hashlib
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from wordle_gpt.core.model import WordleGPT
from wordle_gpt.core.tokenizer_v2 import TOKENS
from wordle_gpt.demo import BUNDLE_FORMAT, TOKENIZER_FORMAT, load_bundle, safe_extract_zip
from wordle_gpt.demo import verify_bundle_files


def tiny_state_dict():
    model = WordleGPT(
        vocab_size=35,
        context_length=12,
        embedding_size=12,
        num_layers=1,
        num_heads=3,
        mlp_size=24,
    )
    return model


def write_bundle(
    directory: Path,
    *,
    tokens=tuple(TOKENS),
    architecture=None,
    words="colon\nstare\n",
    corrupt_model=False,
):
    model = tiny_state_dict()
    torch.save(
        {
            "format": "wordle-gpt-model/1",
            "vocabulary_size": 35,
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
        },
        directory / "model.pt",
    )
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "format": TOKENIZER_FORMAT,
                "vocabulary_size": len(tokens),
                "tokens": list(tokens),
            }
        ),
        encoding="utf-8",
    )
    (directory / "words.txt").write_text(words, encoding="utf-8")
    files = {
        name: {
            "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            "bytes": (directory / name).stat().st_size,
        }
        for name in ("model.pt", "tokenizer.json", "words.txt")
    }
    if corrupt_model:
        (directory / "model.pt").write_bytes((directory / "model.pt").read_bytes() + b"x")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": BUNDLE_FORMAT,
                "model": {"architecture": architecture or model.config},
                "files": files,
            }
        ),
        encoding="utf-8",
    )


class SafeExtractTests(unittest.TestCase):
    def test_round_trip_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bundle.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("manifest.json", "{}")
            target = root / "out"
            safe_extract_zip(archive, target)
            self.assertEqual((target / "manifest.json").read_text(), "{}")

            for name in ("../escape.txt", "/absolute.txt", "a/../../escape.txt"):
                with self.subTest(name=name):
                    bad = root / "bad.zip"
                    with zipfile.ZipFile(bad, "w") as handle:
                        handle.writestr(name, "x")
                    with self.assertRaises(ValueError):
                        safe_extract_zip(bad, root / "bad-out")

    def test_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bundle.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                info = zipfile.ZipInfo("link")
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                handle.writestr(info, "target")
            with self.assertRaises(ValueError):
                safe_extract_zip(archive, root / "out")


class BundleVerificationTests(unittest.TestCase):
    def test_load_bundle_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(directory)
            bundle = load_bundle(directory)
            self.assertEqual(bundle.words, ("colon", "stare"))
            self.assertEqual(bundle.model.config["vocab_size"], 35)

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(directory, corrupt_model=True)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_bundle_files(directory)

    def test_missing_bundle_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(directory)
            (directory / "words.txt").unlink()
            with self.assertRaisesRegex(ValueError, "missing words.txt"):
                verify_bundle_files(directory)

    def test_tokenizer_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_bundle(directory, tokens=tuple(TOKENS[:-1]) + ("<Q>",))
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                load_bundle(directory)

    def test_architecture_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            architecture = dict(tiny_state_dict().config)
            architecture["embedding_size"] = 24
            write_bundle(directory, architecture=architecture)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_bundle(directory)


if __name__ == "__main__":
    unittest.main()
