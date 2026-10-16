"""The three scientific inputs must remain byte-identical to the user's originals."""
import hashlib
import json

import pytest


@pytest.mark.parametrize("name", ["training", "external_validation", "canonical_notebook"])
def test_scientific_original_hashes(repo_root, name):
    manifest = json.loads((repo_root / "docs" / "data_provenance.json").read_text())
    entry = manifest[name]
    path = repo_root / entry["path"]
    if not path.exists() and name != "training":
        pytest.skip(f"Private original is deliberately not published: {entry['path']}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
