import hashlib
from pathlib import Path

import yaml


def _verify_snapshot(name):
    root = Path("third_party") / name
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["repository"].startswith("https://")
    assert len(manifest["revision"]) == 40
    assert manifest["retrieved"] == "2026-08-18"
    assert manifest["license"]
    assert (root / manifest["license"]).is_file()
    assert manifest["known_limitations"]
    assert manifest["files"]
    for entry in manifest["files"]:
        path = root / entry["path"]
        assert path.is_file(), path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"]


def test_wan_snapshot_has_auditable_provenance():
    _verify_snapshot("wan")


def test_flashvsr_snapshot_has_auditable_provenance():
    _verify_snapshot("flashvsr")


def test_third_party_policy_excludes_model_weights():
    forbidden = {".pt", ".pth", ".ckpt", ".safetensors"}
    tracked = [path for path in Path("third_party").rglob("*") if path.is_file()]
    assert tracked
    assert not [path for path in tracked if path.suffix.lower() in forbidden]

