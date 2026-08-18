from importlib.resources import files
from pathlib import Path

import yaml

from distill_codec.integrations.snapshots import default_snapshot_path


def test_default_snapshot_sources_are_packaged_and_match_audit_manifests():
    package_root = files("distill_codec.vendor")
    for project, filename in (("wan", "wan_video_vae.py"), ("flashvsr", "utils.py"), ("flashvsr", "TCDecoder.py")):
        packaged = package_root.joinpath(project, filename)
        assert packaged.is_file()
        assert Path(default_snapshot_path(project, filename)).read_bytes() == packaged.read_bytes()
        manifest = yaml.safe_load(Path("third_party", project, "manifest.yaml").read_text(encoding="utf-8"))
        expected = next(entry["sha256"] for entry in manifest["files"] if entry["path"] == filename)
        import hashlib

        assert hashlib.sha256(packaged.read_bytes()).hexdigest() == expected
