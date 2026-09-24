import json

import pytest
import yaml

from geo_mini_rag import settings
from geo_mini_rag.appraisal import manifest, pass0, pass1
from geo_mini_rag.errors import UserError

POLICY = yaml.safe_load((settings.CONFIG_DIR / "policy.yaml").read_text())


@pytest.fixture
def root_dir(tmp_path, monkeypatch):
    """A small directory under settings.ROOT, so manifest paths resolve as they do in a run."""
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(settings, "MANIFEST_DIR", tmp_path / "manifests")
    return tmp_path


def inventory_to_manifest(root_dir, files: dict[str, bytes]) -> str:
    for name, content in files.items():
        p = root_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    rows = pass0.inventory(str(root_dir), POLICY)
    mid = pass0.manifest_id("policy", rows)
    settings.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with manifest.path_for(0, mid).open("w") as f:
        for r in rows:
            f.write(json.dumps({"manifest_id": mid, "pass": 0, **r.__dict__}) + "\n")
    return mid


def run_pass1(root_dir, files, tmp_path):
    mid = inventory_to_manifest(root_dir, files)
    result = pass1.run(mid, db=tmp_path / "test.duckdb", root=str(root_dir))
    return {r["path"]: r for r in result.rows}, result


def test_identical_files_collapse_to_one_keeper(root_dir, tmp_path):
    body = b"identical well report content " * 50
    rows, result = run_pass1(root_dir, {
        "reports/final.txt": body,
        "backup/deep/copy/final.txt": body,
        "other.txt": b"something else entirely",
    }, tmp_path)
    assert result.duplicates == 1
    kept = [r for r in rows.values() if r["verdict"] == "PENDING"]
    excluded = [r for r in rows.values() if r.get("reason") == "exact_duplicate"]
    assert len(excluded) == 1
    assert excluded[0]["dup_of"].endswith("reports/final.txt"), "shallowest path is the keeper"
    assert excluded[0]["path"].endswith("backup/deep/copy/final.txt")
    assert len(kept) == 2


def test_same_size_different_content_is_not_a_duplicate(root_dir, tmp_path):
    rows, result = run_pass1(root_dir, {
        "a.txt": b"A" * 4096,
        "b.txt": b"A" * 4095 + b"B",
    }, tmp_path)
    assert result.duplicates == 0
    assert all(r["verdict"] == "PENDING" for r in rows.values())
    # Both fit inside prefilter_bytes, so the cheap key already separates them
    # and neither is ever read a second time.
    assert result.hashed == 0


def test_unique_sizes_are_never_read(root_dir, tmp_path):
    _, result = run_pass1(root_dir, {
        "a.txt": b"x" * 100,
        "b.txt": b"y" * 200,
        "c.txt": b"z" * 300,
    }, tmp_path)
    assert result.hashed == 0
    assert result.bytes_read == 0


def test_excluded_files_are_left_out_of_dedupe(root_dir, tmp_path):
    body = b"seismic bytes " * 100
    rows, result = run_pass1(root_dir, {"one.dlis": body, "two.dlis": body}, tmp_path)
    assert result.duplicates == 0, "pass 0 already excluded these as non-documents"
    assert all(r["reason"].startswith("not_a_document") for r in rows.values())


def test_missing_previous_manifest_is_a_user_error(tmp_path):
    with pytest.raises(UserError, match="no manifest"):
        pass1.run("nosuchmanifest", db=tmp_path / "test.duckdb")


def test_admitted_paths_exclude_what_the_passes_rejected(root_dir, tmp_path):
    body = b"identical content " * 40
    mid = inventory_to_manifest(root_dir, {
        "keep.txt": b"a real document",
        "reports/final.txt": body,
        "backup/final.txt": body,
        "survey.dlis": b"\x00" * 3000,
        ".DS_Store": b"junk",
    })
    pass1.run(mid, db=tmp_path / "test.duckdb", root=str(root_dir))
    admitted = manifest.admitted(1, mid)
    assert any(p.endswith("keep.txt") for p in admitted)
    assert sum(1 for p in admitted if p.endswith("final.txt")) == 1, "only the keeper survives"
    assert not any(p.endswith((".dlis", ".DS_Store")) for p in admitted)
