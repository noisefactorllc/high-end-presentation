"""Tests for the Claude Code plugin and marketplace manifests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    return json.loads((ROOT / ".claude-plugin/plugin.json").read_text())


def _marketplace():
    return json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())


def test_plugin_manifest_required_fields():
    m = _manifest()
    assert isinstance(m["name"], str) and m["name"]
    assert isinstance(m["version"], str) and m["version"]
    assert isinstance(m["description"], str) and len(m["description"]) > 20
    assert isinstance(m["author"], dict) and m["author"].get("name")
    assert m["license"] == "MIT"
    assert isinstance(m["keywords"], list) and m["keywords"]


def test_marketplace_entry_matches_plugin():
    m = _manifest()
    entries = _marketplace()["plugins"]
    matching = [e for e in entries if e.get("name") == m["name"]]
    assert len(matching) == 1, f"marketplace must have exactly one entry for {m['name']}"
    e = matching[0]
    assert e.get("version") == m["version"], "marketplace version must match plugin.json"
    assert e.get("source") == "./"
    assert isinstance(e.get("description"), str) and e["description"]


def test_plugin_declares_existing_default_skill():
    m = _manifest()
    if "skills" in m:
        for skill in m["skills"]:
            assert (ROOT / skill / "SKILL.md").is_file(), f"missing {skill}/SKILL.md"
    else:
        skill_dir = ROOT / "skills" / m["name"]
        assert (skill_dir / "SKILL.md").is_file(), (
            "no explicit skills paths in plugin.json; the default "
            f"skills/{m['name']} directory must contain SKILL.md"
        )
