"""Model resolution + hot-reload tests."""
import json

import pytest

from gateway import config, models


@pytest.fixture
def models_file(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "default": "sonnet",
        "aliases": {"gpt-4o": "sonnet", "gpt-4o-mini": "haiku", "gemini-1.5-pro": "opus"},
        "passthrough_prefixes": ["claude-"],
    }))
    monkeypatch.setattr(config, "MODELS_FILE", str(path))
    monkeypatch.setattr(config, "DEFAULT_MODEL", "")
    # reset the module cache so the patched path is picked up
    models._cache.update(mtime=None, path=None, data=None)
    return path


def test_passthrough_prefix(models_file):
    assert models.resolve_model("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert models.resolve_model("claude-3-opus-latest") == "claude-3-opus-latest"


def test_alias_mapping(models_file):
    assert models.resolve_model("gpt-4o") == "sonnet"
    assert models.resolve_model("gpt-4o-mini") == "haiku"
    assert models.resolve_model("gemini-1.5-pro") == "opus"


def test_unknown_falls_back_to_default(models_file):
    assert models.resolve_model("totally-made-up") == "sonnet"
    assert models.resolve_model("") == "sonnet"


def test_default_model_env_override(models_file, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_MODEL", "opus")
    assert models.resolve_model("unknown") == "opus"


def test_hot_reload_on_mtime_change(models_file):
    assert models.resolve_model("gpt-4o") == "sonnet"
    models_file.write_text(json.dumps({
        "default": "haiku", "aliases": {"gpt-4o": "opus"}, "passthrough_prefixes": ["claude-"],
    }))
    # bump mtime to ensure the change is detected even on coarse clocks
    import os
    st = models_file.stat()
    os.utime(models_file, (st.st_atime + 5, st.st_mtime + 5))
    assert models.resolve_model("gpt-4o") == "opus"
    assert models.resolve_model("unknown") == "haiku"


def test_list_ids_includes_aliases_and_canonical(models_file):
    ids = models.list_model_ids()
    assert "gpt-4o" in ids
    for canonical in ("sonnet", "opus", "haiku"):
        assert canonical in ids


def test_openai_payload_shape(models_file):
    payload = models.openai_models_payload()
    assert payload["object"] == "list"
    assert all(m["object"] == "model" and m["owned_by"] == "claude-gateway" for m in payload["data"])


def test_gemini_payload_shape(models_file):
    payload = models.gemini_models_payload()
    assert all(m["name"].startswith("models/") for m in payload["models"])
    assert all("generateContent" in m["supportedGenerationMethods"] for m in payload["models"])
