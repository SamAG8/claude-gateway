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


# ---- per-model effort (models.json "effort" map) ------------------------

@pytest.fixture
def effort_models_file(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "default": "sonnet",
        "aliases": {"gemini-3.1-flash-lite": "haiku"},
        "effort": {"haiku": "low"},
        "passthrough_prefixes": ["claude-"],
    }))
    monkeypatch.setattr(config, "MODELS_FILE", str(path))
    models._cache.update(mtime=None, path=None, data=None)
    return path


def test_effort_per_model_overrides_global(effort_models_file, monkeypatch):
    monkeypatch.setattr(config, "EFFORT", "high")
    assert models.resolve_effort("haiku") == "low"   # per-model wins
    assert models.resolve_effort("opus") == "high"   # others keep the global


def test_effort_none_when_unset(effort_models_file, monkeypatch):
    monkeypatch.setattr(config, "EFFORT", "")
    assert models.resolve_effort("opus") is None      # CLI default
    assert models.resolve_effort("haiku") == "low"    # still overridden


def test_real_config_routes_fast_tier_to_haiku():
    """The shipped models.json routes the ConstraAP fast tier to haiku at low effort
    while the complex (pro) path stays on opus — issue #11 latency follow-up."""
    from pathlib import Path
    data = json.loads(Path("models.json").read_text())
    assert data["aliases"]["gemini-3.1-flash-lite"] == "haiku"
    assert data["aliases"]["gemini-3.1-flash"] == "haiku"
    assert data["effort"]["haiku"] == "low"
    assert data["aliases"]["gemini-3.1-pro-preview"] == "opus"


def test_is_fast_model_default(models_file):
    """Absent a fast_models list, haiku is the fast tier; heavier models are not."""
    assert models.is_fast_model("haiku") is True
    assert models.is_fast_model("sonnet") is False
    assert models.is_fast_model("opus") is False
    assert models.is_fast_model("claude-haiku-4-5-20251001") is True
    assert models.is_fast_model("claude-sonnet-5") is False


def test_dated_model_ids_inherit_family_effort_and_thinking(effort_models_file, monkeypatch):
    monkeypatch.setattr(config, "EFFORT", "high")
    data = json.loads(effort_models_file.read_text())
    data["max_thinking_tokens"] = {"haiku": 0}
    effort_models_file.write_text(json.dumps(data))
    import os
    st = effort_models_file.stat()
    os.utime(effort_models_file, (st.st_atime + 5, st.st_mtime + 5))
    assert models.model_tier("claude-haiku-4-5-20251001") == "haiku"
    assert models.resolve_effort("claude-haiku-4-5-20251001") == "low"
    assert models.resolve_max_thinking_tokens("claude-haiku-4-5-20251001") == 0
    assert models.resolve_effort("claude-opus-4-8") == "high"


def test_is_fast_model_configurable(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "default": "sonnet", "aliases": {},
        "fast_models": ["haiku", "sonnet"],
        "passthrough_prefixes": ["claude-"],
    }))
    monkeypatch.setattr(config, "MODELS_FILE", str(path))
    models._cache.update(mtime=None, path=None, data=None)
    assert models.is_fast_model("sonnet") is True
    assert models.is_fast_model("opus") is False
