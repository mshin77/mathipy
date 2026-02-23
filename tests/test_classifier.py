"""Tests for mathipy.classifier module."""

import json

import pytest

from mathipy.classifier import VisualModelClassifier
from mathipy.visual import visual_models


class TestTextOnlyResult:
    def test_all_false(self):
        r = VisualModelClassifier.text_only_result()
        for m in visual_models:
            assert r[m] is False

    def test_primary_text_only(self):
        r = VisualModelClassifier.text_only_result()
        assert r["primary"] == "text_only"

    def test_model_count_zero(self):
        r = VisualModelClassifier.text_only_result()
        assert r["model_count"] == 0


class TestParseClassifyResponse:
    def test_valid_json(self):
        payload = {m: False for m in visual_models}
        payload["bar_graph"] = True
        payload["primary"] = "bar_graph"
        raw = json.dumps(payload)
        r = VisualModelClassifier._parse_classify_response(raw)
        assert r["bar_graph"] is True
        assert r["primary"] == "bar_graph"
        assert r["model_count"] == 1

    def test_json_with_code_fence(self):
        payload = {m: False for m in visual_models}
        payload["table"] = True
        payload["primary"] = "table"
        raw = f"```json\n{json.dumps(payload)}\n```"
        r = VisualModelClassifier._parse_classify_response(raw)
        assert r["table"] is True
        assert r["primary"] == "table"
        assert r["model_count"] == 1

    def test_unknown_primary_falls_back_to_other(self):
        payload = {m: False for m in visual_models}
        payload["primary"] = "unknown_type"
        raw = json.dumps(payload)
        r = VisualModelClassifier._parse_classify_response(raw)
        assert r["primary"] == "other"

    def test_multiple_true(self):
        payload = {m: False for m in visual_models}
        payload["bar_graph"] = True
        payload["table"] = True
        payload["primary"] = "bar_graph"
        raw = json.dumps(payload)
        r = VisualModelClassifier._parse_classify_response(raw)
        assert r["model_count"] == 2

    def test_no_json_returns_all_false(self):
        r = VisualModelClassifier._parse_classify_response("I cannot classify this image.")
        assert r["model_count"] == 0
        assert r["primary"] == "other"


class TestConstructor:
    def test_default_provider(self):
        c = VisualModelClassifier()
        assert c.provider == "gemini"

    def test_openai_provider(self):
        c = VisualModelClassifier(provider="openai")
        assert c.provider == "openai"

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            VisualModelClassifier(provider="invalid")
