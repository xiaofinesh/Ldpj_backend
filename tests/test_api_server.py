"""Tests for integration.api_server auth (X-API-Key verification)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import HTTPException

from integration import api_server


@pytest.fixture
def restore_key():
    """Save/restore the module-global api_key around each test."""
    saved = api_server._refs.get("api_key")
    yield
    api_server._refs["api_key"] = saved


class TestVerifyKey:
    def test_correct_key_passes(self, restore_key):
        api_server._refs["api_key"] = "secret"
        assert api_server._verify_key("secret") == "secret"

    def test_wrong_key_rejected(self, restore_key):
        api_server._refs["api_key"] = "secret"
        with pytest.raises(HTTPException) as ei:
            api_server._verify_key("nope")
        assert ei.value.status_code == 403

    def test_missing_key_rejected(self, restore_key):
        api_server._refs["api_key"] = "secret"
        with pytest.raises(HTTPException):
            api_server._verify_key(None)

    def test_null_configured_key_fails_closed(self, restore_key):
        """A null/empty configured api_key must NOT become an auth bypass."""
        api_server._refs["api_key"] = None
        with pytest.raises(HTTPException):
            api_server._verify_key(None)        # would be None==None pre-fix
        with pytest.raises(HTTPException):
            api_server._verify_key("anything")

    def test_empty_string_key_fails_closed(self, restore_key):
        api_server._refs["api_key"] = ""
        with pytest.raises(HTTPException):
            api_server._verify_key("")

    def test_non_ascii_key_does_not_500(self, restore_key):
        """A non-ASCII configured key must compare via bytes, not raise
        TypeError from compare_digest (which would be HTTP 500 = auth wedge)."""
        api_server._refs["api_key"] = "密钥secret"
        assert api_server._verify_key("密钥secret") == "密钥secret"
        with pytest.raises(HTTPException):
            api_server._verify_key("wrong")
