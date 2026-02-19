"""Tests for electron-builder.yml build configuration.

Verifies the build config ships models-manifest.json instead of
bundled GGUF model files, and only targets macOS and Linux.
"""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BUILDER_CONFIG = PROJECT_ROOT / "ui" / "electron-builder.yml"


def _load_builder_config() -> dict:
    return yaml.safe_load(BUILDER_CONFIG.read_text())


class TestBuildConfigExtraResources:
    """Build config ships manifest, not bundled models."""

    def test_no_models_directory_in_extra_resources(self):
        """models/ must NOT be in extraResources (GGUF files too large to bundle)."""
        config = _load_builder_config()
        extra = config.get("extraResources", [])
        from_paths = [entry.get("from", "") for entry in extra]
        assert not any("models/" in p for p in from_paths), (
            f"extraResources still contains models/ entry: {from_paths}"
        )

    def test_models_manifest_in_extra_resources(self):
        """models-manifest.json must ship with the app for on-demand download."""
        config = _load_builder_config()
        extra = config.get("extraResources", [])
        from_paths = [entry.get("from", "") for entry in extra]
        assert any("models-manifest.json" in p for p in from_paths), (
            f"extraResources missing models-manifest.json: {from_paths}"
        )

    def test_no_win_target(self):
        """Only macOS and Linux targets; win section must not exist."""
        config = _load_builder_config()
        assert "win" not in config, "win target should be removed"

    def test_mac_and_linux_targets_present(self):
        """mac and linux targets must remain."""
        config = _load_builder_config()
        assert "mac" in config, "mac target missing"
        assert "linux" in config, "linux target missing"
