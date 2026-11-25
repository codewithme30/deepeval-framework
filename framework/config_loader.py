"""Loads config/settings.yaml and config/thresholds.yaml (plus .env and env-var overrides).

Everything tunable is read from here. Tests never contain model names or magic numbers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env", override=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Config:
    """Read-only view of settings.yaml with dotted-key access: config.get('limits.max_goldens')."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"Missing '{dotted}' in config/settings.yaml")
        return value

    def path(self, dotted: str) -> Path:
        """A path setting resolved against the repository root."""
        return ROOT / self.require(dotted)

    @property
    def judge_model(self) -> str:
        return os.getenv("JUDGE_MODEL") or self.require("judge_model")

    @property
    def app_mode(self) -> str:
        mode = (os.getenv("APP_MODE") or self.require("app_mode")).lower()
        if mode not in {"stub", "live"}:
            raise ValueError(f"APP_MODE must be 'stub' or 'live', got '{mode}'")
        return mode

    @property
    def live_model(self) -> str:
        return self.require("live_model")

    @property
    def reports_dir(self) -> Path:
        path = self.path("paths.reports_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path


class Thresholds:
    """Per-metric pass thresholds from thresholds.yaml."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, name: str) -> float:
        if name not in self._data:
            raise KeyError(
                f"No threshold named '{name}' in config/thresholds.yaml. "
                f"Add it there (known: {', '.join(sorted(self._data))})."
            )
        return float(self._data[name])

    __getitem__ = get


def load_config() -> Config:
    return Config(_read_yaml(CONFIG_DIR / "settings.yaml"))


def load_thresholds() -> Thresholds:
    return Thresholds(_read_yaml(CONFIG_DIR / "thresholds.yaml"))


def openai_key_present() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
