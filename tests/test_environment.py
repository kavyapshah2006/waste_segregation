"""Installed versions must match the pins in pyproject.toml."""

import tomllib
from importlib.metadata import version
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pins() -> dict[str, str]:
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    return dict(dep.split("==") for dep in deps if "==" in dep)


def test_pinned_versions_installed() -> None:
    for name, pinned in _pins().items():
        # Local version labels such as "+cpu" are ignored.
        assert version(name).split("+")[0] == pinned, name
