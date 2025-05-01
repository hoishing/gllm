"""GLLM - A CLI tool for generating terminal commands using Gemini LLM."""

import tomllib
from pathlib import Path


def get_version() -> str:
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)["project"]["version"]


__version__ = get_version()
