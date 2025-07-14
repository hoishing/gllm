import tomllib
from pathlib import Path


def pyproject_data() -> dict:
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        return tomllib.load(f)


metadata = pyproject_data()["project"]

readme_md = """\
# GLLM

[![ruff-badge]][ruff] [![pypi-badge]][pypi-url] ![MIT] [![uv-badge]][uv]

> {description}

## Installation

- global install using [uv]

```bash
uv tool install gllm-cli
```

## Configuration

GLLM requires a Google [Google API key]. You can set it up in two ways:

1. provided by `--key` option

   ```bash
   gllm --key=YOUR_GEMINI_API_KEY your prompt
   ```

2. Set it as an environment variable:

   ```bash
   export GEMINI_API_KEY=your-api-key-here
   ```

## Usage

After installation, you can use the `gllm` command directly from your terminal:

```bash
# Basic usage
gllm list all files in the current directory

# Use a different model, default to `gemini-2.5-flash`
gllm --model gemini-2.5-pro show disk usage

# Customize the system prompt
gllm --system-prompt "Generate PowerShell commands" list files in the current directory
```

### Options

- `--model`: [Gemini model] to use (default: gemini-2.5-flash)
- `--system-prompt`: System prompt for the LLM
- `--key`: your gemini api key

## Questions

- [Github issue]
- [LinkedIn]

[Gemini model]: https://ai.google.dev/gemini-api/docs/models
[Github issue]: https://github.com/hoishing/gllm/issues
[Google API key]: https://ai.google.dev/gemini-api/docs/api-key
[LinkedIn]: https://www.linkedin.com/in/kng2
[MIT]: https://img.shields.io/github/license/hoishing/gllm
[pypi-badge]: https://img.shields.io/pypi/v/gllm-cli
[pypi-url]: https://pypi.org/project/gllm-cli/
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff]: https://github.com/astral-sh/ruff
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv]: https://docs.astral.sh/uv/
"""


if __name__ == "__main__":
    readme_md = readme_md.format(description=metadata["description"])
    open("README.md", "w").write(readme_md)
