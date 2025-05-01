# GLLM

[![ruff-badge]][ruff] [![pypi-badge]][pypi-url] ![MIT] [![uv-badge]][uv]

> A CLI tool that uses Google Gemini to generate terminal commands from natural language descriptions.

## Installation

- global install using [uv]

```bash
uv tool install gllm-cli
```

## Configuration

GLLM requires a Google [Gemini API key]. You can set it up in two ways:

1. Create a `.env` file in your working directory:

   ```ini
   GEMINI_API_KEY=your-api-key-here
   ```

2. Set it as an environment variable:

   ```bash
   export GEMINI_API_KEY=your-api-key-here
   ```

## Usage

After installation, you can use the `gllm` command directly from your terminal:

```bash
# Basic usage
gllm "list all files in the current directory"

# Use a different model, default to `gemini-2.0-flash-lite`
gllm --model "gemini-2.5-flash-preview-04-17" "show disk usage"

# Customize the system prompt
gllm --system-prompt "Generate PowerShell commands" "create a new directory"
```

### Options

- `REQUEST`: Your natural language description of the command you need
- `--model`: [Gemini model] to use (default: gemini-2.0-flash-lite)
- `--system-prompt`: System prompt for the LLM

## Development

To set up the development environment:

1. Clone the repository
2. Install development dependencies:

   ```bash
   uv sync
   ```

3. Activate the development environment:

   ```bash
   source .venv/bin/activate
   ```

## Questions?

Open a [github issue]

[Gemini API key]: https://ai.google.dev/gemini-api/docs/api-key
[Gemini model]: https://ai.google.dev/gemini-api/docs/models
[github issue]: https://github.com/hoishing/gllm/issues
[MIT]: https://img.shields.io/github/license/hoishing/gllm
[pypi-badge]: https://img.shields.io/pypi/v/gllm-cli
[pypi-url]: https://pypi.org/project/gllm-cli/
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff]: https://github.com/astral-sh/ruff
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv]: https://docs.astral.sh/uv/
