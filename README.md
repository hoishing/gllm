# GLLM

[![ruff-badge]][ruff] [![pypi-badge]][pypi-url] ![MIT] [![uv-badge]][uv]

> A CLI tool that uses [Groq] LLM to generate terminal commands from natural language descriptions.

## Installation

- global install using [uv]

```bash
uv tool install gllm-cli
```

## Configuration

GLLM requires a Groq API key. You can set it up in two ways:

1. Create a `.env` file in your working directory:

   ```ini
   GROQ_API_KEY=your-api-key-here
   ```

2. Set it as an environment variable:

   ```bash
   export GROQ_API_KEY=your-api-key-here
   ```

## Usage

After installation, you can use the `gllm` command directly from your terminal:

```bash
# Basic usage
gllm "list all files in the current directory"

# Use a different model
gllm --model "llama-3.3-70b-versatile" "show disk usage"

# Customize the system prompt
gllm --system-prompt "Generate PowerShell commands" "create a new directory"
```

### Options

- `REQUEST`: Your natural language description of the command you need
- `--model`: [Groq model] to use (default: llama-3.3-70b-versatile)
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

Open a [github issue] or ping me on [X]

[github issue]: https://github.com/hoishing/gllm/issues
[Groq model]: https://console.groq.com/docs/models
[Groq]: https://console.groq.com/docs
[MIT]: https://img.shields.io/github/license/hoishing/gllm
[pypi-badge]: https://img.shields.io/pypi/v/gllm-cli
[pypi-url]: https://pypi.org/project/gllm-cli/
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff]: https://github.com/astral-sh/ruff
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv]: https://docs.astral.sh/uv/
[X]: https://x.com/intent/tweet?text=https://github.com/hoishing/gllm/%20%0D@hoishing
