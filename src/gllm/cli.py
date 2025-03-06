"""Command line interface for GLLM."""

import click
from . import core

SYSTEM_PROMPT = "Help the user to create a macOS (not Linux) terminal command based on the user request. Only reply with the terminal command, no other text."
DEFAULT_MODEL = "llama-3.3-70b-versatile"


@click.command()
@click.argument("request")
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help="Groq model to use",
)
@click.option(
    "--system-prompt",
    default=SYSTEM_PROMPT,
    help="System prompt for the LLM",
)
def main(request: str, model: str, system_prompt: str) -> None:
    """Get terminal command suggestions using Groq LLM.

    REQUEST is your natural language description of the command you need.
    """
    try:
        response = core.get_command(
            user_prompt=request,
            model=model,
            system_prompt=system_prompt,
        )
        click.echo(response)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
