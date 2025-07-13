"""Command line interface for GLLM."""

import click
from . import __version__, core

SYSTEM_PROMPT = "Help the user to create a macOS (not Linux) terminal command based on the user request. Only reply with the terminal command, no other text."
DEFAULT_MODEL = "gemini-2.5-flash"


def version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Show version and exit."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"gllm {__version__}")
    ctx.exit()


@click.command()
@click.argument("requests", nargs=-1, required=True)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help="Gemini model to use",
)
@click.option(
    "--system-prompt",
    default=SYSTEM_PROMPT,
    help="System prompt for the LLM",
)
@click.option(
    "--version",
    is_flag=True,
    callback=version_callback,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
@click.option(
    "--key",
    help="Gemini API key",
)
def main(
    requests: tuple[str, ...], model: str, system_prompt: str, key: str | None = None
) -> None:
    """Get terminal command suggestions using Google Gemini.

    REQUESTS: natural language descriptions of the command

        e.g. `gllm show disk usage`
    """
    try:
        # print(f"{requests=}")
        response = core.get_command(
            user_prompt=requests,
            model=model,
            system_prompt=system_prompt,
            key=key,
        )
        click.echo(response)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
