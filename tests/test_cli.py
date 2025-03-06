"""Tests for the CLI interface of GLLM."""

import pytest
from click.testing import CliRunner
from gllm import cli


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


def test_cli_success(runner, mocker):
    """Test successful CLI command execution."""
    # Mock the core.get_command function
    mock_get_command = mocker.patch("gllm.core.get_command")
    mock_get_command.return_value = "ls -la"

    # Run the command
    result = runner.invoke(cli.main, ["list files"])

    assert result.exit_code == 0
    assert result.output.strip() == "ls -la"
    mock_get_command.assert_called_once_with(
        user_prompt="list files",
        model=cli.DEFAULT_MODEL,
        system_prompt=cli.SYSTEM_PROMPT,
    )


def test_cli_with_custom_options(runner, mocker):
    """Test CLI command with custom model and system prompt."""
    # Mock the core.get_command function
    mock_get_command = mocker.patch("gllm.core.get_command")
    mock_get_command.return_value = "find . -name '*.py'"

    # Custom options
    custom_model = "llama3-8b-8192"
    custom_prompt = "Custom system prompt"

    # Run the command with custom options
    result = runner.invoke(
        cli.main,
        [
            "find Python files",
            "--model",
            custom_model,
            "--system-prompt",
            custom_prompt,
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "find . -name '*.py'"
    mock_get_command.assert_called_once_with(
        user_prompt="find Python files",
        model=custom_model,
        system_prompt=custom_prompt,
    )


def test_cli_error_handling(runner, mocker):
    """Test CLI error handling."""
    # Mock the core.get_command function to raise an exception
    mock_get_command = mocker.patch("gllm.core.get_command")
    mock_get_command.side_effect = Exception("API Error")

    # Run the command
    result = runner.invoke(cli.main, ["list files"])

    assert result.exit_code != 0
    assert "Error: API Error" in result.output


def test_cli_no_arguments(runner):
    """Test CLI behavior with no arguments."""
    result = runner.invoke(cli.main)
    assert result.exit_code != 0
    assert "Missing argument" in result.output
