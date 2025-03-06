"""Tests for the core functionality of GLLM."""

import os
import pytest
from gllm import core


@pytest.fixture
def api_key():
    ...

def test_get_command_success(mocker):
    """Test successful command generation."""
    # Mock environment variable
    mocker.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})

    # Mock Groq client
    mock_client = mocker.patch("groq.Groq")
    mock_completion = mocker.MagicMock()
    mock_completion.choices = [
        mocker.MagicMock(message=mocker.MagicMock(content="ls -la"))
    ]
    mock_client.return_value.chat.completions.create.return_value = mock_completion

    # Test the function
    result = core.get_command("list files in current directory")
    assert result == "ls -la"

    # Verify the correct parameters were passed
    mock_client.return_value.chat.completions.create.assert_called_once_with(
        messages=[
            {
                "role": "system",
                "content": "Help the user to create a terminal command based on the user request.",
            },
            {"role": "user", "content": "list files in current directory"},
        ],
        model="llama-3.3-70b-versatile",
    )


def test_get_command_custom_model_and_prompt(mocker):
    """Test command generation with custom model and system prompt."""
    # Mock environment variable
    mocker.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})

    # Mock Groq client
    mock_client = mocker.patch("groq.Groq")
    mock_completion = mocker.MagicMock()
    mock_completion.choices = [
        mocker.MagicMock(message=mocker.MagicMock(content="find . -name '*.py'"))
    ]
    mock_client.return_value.chat.completions.create.return_value = mock_completion

    # Test the function with custom parameters
    custom_model = "mixtral-8x7b"
    custom_prompt = "Generate Python-related commands only"
    result = core.get_command(
        "find Python files", model=custom_model, system_prompt=custom_prompt
    )

    assert result == "find . -name '*.py'"
    mock_client.return_value.chat.completions.create.assert_called_once_with(
        messages=[
            {"role": "system", "content": custom_prompt},
            {"role": "user", "content": "find Python files"},
        ],
        model=custom_model,
    )


def test_get_command_missing_api_key(mocker):
    """Test behavior when API key is missing."""
    # Mock environment variable to be empty
    mocker.patch.dict(os.environ, {}, clear=True)

    with pytest.raises(Exception) as exc_info:
        core.get_command("list files")
    assert "GROQ_API_KEY" in str(exc_info.value)


def test_get_command_api_error(mocker):
    """Test behavior when API call fails."""
    # Mock environment variable
    mocker.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})

    # Mock Groq client to raise an exception
    mock_client = mocker.patch("groq.Groq")
    mock_client.return_value.chat.completions.create.side_effect = Exception(
        "API Error"
    )

    with pytest.raises(Exception) as exc_info:
        core.get_command("list files")
    assert "API Error" in str(exc_info.value)
