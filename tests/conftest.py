# """Shared test fixtures for GLLM tests."""

# import pytest
# from click.testing import CliRunner


# @pytest.fixture
# def mock_env(mocker):
#     """Mock environment variables."""
#     return mocker.patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})


# @pytest.fixture
# def mock_groq_client(mocker):
#     """Mock Groq client with default success response."""
#     mock_client = mocker.patch("groq.Groq")
#     mock_completion = mocker.MagicMock()
#     mock_completion.choices = [
#         mocker.MagicMock(message=mocker.MagicMock(content="test command"))
#     ]
#     mock_client.return_value.chat.completions.create.return_value = mock_completion
#     return mock_client


# @pytest.fixture
# def cli_runner():
#     """Provide a CLI runner for testing."""
#     return CliRunner()
