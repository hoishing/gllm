"""Core functionality for the GLLM package."""

import os
from google.genai import Client
from google.genai.types import GenerateContentConfig


def get_command(
    user_prompt: tuple[str, ...],
    model: str,
    system_prompt: str,
    key: str | None = None,
) -> str:
    """
    Get terminal command suggestion from Gemini LLM.

    Args:
        user_prompt: The user's request for a terminal command
        model: The Gemini model to use
        system_prompt: The system prompt for the LLM

    Returns:
        str: The suggested terminal command
    """

    # Initialize gemini client
    if not (api_key := key or os.getenv("GOOGLE_API_KEY")):
        raise ValueError("No API key provided. Please set the GOOGLE_API_KEY environment variable or pass the --key option.")

    client = Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )

    return response.text
