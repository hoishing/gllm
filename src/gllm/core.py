"""Core functionality for the GLLM package."""

import os
from dotenv import load_dotenv
from google.genai import Client, types


def get_command(
    user_prompt: str,
    model: str,
    system_prompt: str,
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
    # Load environment variables
    load_dotenv()

    # Initialize gemini client
    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )

    return response.text
