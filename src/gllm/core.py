"""Core functionality for the GLLM package."""

import os
from dotenv import load_dotenv
from groq import Groq


def get_command(
    user_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    system_prompt: str = "Help the user to create a terminal command based on the user request.",
) -> str:
    """
    Get terminal command suggestion from Groq LLM.

    Args:
        user_prompt: The user's request for a terminal command
        model: The Groq model to use
        system_prompt: The system prompt for the LLM

    Returns:
        str: The suggested terminal command
    """
    # Load environment variables
    load_dotenv()

    # Initialize Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Create chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content
