import time
from typing import Any


def call_api(client: Any, model_name: str, messages: list[dict], max_retries: int = 8) -> str:
    """Dispatch a chat completion call to the appropriate provider client.

    Provider is detected by inspecting the client's module name, so no explicit
    provider parameter is needed.

    Args:
        client: Initialized API client (``anthropic.Anthropic`` or ``openai.OpenAI``).
        model_name: Model ID to use for the completion.
        messages: List of chat messages with ``role`` and ``content`` keys.

    Returns:
        Raw text response from the model.

    Raises:
        ValueError: If the client's provider is not supported.
    """
    provider = type(client).__module__.split(".")[0]

    for attempt in range(max_retries):
        try:
            return _call_api_once(client, provider, model_name, messages)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise

    return _call_api_once(client, provider, model_name, messages)


def _call_api_once(client: Any, provider: str, model_name: str, messages: list[dict]) -> str:
    if provider == "anthropic":
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        kwargs = {"model": model_name, "max_tokens": 5, "messages": user_messages}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()
    elif provider == "openai":
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=5,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    else:
        raise ValueError(
            f"Unsupported API client: '{provider}'. Expected 'anthropic' or 'openai'."
        )
