import os

import requests

from utils import get_logger

logger = get_logger(__name__)

PROVIDERS = [
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
        ],
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": [
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "openai/gpt-oss-20b:free",
        ],
    },
    {
        "name": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key_env": "GEMINI_API_KEY",
        "models": [
            "gemini-flash-latest",
            "gemini-2.0-flash",
        ],
    },
]


def _local_chat(messages, temperature, max_tokens):
    from local_models import ensure_llm
    from llama_cpp import Llama

    model_path = ensure_llm()
    logger.info(f"Loading local LLM: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)

    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"

    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|user|>", "<|system|>", "<|assistant|>"],
        echo=False,
    )
    content = response["choices"][0]["text"].strip()
    if not content:
        raise RuntimeError("local LLM returned empty response")
    logger.info("Success via local LLM")
    return content


def chat_completion(messages, temperature: float = 0.8, max_tokens: int = 2000, providers=None) -> str:
    providers = providers if providers is not None else PROVIDERS
    errors = []

    for provider in providers:
        api_key = os.environ.get(provider["api_key_env"])
        if not api_key:
            logger.warning(f"Skipping {provider['name']}: {provider['api_key_env']} not set")
            continue

        for model in provider["models"]:
            try:
                logger.info(f"Trying {provider['name']}/{model}...")
                response = requests.post(
                    provider["base_url"],
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=60,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise RuntimeError("empty response")

                logger.info(f"Success via {provider['name']}/{model}")
                return content
            except Exception as exc:
                logger.warning(f"{provider['name']}/{model} failed: {exc}")
                errors.append(f"{provider['name']}/{model}: {exc}")
                continue

    try:
        return _local_chat(messages, temperature, max_tokens)
    except Exception as exc:
        errors.append(f"local_llm: {exc}")

    raise RuntimeError("All text-generation providers/models failed:\n" + "\n".join(errors))
