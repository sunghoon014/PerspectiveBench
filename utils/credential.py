from pathlib import Path

from llm_clients.openai_client import OpenAILLMClient
from utils.logger import logger


def create_openai_llm_client(model_config: dict, config: dict) -> OpenAILLMClient:
    model = model_config["model"]
    provider = model_config["provider"]
    params = model_config["params"]

    logger.info(f"Model: {model}")
    logger.info(f"Params: {params}")
    logger.info(f"Provider: {provider}")
    if provider == "openrouter":
        api_key = config()["openrouter"]["api_key"]
    elif provider == "openai":
        api_key = config()["openai"]["api_key"]
    else:
        raise ValueError(f"Invalid provider: {provider}")

    llm_client = OpenAILLMClient(
        model=model, provider=provider, api_key=api_key, params=params
    )
    return llm_client
