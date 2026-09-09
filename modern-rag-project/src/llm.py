"""
Provider-agnostic factory functions for chat models, vision models, and
embedding models. Swapping LLM_PROVIDER in config.py / .env is enough to
switch the whole pipeline between OpenAI and a local Ollama model.
"""
from src import config


def get_chat_llm(temperature: float = 0.0):
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file, or set "
                "LLM_PROVIDER=ollama to use a local model instead."
            )
        return ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            temperature=temperature,
            api_key=config.OPENAI_API_KEY,
        )
    elif config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def get_vision_llm(temperature: float = 0.0):
    """Chat model with image-input support, used to caption images/charts/tables."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=config.OPENAI_VISION_MODEL,
            temperature=temperature,
            api_key=config.OPENAI_API_KEY,
        )
    elif config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_VISION_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def get_embeddings():
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
    elif config.LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")
