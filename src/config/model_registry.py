import os

import dspy

_OPENAI_REQUEST_TIMEOUT_SECONDS = 120


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw}") from exc


MODEL_REGISTRY = {
    "gpt_nano": lambda: dspy.LM(
        model="openai/gpt-5-nano", timeout=_OPENAI_REQUEST_TIMEOUT_SECONDS
    ),
    "gpt_mini": lambda: dspy.LM(
        model="openai/gpt-5-mini", timeout=_OPENAI_REQUEST_TIMEOUT_SECONDS
    ),
    "gpt_5.2": lambda: dspy.LM(
        model="openai/gpt-5.2", timeout=_OPENAI_REQUEST_TIMEOUT_SECONDS
    ),
    "ollama_gemma3_1b": lambda: dspy.LM(
        model="ollama_chat/gemma3:1b",
        api_base=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
        temperature=_env_float("OLLAMA_GEMMA3_1B_TEMPERATURE", 0.1),
    ),
    "ollama_medgemma_27b": lambda: dspy.LM(
        model="ollama_chat/alibayram/medgemma:27b",
        api_base=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
        temperature=_env_float("OLLAMA_MEDGEMMA_27B_TEMPERATURE", 0.1),
    ),
}


def load_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]()
