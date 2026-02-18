import os

import dspy

_OPENAI_REQUEST_TIMEOUT_SECONDS = 120

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
        temperature=0.1,
    ),
    "ollama_medgemma_27b": lambda: dspy.LM(
        model="ollama_chat/alibayram/medgemma:27b",
        api_base=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
        temperature=0.1,
    ),
}


def load_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]()
