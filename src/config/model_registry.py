import os

import dspy

MODEL_REGISTRY = {
    "gpt_nano": lambda: dspy.LM(model="openai/gpt-5-nano"),
    "gpt_mini": lambda: dspy.LM(model="openai/gpt-5-mini"),
    "gpt_5.2": lambda: dspy.LM(
        model="openai/gpt-5.2",
        model_type="responses",
        temperature=1.0,
        max_tokens=16000,
        reasoning={"effort": "medium"},
    ),
    "ollama_gemma3_1b": lambda: dspy.LM(
        model="ollama_chat/gemma3:1b",
        api_base=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
    ),
    "ollama_medgemma_27b": lambda: dspy.LM(
        model="ollama_chat/alibayram/medgemma:27b",
        api_base=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
    ),
}


def load_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]()
