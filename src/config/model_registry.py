import dspy

MODEL_REGISTRY = {
    "gpt_nano": lambda: dspy.LM(model="openai/gpt-5-nano"),
    "gpt_mini": lambda: dspy.LM(model="openai/gpt-5-mini"),
    "gpt_5.1": lambda: dspy.LM(model="openai/gpt-5.1"),
}


def load_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]()
