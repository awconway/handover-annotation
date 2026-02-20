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


def _build_openai_model(
    model_id: str, *, reasoning_effort: str | None = None
) -> dspy.LM:
    kwargs = {
        "model": model_id,
        "timeout": _OPENAI_REQUEST_TIMEOUT_SECONDS,
    }
    if reasoning_effort is not None:
        kwargs["model_type"] = "responses"
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return dspy.LM(**kwargs)


MODEL_REGISTRY = {
    "gpt_nano": lambda reasoning_effort=None: _build_openai_model(
        "openai/gpt-5-nano", reasoning_effort=reasoning_effort
    ),
    "gpt_mini": lambda reasoning_effort=None: _build_openai_model(
        "openai/gpt-5-mini", reasoning_effort=reasoning_effort
    ),
    "gpt_5.2": lambda reasoning_effort=None: _build_openai_model(
        "openai/gpt-5.2", reasoning_effort=reasoning_effort
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


def load_model(name: str, *, reasoning_effort: str | None = None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    if reasoning_effort is None:
        return MODEL_REGISTRY[name]()

    if name.startswith("gpt_"):
        return MODEL_REGISTRY[name](reasoning_effort=reasoning_effort)

    raise ValueError(
        "reasoning_effort is only supported for OpenAI GPT models; "
        f"got model '{name}'."
    )
