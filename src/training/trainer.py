from checklist_task.signatures import build_predictor
from config.dspy_settings import configure_dspy
from config.model_registry import load_model
from config.optimiser_registry import load_optimiser


def train(model_name: str, optimiser_name: str, trainset, valset):
    lm = load_model(model_name)
    configure_dspy(lm)

    predictor = build_predictor()

    optimiser_fn = load_optimiser(optimiser_name)
    predictor = optimiser_fn(predictor, trainset, valset)

    return predictor
