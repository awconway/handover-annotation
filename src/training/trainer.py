import dspy

from checklist_task.signatures import build_predictor
from config.model_registry import load_model
from config.optimiser_registry import load_optimiser


def train(model_name: str, optimiser_name: str, trainset, valset):
    lm = load_model(model_name)
    dspy.settings.configure(lm=lm)

    predictor = build_predictor()

    optimiser_fn = load_optimiser(optimiser_name)
    predictor = optimiser_fn(predictor, trainset, valset)

    return predictor
