from tkinter.constants import TRUE

import dspy

from checklist_task.signatures import build_predictor
from config.settings import DATA_FILE, EVAL_RESULTS_FILE, MODEL_NAME, OUTPUT_MODEL_FILE
from data.dataset import prepare_dataset
from eval.evaluator import evaluate_checklist

_, testset = prepare_dataset(DATA_FILE)

predictor = build_predictor()

lm = dspy.LM(model="openai/gpt-5.1")
dspy.settings.configure(lm=lm)
predictor.load(OUTPUT_MODEL_FILE)
for name, pred in predictor.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")
score = evaluate_checklist(predictor, testset, EVAL_RESULTS_FILE)
print("Evaluation complete. Score:", score)
