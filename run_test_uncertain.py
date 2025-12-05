import dspy

from config.settings import DATA_FILE, EVAL_RESULTS_FILE, OUTPUT_MODEL_FILE
from data.dataset import prepare_dataset
from eval.evaluator import evaluate
from uncertain_span_task.signatures import build_predictor

lm = dspy.LM(model="openai/gpt-5-nano")
dspy.settings.configure(lm=lm)

_, testset = prepare_dataset(DATA_FILE)

predictor = build_predictor()
case = testset[17]
print(f"text: {case.text}")
print(predictor(text=case.text))
