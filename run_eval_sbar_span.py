import dspy

from config.settings import DATA_FILE, EVAL_RESULTS_FILE, MODEL_NAME, OUTPUT_MODEL_FILE
from data.dataset import prepare_dataset_sbar_span
from eval.evaluator import evaluate_sbar
from sbar_span_task.signatures import build_predictor

_, testset = prepare_dataset_sbar_span(DATA_FILE)

predictor = build_predictor()

lm = dspy.LM(model="openai/gpt-5-nano")
dspy.settings.configure(lm=lm)
# predictor.load(OUTPUT_MODEL_FILE)

score = evaluate_sbar(predictor, testset, EVAL_RESULTS_FILE)
print(predictor.inspect_history(-1))
print("Evaluation complete. Score:", score)
