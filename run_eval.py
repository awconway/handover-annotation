from checklist_task.signatures import build_predictor
from config.settings import DATA_FILE, EVAL_RESULTS_FILE, OUTPUT_MODEL_FILE
from data.dataset import prepare_dataset
from eval.evaluator import evaluate

_, testset = prepare_dataset(DATA_FILE)

predictor = build_predictor()
predictor.load(OUTPUT_MODEL_FILE)

score = evaluate(predictor, testset, EVAL_RESULTS_FILE)
print(predictor.inspect_history(-1))
print("Evaluation complete. Score:", score)
