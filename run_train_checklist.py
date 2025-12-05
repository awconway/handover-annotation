from config.settings import DATA_FILE, MODEL_NAME, OPTIMISER_NAME, OUTPUT_MODEL_FILE
from data.dataset import prepare_dataset
from training.trainer import train

trainset, valset = prepare_dataset(DATA_FILE)

predictor = train(MODEL_NAME, OPTIMISER_NAME, trainset, valset)
predictor.save(OUTPUT_MODEL_FILE)
for name, pred in predictor.named_predictors():
    print("================================")
    print(f"Predictor: {name}")
    print("================================")
    print("Prompt:")
    print(pred.signature.instructions)
    print("*********************************")

print("Training complete. Saved to", OUTPUT_MODEL_FILE)
