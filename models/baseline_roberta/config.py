from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = MODEL_DIR / 'data' / 'Sarcasm_Headlines_Dataset_v2.json'

# Tokenizer parameters
RANDOM_STATE = 48
MAX_LENGTH = 128
MAX_LENGTH_CONTEXT = 256

# Train, test, split parameters
TEST_SIZE = 0.2
TEST_VAL_SIZE = 0.5

# Model parameters
MODEL_NAME = 'roberta-base'
NUM_LABELS = 2
LEARNING_RATE = 1.4054021095969553e-05
TRAIN_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 5
NUM_WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1616983040263596
LABEL_SMOOTHING = 0.1

# Evaluation parameters
EVAL_BATCH_SIZE = 32

# HF Hub
HF_OUTPUT_DIR = 'roberta-sarcasm-detection-finetuned-v3'
REPO_ID = 'kiankiat/roberta-sarcasm-detection-finetuned-v3'

# Optuna HFT
NUM_TRIALS = 10
NUM_STARTUP_TRIALS = 5
NUM_WARMUP_STEPS = 2