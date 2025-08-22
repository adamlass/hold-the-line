import sys
from dataloading import speech_datasets
from dataloading.DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from dataloading.SpeechDataset import SpeechDataset

from models import get_model
from models.BaseModel import BaseModel
from models.ModelIdentifier import ModelIdentifier
import argparse
import wandb

# -------------
# defining args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tedlium")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--load_model", type=str, default=None, help="Full name of the model to load for evaluation. Formart: '[name]_[checkpoint]'. Example: best-model-42_v2_1")
parser.add_argument("--sample_n", type=int, default=None)
parser.add_argument("--model_name", type=str, default="openai/whisper-tiny")
parser.add_argument("--model_implementation", type=str, default="whisper")


# ------------
# parsing args
args = parser.parse_args()
DATASET: str = args.dataset
BATCH_SIZE: int = args.batch_size
LOAD_MODEL: str = args.load_model
SAMPLE_N: int = args.sample_n
MODEL_NAME: str = args.model_name
MODEL_IMPLEMENTATION: str = args.model_implementation

# ---------------
# processing args
load_model: ModelIdentifier = None if LOAD_MODEL is None else ModelIdentifier(LOAD_MODEL)

dataset: SpeechDataset = speech_datasets[DATASET]
assert dataset is not None, f"dataset {DATASET} not found"

model: BaseModel = get_model(MODEL_IMPLEMENTATION, MODEL_NAME)

# --------------------------
# loading and training model
wandb.login()

model.load_model()

model.train(dataset, 
            epochs=EPOCHS, 
            resume_model=load_model, 
            batch_size=BATCH_SIZE, 
            n=SAMPLE_N,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            save_total_limit=SAVE_LIMIT,
            eval_on_start=EVAL_ON_START,
            )
