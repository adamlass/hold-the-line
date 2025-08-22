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
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--resume_model", type=str, default=None, help="Full name of the model to resume training from. Formart: '[name]_[checkpoint]'. Example: best-model-42_v2_1")
parser.add_argument("--sample_n", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps.")
parser.add_argument("--model_name", type=str, default="openai/whisper-tiny")
parser.add_argument("--model_implementation", type=str, default="whisper")
parser.add_argument("--save_steps", type=int, default=None)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--logging_steps", type=int, default=25)
parser.add_argument("--save_limit", type=int, default=5)
parser.add_argument("--eval_on_start", type=bool, default=True)


# ------------
# parsing args
args = parser.parse_args()
DATASET: str = args.dataset
EPOCHS: int = args.epochs
BATCH_SIZE: int = args.batch_size
RESUME_MODEL: str = args.resume_model
SAMPLE_N: int = args.sample_n
LEARNING_RATE: float = args.learning_rate
WARMUP_STEPS: int = args.warmup_steps
MODEL_NAME: str = args.model_name
MODEL_IMPLEMENTATION: str = args.model_implementation
EVAL_STEPS: int = args.eval_steps
SAVE_STEPS: int = args.save_steps or EVAL_STEPS
LOGGING_STEPS: int = args.logging_steps
SAVE_LIMIT: int = args.save_limit
EVAL_ON_START: bool = args.eval_on_start

# ---------------
# processing args
resume_model: ModelIdentifier = None if RESUME_MODEL is None else ModelIdentifier(RESUME_MODEL)

dataset: SpeechDataset = speech_datasets[DATASET]
assert dataset is not None, f"dataset {DATASET} not found"

model: BaseModel = get_model(MODEL_IMPLEMENTATION, MODEL_NAME)

# --------------------------
# loading and training model
wandb.login()

model.load_model()

model.train(dataset, 
            epochs=EPOCHS, 
            resume_model=resume_model, 
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
