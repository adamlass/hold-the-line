import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import IterativeSFTTrainer
import torch
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from dotenv import load_dotenv
from huggingface_hub import login
import os
# from accelerate import PartialState
# device_string = PartialState().process_index
# print("device_string:", device_string)


load_dotenv()

access_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
login(token=access_token)

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
    # device_map={'':device_string}
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
)

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")

dataset = dataset.map(apply_template, batched=True)

trainer=IterativeSFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    # train_dataset=dataset,
    # dataset_text_field="text",
    # max_seq_length=max_seq_length,
    # dataset_num_proc=2,
    # packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

print("Starting training")
# trainer.train()

tokenizer.decode

