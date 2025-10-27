import json
import random
import torch
from collections import defaultdict
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

file_name = '/home/shashwat/Topic_Modelling/check/data.jsonl'

with open(file_name, 'r') as f:
    data_qna = [json.loads(line) for line in f if line.strip()]

# Group data by type
field_type = defaultdict(list)
for record in data_qna:
    field_type[record['type']].append(record)

train_set, val_set, test_set = [], [], []

# Stratify & shuffle
for type_key, records in field_type.items():
    random.shuffle(records)
    n = len(records)
    train_end = int(0.8 * n)
    test_start = train_end + int(0.1 * n)

    train_set.extend(records[:train_end])
    val_set.extend(records[train_end:test_start])
    test_set.extend(records[test_start:])

random.shuffle(train_set)
random.shuffle(val_set)
random.shuffle(test_set)

print(len(train_set))
print(len(val_set))
print(len(test_set))

dataset = ""

# 4-bit quantisation config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "Qwen/Qwen3-4B-Instruct-2507"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             torch_dtype=torch.bfloat16,)

model.config.use_cache = False # Disable kv cache
model.config.pretraining_tp = 1 # Ignore pre-training tensor parallelism
# print(model.get_memory_footprint())

SYSTEM_PROMPT = "You are an expert."

# Convert tokenizer into data collator for SFTTrainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define LoRA parameters
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="qwen3_output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    bf16=True,
    load_best_model_at_end=True,
    report_to="none",
)

# Initialise the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=dataset, # Update
    eval_dataset=dataset, # Update
    data_collator=data_collator,
)

trainer.train()

# Save the fine-tuned model
model_save_path = "tiny_qwen"
model.save_pretrained(str(model_save_path))
tokenizer.save_pretrained(str(model_save_path))