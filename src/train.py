import json
import random
import torch
from collections import defaultdict
from datasets import Dataset, DatasetDict
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
data_file = "../data/data.jsonl"
data_save_path = "../data/datasets"
model_save_path = "../model/tiny_qwen"

with open(data_file, 'r') as f:
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

# Convert lists to Hugging Face datasets
train_dataset = Dataset.from_list(train_set)
val_dataset = Dataset.from_list(val_set)
test_dataset = Dataset.from_list(test_set)

all_datasets = DatasetDict({'train': train_dataset,
                            'val': val_dataset,
                            'test': test_dataset})
all_datasets.save_to_disk(data_save_path)

# 4-bit quantisation config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False # Disable kv cache
model.config.pretraining_tp = 1 # Ignore pre-training tensor parallelism
# print(model.get_memory_footprint())

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define system instruction
SYS_PROMPT = """
### Instruction: You are an expert AI automotive diagnostic assistant.
### Goal: Your primary goal is to analyse vehicle symptoms and provide clear, concise diagnostic steps.
### Output: Give a structured output in a markdown format using headings and bullet points.
### Constraints: If the question is not related to vehicle diagnostics, politely refuse to answer.
"""

# Pre-processing function
def apply_chat_template(example, tokenizer, max_length=1024):
    """
    Takes each example from the given dataset and applies the chat template.
    Adds system/user/assistant content and tokenises the result.
    """
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenised_output = tokenizer(
        formatted_input,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    return tokenised_output

# Apply the chat template to the selected dataset
tokenised_dataset = all_datasets.map(
    lambda ex: apply_chat_template(ex, tokenizer),
    remove_columns=['type', 'question', 'answer'],
)

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
    train_dataset=tokenised_dataset['train'],
    eval_dataset=tokenised_dataset['val'],
    data_collator=data_collator,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)