import os
import torch
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from peft import LoraConfig, PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
data_file = "../data.jsonl"
data_save_path = "../data"
model_checkpoint = "../model/tiny_qwen_output"
model_save_path = "../model/tiny_qwen"

# Define system instruction
SYS_PROMPT = """
<system_prompt>
  <role>You are an expert AI automotive diagnostic assistant.</role>
  
  <instructions>
    You MUST follow this logic:
    1.  Analyze the user's question.
    2.  IF the question is about vehicle symptoms, follow <on_topic_rules>.
    3.  IF the question is ANYTHING ELSE, follow <off_topic_rules>.
  </instructions>

  <on_topic_rules>
    <task>Provide diagnostic steps.</task>
    <output>
    - Start by identifying the **Component** and **System** in bold.
    - Then use "## Diagnostic Steps" heading.
    - Use bullet points for the steps.
    </output>
  </on_topic_rules>
  
  <off_topic_rules>
    <task>Strictly refuse all non-vehicle questions.</task>
    <examples_to_refuse>
      - Conversational chit-chat (e.g., "how are you").
      - Trivia (e.g., "who painted the mona lisa").
      - Questions about yourself or your training.
      - Any other non-automotive topic.
    </examples_to_refuse>
    <output>Respond ONLY with one of the standard refusal answers from your training data.</output>
  </off_topic_rules>
</system_prompt>
"""

# Use seed for reproducibility
SEED = 18 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data processing & formatting
def load_data(data, save_path):
    df = pd.read_json(data, lines=True)

    # Stratify, shuffle, and split the data
    df_train, df_temp = train_test_split(df, test_size=0.2,
                                         stratify=df["type"], shuffle=True, random_state=18)

    df_val, df_test = train_test_split(df_temp, test_size=0.5,
                                       stratify=df_temp["type"], shuffle=True, random_state=18)
    
    os.makedirs(save_path, exist_ok=True)
    df_train.to_json(os.path.join(save_path, "train.jsonl"), orient="records", lines=True)
    df_val.to_json(os.path.join(save_path, "val.jsonl"), orient="records", lines=True)
    df_test.to_json(os.path.join(save_path, "test.jsonl"), orient="records", lines=True)

    # Convert lists to Hugging Face datasets
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    all_data = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})
    return all_data

all_datasets = load_data(data_file, data_save_path)

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
    dtype=torch.bfloat16,
)

model.config.use_cache = False # Disable kv cache
model.config.pretraining_tp = 1 # Ignore pre-training tensor parallelism

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define LoRA parameters
peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

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
    remove_columns=["type", "question", "answer"],
)

# Convert tokenizer into data collator for SFTTrainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_checkpoint,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    num_train_epochs=5,
    learning_rate=1e-5,
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
    train_dataset=tokenised_dataset["train"],
    eval_dataset=tokenised_dataset["val"],
    data_collator=data_collator,
)

trainer.train()
print("Fine-tuning complete.")

# Save the fine-tuned model
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

# Load and save the trained adapter onto the base model
merged_model = PeftModel.from_pretrained(base_model, model_save_path)
merged_model = merged_model.merge_and_unload()

merged_model_save_path = model_save_path + "_merged"
merged_model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)