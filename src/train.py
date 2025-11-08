import os
import torch
import random
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
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
)
from trl import SFTTrainer

def set_seed(seed):
    """
    Sets seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_data(data_file, save_path, random_state):
    """
    Loads, stratifies, shuffles, and splits the JSONL dataset.
    """
    df = pd.read_json(data_file, lines=True)

    # Stratify, shuffle, and split the data
    df_train, df_temp = train_test_split(
        df,
        test_size=0.2,
        stratify=df["type"],
        shuffle=True,
        random_state=random_state,
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp["type"],
        shuffle=True,
        random_state=random_state,
    )
    
    # Save data splits
    os.makedirs(save_path, exist_ok=True)
    df_train.to_json(os.path.join(save_path, "train.jsonl"), orient="records", lines=True)
    df_val.to_json(os.path.join(save_path, "val.jsonl"), orient="records", lines=True)
    df_test.to_json(os.path.join(save_path, "test.jsonl"), orient="records", lines=True)

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    all_data = DatasetDict({"train": train_dataset,
                            "val": val_dataset,
                            "test": test_dataset})
    return all_data

def apply_chat_template(sys_prompt, example, tokenizer, max_length=1024):
    """
    Takes each example from the given dataset and applies the chat template.
    """
    messages = [
        {"role": "system", "content": sys_prompt},
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

def create_bnb_config():
    """
    Creates 4-bit quantisation configuration.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def create_peft_config():
    """
    Creates LoRA configuration.
    """
    return LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.15,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

def load_model_and_tokenizer(model_id, bnb_config, **model_kwargs):
    """
    Loads model and tokeniser with quantisation.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        **model_kwargs,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id, **model_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_model_for_peft(model, peft_config):
    """
    Prepares model for PEFT training.
    """
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def create_train_args(output_dir):
    """
    Defines training arguments.
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_train_epochs=6,
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

def train_sft(model, tokenizer, training_args, train_dataset, eval_dataset, data_collator):
    """
    Trains the model using SFTTrainer.
    """
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    return trainer

def merge_model(trainer, tokenizer, merged_model_path):
    """
    Merges the trained adapter onto the base model.
    """
    model = trainer.model.merge_and_unload()
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model saved at: {merged_model_path}")