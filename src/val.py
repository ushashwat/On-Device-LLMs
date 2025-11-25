"""val.py script for llm testing."""
import random
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

def set_seed(seed: int) -> None:
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_test_data(test_file: str) -> Dataset:
    """Loads test data from JSONL file."""
    df_test = pd.read_json(test_file, lines=True)
    data_test = Dataset.from_pandas(df_test)
    return data_test

def apply_chat_template(
        sys_prompt: str,
        example: dict[str, str],
        tokenizer: PreTrainedTokenizerBase,
        model_name: str,
    ) -> dict[str, str]:
    """Applies the chat template to a batch of samples from the test set."""
    if model_name == "gemma":
        messages = [{"role": "user", "content": f"{sys_prompt}\n{example['question']}"}]
    else:
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": example["question"]}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt}

def load_model_and_tokenizer(
        model_path: str,
        **model_kwargs,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads merged model and tokeniser."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        **model_kwargs,
    )
    return model, tokenizer

def generate_preds(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompts_list: list[str],
    ) -> list[str]:
    """Generates predictions using model pipeline."""
    generation_kwargs = {
        "max_new_tokens": 256,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False,
    }

    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    outputs = pipe(prompts_list, **generation_kwargs, batch_size=32)
    preds = [output[0]["generated_text"].strip() for output in outputs]
    return preds

def save_eval_results(data_test: Dataset, predictions: list[str], results_file: str) -> None:
    """Saves evaluation results to a CSV file."""
    df_results = pd.DataFrame({
        "question": data_test["question"],
        "answer_ref": data_test["answer"],
        "answer_pred": predictions
    })
    df_results.to_csv(results_file, index=False)
