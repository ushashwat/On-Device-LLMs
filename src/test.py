import torch
import pandas as pd
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

model_id = "Qwen/Qwen3-4B-Instruct-2507"
test_file = "../data/test.jsonl"
model_path = "../model/tiny_qwen"
results_path = "../data/evaluation.csv"

# Define system instruction
SYS_PROMPT = """
### Instruction: You are an expert AI automotive diagnostic assistant.
### Goal: Your primary goal is to analyse vehicle symptoms and provide clear, concise diagnostic steps.
### Output: Give a structured output in a markdown format using headings and bullet points.
### Constraints: If the question is not related to vehicle diagnostics, politely refuse to answer.
"""

# Read test set and convert to Dataset object
df_test = pd.read_json(test_file, lines=True)
data_test = Dataset.from_pandas(df_test)

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

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side="left"
)

# Load trained adapters and set to eval mode
model = PeftModel.from_pretrained(model, model_path)
model.eval()

generation_kwargs = {
    "max_new_tokens": 512,
    "eos_token_id": tokenizer.eos_token_id,
    "return_full_text": False,
}

# Build the prompt
def apply_chat_template(example):
    """
    Applies the chat template to a batch of samples from the test set.
    """
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": example["question"]}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {"prompt": prompt}

# Apply chat template & get the list of prompts
data_prompts = data_test.map(apply_chat_template)
prompts_list = data_prompts["prompt"][:]

# Create pipeline and get predictions
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
outputs = pipe(prompts_list, **generation_kwargs, batch_size=8)
pred = [output[0]["generated_text"].strip() for output in outputs]

# Save evaluation results
df_results = pd.DataFrame({
    "question": data_test["question"],
    "answer_ref": data_test["answer"],
    "answer_pred": pred
})

print("Evaluation complete.")
df_results.to_csv(results_path, index=False)