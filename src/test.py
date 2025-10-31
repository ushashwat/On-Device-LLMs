import torch
import pandas as pd
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

df_test = pd.read_json(test_file, lines=True)

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
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

# Load trained adapters and set to eval mode
model = PeftModel.from_pretrained(model, model_path)
model.eval()

# Build the prompt
def apply_chat_template(example, tokenizer):
    """
    Takes each example from the test set and applies the chat template.
    """
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": example["question"]}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text

# Apply chat template to test set
prompts = df_test.apply(
    lambda ex: apply_chat_template(ex, tokenizer),
    axis=1
).to_list()

generation_kwargs = {
    "max_new_tokens": 512,
    "eos_token_id": tokenizer.eos_token_id,
    "return_full_text": False,
}

# Create pipeline and run evaluation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
outputs = pipe(prompts, **generation_kwargs)



# Get predictions
pred = [output[0]['generated_text'].strip() for output in outputs]
results = pd.DataFrame({
    "question": df_test["question"],
    "answer_ref": df_test["answer"],
    "answer_pred": pred
})

print("Evaluation complete.")

# Save evaluation results
results.to_csv(results_path, index=False)