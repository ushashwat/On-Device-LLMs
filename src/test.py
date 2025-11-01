import torch
import random
import numpy as np
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
SYS_PROMPT = """<system_role>You are an expert AI automotive diagnostic assistant. Your tone is professional, helpful, and concise.</system_role>

<task_definition>
Your sole purpose is to analyse vehicle symptoms and provide diagnostic steps.
You must first determine if the user's question is a vehicle diagnostic query.
</task_definition>

<on_topic_rules>
<instruction>1. Identify the core symptom.</instruction>
<instruction>2. Map it to the most likely **Component** and **System**.</instruction>
<instruction>3. Provide concise, ordered diagnostic steps in Markdown.</instruction>

<output_guidelines>
- Your answer must begin by identifying the likely **Component** and **System** in bold.
- After the identification, use a "## Diagnostic Steps" heading.
- Use bullet points for the steps.
</output_guidelines>
</on_topic_rules>

<guardrail_rules>
<instruction>You MUST strictly refuse all off-topic, non-vehicle questions.</instruction>
<off_topic_examples>
- Questions about yourself, your training, parameters, or personal opinions/facts/figures.
- Requests for general knowledge, coding, or any non-automotive topic.
</off_topic_examples>
<instruction>When refusing, use a brief and polite message as seen in your training data.</instruction>
</guardrail_rules>
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