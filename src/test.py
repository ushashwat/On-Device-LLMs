import torch
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

test_file = "../data/test.jsonl"
results_path = "../data/evaluation.csv"
model_path = "../model/tiny_qwen"
merged_model_path = model_path + "_merged"

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

# Read test set and convert to Dataset object
df_test = pd.read_json(test_file, lines=True)
data_test = Dataset.from_pandas(df_test)

# Load merged model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    merged_model_path,
    trust_remote_code=True,
    padding_side="left"
)

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
model.eval()
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
outputs = pipe(prompts_list, **generation_kwargs, batch_size=32)
pred = [output[0]["generated_text"].strip() for output in outputs]

# Save evaluation results
df_results = pd.DataFrame({
    "question": data_test["question"],
    "answer_ref": data_test["answer"],
    "answer_pred": pred
})

print("Evaluation complete.")
df_results.to_csv(results_path, index=False)