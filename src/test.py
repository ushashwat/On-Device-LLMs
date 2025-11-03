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

SEED = 18
TEST_FILE = "../data/test.jsonl"
RESULTS_PATH = "../data/evaluation.csv"
MODEL_PATH = "../model/tiny_qwen_merged"

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

def process_test_data(test_file):
    """
    Loads test data from JSONL file.
    """
    df_test = pd.read_json(test_file, lines=True)
    data_test = Dataset.from_pandas(df_test)
    return data_test

def apply_chat_template(example, tokenizer):
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

def load_model_and_tokenizer(merged_model_path):
    """
    Loads merged model and tokeniser.
    """
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
    return model, tokenizer

def generate_preds(model, tokenizer, prompts_list):
    """
    Generates predictions using model pipeline.
    """
    generation_kwargs = {
        "max_new_tokens": 512,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False,
    }
    
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    outputs = pipe(prompts_list, **generation_kwargs, batch_size=32)
    preds = [output[0]["generated_text"].strip() for output in outputs]
    return preds

def save_eval_results(data_test, predictions, results_path):
    """
    Saves evaluation results to a CSV file.
    """
    df_results = pd.DataFrame({
        "question": data_test["question"],
        "answer_ref": data_test["answer"],
        "answer_pred": predictions
    })
    df_results.to_csv(results_path, index=False)

def main():
    print("Starting evaluation pipeline..")
    set_seed(SEED)

    print("Loading test data..")
    data_test = process_test_data(TEST_FILE)

    print("Loading model and tokeniser..")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # Apply chat template & get the list of prompts
    data_prompts = data_test.map(
        lambda ex: apply_chat_template(ex, tokenizer)
    )
    prompts_list = data_prompts["prompt"][:]

    print("Generating predictions..")
    preds = generate_preds(model, tokenizer, prompts_list)

    print("Saving results..")
    save_eval_results(data_test, preds, RESULTS_PATH)

    print("Evaluation pipeline finished.")

if __name__ == "__main__":
    main()