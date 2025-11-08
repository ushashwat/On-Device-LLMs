import os
import sys
import argparse
import train, eval, pred
from transformers import DataCollatorForLanguageModeling

SEED = 18
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_FILE = "/home/shashwat/Topic_Modelling/check/data/data.jsonl"
TEST_FILE = "/home/shashwat/Topic_Modelling/check/data/test.jsonl"
RESULTS_FILE = "/home/shashwat/Topic_Modelling/check/data/evaluation.csv"
DATA_PATH = "/home/shashwat/Topic_Modelling/check/data"
MODEL_PATH = "/home/shashwat/Topic_Modelling/check/model"

USER_PROMPT = "My window is stuck, what do i do?"

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
    <task>Provide maximum TWO diagnostic steps.</task>
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

def run_train(paths, model_id, model_kwargs):
    print("Starting training pipeline..")
    train.set_seed(SEED)

    checkpoint_path = paths["checkpoint_path"]
    adapter_path = paths["adapter_path"]
    merged_path = paths["merged_path"]

    all_datasets = train.process_data(DATA_FILE, DATA_PATH, random_state=SEED)

    # Load model & tokeniser
    bnb_config = train.create_bnb_config()
    peft_config = train.create_peft_config()
    model, tokenizer = train.load_model_and_tokenizer(model_id, bnb_config, **model_kwargs)
    model = train.prepare_model_for_peft(model, peft_config)

    print("Tokenising datasets..")
    tokenised_dataset = all_datasets.map(
        lambda ex: train.apply_chat_template(SYS_PROMPT, ex, tokenizer),
        remove_columns=["type", "question", "answer"],
    )

    # Convert tokeniser into data collator for SFTTrainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("Fine-tuning model..")
    training_args = train.create_train_args(checkpoint_path)
    trainer = train.train_sft(
        model,
        tokenizer,
        training_args,
        tokenised_dataset["train"],
        tokenised_dataset["val"],
        data_collator,
    )
    print("Fine-tuning complete.")

    # Save the fine-tuned adapter & merged model
    print(f"Saving adapter to: {adapter_path}")
    trainer.save_model(adapter_path)

    train.merge_model(trainer, tokenizer, merged_path)
    print("Training pipeline finished.")

def run_eval(paths, model_kwargs):
    print("Starting evaluation pipeline..")
    eval.set_seed(SEED)

    merged_path = paths["merged_path"]

    data_test = eval.process_test_data(TEST_FILE)

    print("Loading model and tokeniser..")
    model, tokenizer = eval.load_model_and_tokenizer(merged_path, **model_kwargs)

    # Apply chat template & get the list of prompts
    data_prompts = data_test.map(
        lambda ex: eval.apply_chat_template(SYS_PROMPT, ex, tokenizer)
    )
    prompts_list = data_prompts["prompt"][:]

    preds = eval.generate_preds(model, tokenizer, prompts_list)

    print("Saving results..")
    eval.save_eval_results(data_test, preds, RESULTS_FILE)
    print("Evaluation pipeline finished.")

def run_pred(paths, model_kwargs):
    print("Starting inference pipeline..")
    merged_path = paths["merged_path"]

    model, tokenizer = pred.load_model_and_tokenizer(merged_path, **model_kwargs)

    # Run inference
    print(f"Prompt: {USER_PROMPT}\n")
    response = pred.generate_response(SYS_PROMPT, USER_PROMPT, model, tokenizer)
    print(f"Reply:\n {response}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune QLoRA LLM.")
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        choices=["train", "eval", "pred"],
        help="The script to run"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["gemma", "qwen"],
        help="The LLM to fine-tune",
    )
    args = parser.parse_args()

    paths = {
        "checkpoint_path": os.path.join(MODEL_PATH, f"tiny_{args.model_name}_output"),
        "adapter_path": os.path.join(MODEL_PATH, f"tiny_{args.model_name}_adapter"),
        "merged_path": os.path.join(MODEL_PATH, f"tiny_{args.model_name}_merged"),
    }
    
    model_id = ""
    model_kwargs = {}
    if args.model_name == "gemma":
        model_id = "google/gemma-3-1b-it"
        if not HF_TOKEN:
            print("Error: Gemma3 is a gated model and requires authentication.")
            sys.exit(1)
        model_kwargs = {"token": HF_TOKEN}
    elif args.model_name == "qwen":
        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        model_kwargs = {"trust_remote_code": True}
    
    if args.script == "train":
        run_train(paths, model_id, model_kwargs)
    elif args.script == "eval":
        run_eval(paths, model_kwargs)
    elif args.script == "pred":
        run_pred(paths, model_kwargs)

if __name__ == "__main__":
    main()