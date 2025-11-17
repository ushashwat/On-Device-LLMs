"""pipeline.py script for orchestration."""
import os
import sys
import argparse
from transformers import DataCollatorForLanguageModeling
import src.train as train
import src.evaluate as evaluate
import src.pred as pred
import src.convert as convert
from src.logger_config import setup_logger

logger = setup_logger(__name__, "pipeline.log")

def load_prompt(file_path: str) -> str:
    """Helper function to read a prompt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Prompt file not found: %s", file_path)
        sys.exit(1)

class EdgeLLMPipeline:
    """Pipeline for training, evaluating, and deploying Edge LLMs."""
    ALL_MODELS = ["gemma", "qwen"]
    GEMMA_MODEL_ID = "google/gemma-3-1b-it"
    QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    SEED = 18

    DATA_FILE = "../data/data.jsonl"
    TEST_FILE = "../data/test.jsonl"
    RESULTS_FILE = "../data/evaluation.csv"
    DATA_PATH = "../data"
    MODEL_PATH = "../model"
    PROMPT_PATH = "../prompts"
    USER_PROMPT = "My window is stuck, what do i do now?"

    def __init__(self, model_name: str) -> None:
        """Initialises the Edge LLM pipeline."""
        if model_name not in self.ALL_MODELS:
            raise ValueError("Unsupported LLM!")

        self.hf_token = os.getenv("HF_TOKEN")
        self.model_name = model_name
        self.model_id = ""
        self.model_kwargs = {}
        self.sys_prompt = ""

        self.paths = {
            "checkpoint_path": os.path.join(self.MODEL_PATH, f"tiny_{model_name}_output"),
            "adapter_path": os.path.join(self.MODEL_PATH, f"tiny_{model_name}_adapter"),
            "merged_path": os.path.join(self.MODEL_PATH, f"tiny_{model_name}_merged"),
            "litert_path": os.path.join(self.MODEL_PATH, f"tiny_{model_name}_litert"),
        }

        if model_name == "gemma":
            self.model_id = self.GEMMA_MODEL_ID
            self.sys_prompt = load_prompt(os.path.join(self.PROMPT_PATH, "prompt_short.txt"))
            if not self.hf_token:
                raise EnvironmentError("Gemma3 is a gated model and requires Hugging Face token.")
            self.model_kwargs = {"token": self.hf_token}
        elif model_name == "qwen":
            self.model_id = "Qwen/Qwen3-4B-Instruct-2507"
            self.sys_prompt = load_prompt(os.path.join(self.PROMPT_PATH, "prompt_long.txt"))
            self.model_kwargs = {"trust_remote_code": True}

    def run_train(self) -> None:
        """Executes the training pipeline."""
        logger.info("Starting training pipeline..")
        train.set_seed(self.SEED)

        checkpoint_path = self.paths["checkpoint_path"]
        adapter_path = self.paths["adapter_path"]
        merged_path = self.paths["merged_path"]

        all_datasets = train.process_data(self.DATA_FILE, self.DATA_PATH, random_state=self.SEED)

        # Load model & tokeniser
        peft_config = train.create_peft_config()
        model, tokenizer = train.load_model_and_tokenizer(
            self.model_id,
            **self.model_kwargs
        )
        model = train.prepare_model_for_peft(model, peft_config)

        tokenised_dataset = all_datasets.map(
            lambda ex: train.apply_chat_template(self.sys_prompt, ex, tokenizer, self.model_name),
            remove_columns=["type", "question", "answer"],
        )

        # Convert tokeniser into data collator for SFTTrainer
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        logger.info("Fine-tuning model..")
        training_args = train.create_train_args(checkpoint_path)
        trainer = train.train_sft(
            model,
            tokenizer,
            training_args,
            tokenised_dataset["train"],
            tokenised_dataset["val"],
            data_collator,
        )
        logger.info("Fine-tuning complete.")

        # Save the fine-tuned adapter & merged model
        logger.info("Saving adapter to: %s", adapter_path)
        trainer.save_model(adapter_path)

        train.merge_model(trainer, tokenizer, merged_path)
        logger.info("Training pipeline finished.")

    def run_eval(self) -> None:
        """Executes the evaluation pipeline."""
        logger.info("Starting evaluation pipeline..")
        evaluate.set_seed(self.SEED)

        merged_path = self.paths["merged_path"]

        data_test = evaluate.process_test_data(self.TEST_FILE)

        logger.info("Loading model and tokeniser..")
        model, tokenizer = evaluate.load_model_and_tokenizer(merged_path, **self.model_kwargs)

        # Apply chat template & get the list of prompts
        data_prompts = data_test.map(
            lambda ex: evaluate.apply_chat_template(self.sys_prompt, ex, tokenizer, self.model_name)
        )
        prompts_list = data_prompts["prompt"][:]

        preds = evaluate.generate_preds(model, tokenizer, prompts_list)

        evaluate.save_eval_results(data_test, preds, self.RESULTS_FILE)
        logger.info("Evaluation pipeline finished.")

    def run_pred(self) -> None:
        """Executes the inference pipeline."""
        logger.info("Starting inference pipeline..")
        merged_path = self.paths["merged_path"]

        model, tokenizer = pred.load_model_and_tokenizer(merged_path, **self.model_kwargs)

        # Run inference
        logger.info("Prompt: %s\n", self.USER_PROMPT)
        response = pred.generate_response(
            self.sys_prompt,
            self.USER_PROMPT,
            model,
            tokenizer,
            self.model_name
        )
        logger.info("Reply:\n %s", response)
        logger.info("Inference pipeline finished.")

    def run_convert(self) -> None:
        """Converts the merged model to TFLite format."""
        logger.info("Starting conversion pipeline..")

        merged_path = self.paths["merged_path"]
        litert_path = self.paths["litert_path"]

        convert.create_tflite(
            merged_path,
            litert_path,
        )
        logger.info("Conversion pipeline finished.")

def main() -> None:
    """The main function for orchestration."""
    parser = argparse.ArgumentParser(description="Fine-tune QLoRA LLM.")
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        choices=["train", "eval", "pred", "convert"],
        help="The script to run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["gemma", "qwen"],
        help="The LLM to fine-tune",
    )
    args = parser.parse_args()

    pipeline = EdgeLLMPipeline(model_name=args.model_name)

    if args.script == "train":
        pipeline.run_train()
    elif args.script == "eval":
        pipeline.run_eval()
    elif args.script == "pred":
        pipeline.run_pred()
    elif args.script == "convert":
        pipeline.run_convert()

if __name__ == "__main__":
    main()
