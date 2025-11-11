import os
import sys
import logging
import argparse
import train, eval, pred
from transformers import DataCollatorForLanguageModeling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EdgeLLMPipeline:
    """
    Pipeline for training, evaluating, and deploying Edge LLMs.
    """
    ALL_MODELS = ["gemma", "qwen"]
    GEMMA_MODEL_ID = "google/gemma-3-1b-it"
    QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    SEED = 18

    DATA_FILE = "../data/data.jsonl"
    TEST_FILE = "../data/test.jsonl"
    RESULTS_FILE = "../data/evaluation.csv"
    DATA_PATH = "../data"
    MODEL_PATH = "../model"

    USER_PROMPT = "My window is stuck, what do i do now?"
    SYS_PROMPT_SHORT = """
    You are an automotive diagnostic assistant who replies concisely.
    For vehicle symptoms, provide max 2 diagnostic steps identifying the Component and System in bold.
    Then list steps under '## Diagnostic Steps'.
    For non-vehicle questions (chit-chat, trivia, questions about yourself), politely refuse.
    """
    SYS_PROMPT_LONG = """
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

    def __init__(self, model_name: str) -> None:
        """
        Initialises the Edge LLM pipeline.
        """
        if model_name not in self.ALL_MODELS:
            raise ValueError("Unsupported LLM!")
        
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.model_name = model_name
        self.model_id = ""
        self.model_kwargs = {}
        self.SYS_PROMPT = ""

        self.paths = {
            "checkpoint_path": os.path.join(self.MODEL_PATH, f"tiny_{model_name}_output"),
            "adapter_path": os.path.join(self.MODEL_PATH, f"tiny_{model_name}_adapter"),
            "merged_path": os.path.join(self.MODEL_PATH, f"tiny_{model_name}_merged"),
        }

        if model_name == "gemma":
            self.model_id = self.GEMMA_MODEL_ID
            self.SYS_PROMPT = self.SYS_PROMPT_SHORT
            if not self.HF_TOKEN:
                raise EnvironmentError("Gemma3 is a gated model and requires Hugging Face token.")
            self.model_kwargs = {"token": self.HF_TOKEN}
        elif model_name == "qwen":
            self.model_id = "Qwen/Qwen3-4B-Instruct-2507"
            self.SYS_PROMPT = self.SYS_PROMPT_LONG
            self.model_kwargs = {"trust_remote_code": True}

    def run_train(self) -> None:
        """
        Executes the training pipeline.
        """
        logger.info("Starting training pipeline..")
        train.set_seed(self.SEED)

        checkpoint_path = self.paths["checkpoint_path"]
        adapter_path = self.paths["adapter_path"]
        merged_path = self.paths["merged_path"]

        all_datasets = train.process_data(self.DATA_FILE, self.DATA_PATH, random_state=self.SEED)

        # Load model & tokeniser
        bnb_config = train.create_bnb_config()
        peft_config = train.create_peft_config()
        model, tokenizer = train.load_model_and_tokenizer(self.model_id, bnb_config, **self.model_kwargs)
        model = train.prepare_model_for_peft(model, peft_config)

        tokenised_dataset = all_datasets.map(
            lambda ex: train.apply_chat_template(self.SYS_PROMPT, ex, tokenizer, self.model_name),
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
        logger.info(f"Saving adapter to: {adapter_path}")
        trainer.save_model(adapter_path)

        train.merge_model(trainer, tokenizer, merged_path)
        logger.info("Training pipeline finished.")

    def run_eval(self) -> None:
        """
        Executes the evaluation pipeline.
        """
        logger.info("Starting evaluation pipeline..")
        eval.set_seed(self.SEED)

        merged_path = self.paths["merged_path"]

        data_test = eval.process_test_data(self.TEST_FILE)

        logger.info("Loading model and tokeniser..")
        model, tokenizer = eval.load_model_and_tokenizer(merged_path, **self.model_kwargs)

        # Apply chat template & get the list of prompts
        data_prompts = data_test.map(
            lambda ex: eval.apply_chat_template(self.SYS_PROMPT, ex, tokenizer, self.model_name)
        )
        prompts_list = data_prompts["prompt"][:]

        preds = eval.generate_preds(model, tokenizer, prompts_list)

        eval.save_eval_results(data_test, preds, self.RESULTS_FILE)
        logger.info("Evaluation pipeline finished.")

    def run_pred(self):
        """
        Executes the inference pipeline.
        """
        logger.info("Starting inference pipeline..")
        merged_path = self.paths["merged_path"]

        model, tokenizer = pred.load_model_and_tokenizer(merged_path, **self.model_kwargs)

        # Run inference
        logger.info(f"Prompt: {self.USER_PROMPT}\n")
        response = pred.generate_response(self.SYS_PROMPT, self.USER_PROMPT, model, tokenizer, self.model_name)
        logger.info(f"Reply:\n {response}")

def main() -> None:
    """
    Main entry point for the pipeline.
    """
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

    try:
        pipeline = EdgeLLMPipeline(model_name=args.model_name)
        
        if args.script == "train":
            pipeline.run_train()
        elif args.script == "eval":
            pipeline.run_eval()
        elif args.script == "pred":
            pipeline.run_pred()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()