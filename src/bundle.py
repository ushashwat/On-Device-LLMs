"""bundle.py script for creating the .task bundle."""
import os
from mediapipe.tasks.python.genai import bundler
from src.logger_config import setup_logger

logger = setup_logger(__name__, "pipeline.log")
MODEL_PATH = "../model"

def create_task(tflite_model: str, tokenizer: str, output_dir: str) -> None:
    """Bundles .tflite model and tokenizer into a .task file."""
    task_file = os.path.join(output_dir, "tiny_gemma.task")

    config = bundler.BundleConfig(
        tflite_model=tflite_model,
        tokenizer_model=tokenizer,
        start_token="<bos>",
        stop_tokens=["<eos>", "<end_of_turn>"],
        output_filename=task_file,
    )
    bundler.create_bundle(config)

def main():
    """Main execution logic creating a .task file."""
    logger.info("Starting bundling pipeline..")

    merged_path = os.path.join(MODEL_PATH, "tiny_gemma_merged")
    litert_path = os.path.join(MODEL_PATH, "tiny_gemma_litert")

    tokenizer = os.path.join(merged_path, "tokenizer.model")
    tflite_model = os.path.join(litert_path, "gemma3_1b_finetune_q4_block32_ekv1024.tflite")

    create_task(
        tflite_model,
        tokenizer,
        litert_path,
    )
    logger.info("Bundled into a .task file format.")

if __name__ == "__main__":
    main()
