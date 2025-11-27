"""test_run_pipeline.py for integration testing."""
import os
import json
from unittest.mock import patch
import pytest
from transformers import TrainingArguments
from src.pipeline import EdgeLLMPipeline

@pytest.fixture(name="setup_env")
def fixture_setup_env(tmp_path):
    """Sets up a temporary environment with dummy data and paths."""
    paths = {
        "data": tmp_path / "data",
        "model": tmp_path / "model",
        "prompts": tmp_path / "prompts",
    }
    for p in paths.values():
        p.mkdir()

    # Create dummy data & prompt
    data_file = paths["data"] / "data.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for _ in range(10):
            f.write(json.dumps({"type": "diag", "question": "q", "answer": "a"}) + "\n")

    with open(paths["prompts"] / "prompt_short.txt", "w", encoding="utf-8") as f:
        f.write("System Prompt")

    return paths, str(data_file)

@pytest.mark.integration
def test_pipeline_lifecycle(setup_env):
    """Runs the full pipeline."""
    paths, data_file = setup_env
    temp_results_file = str(paths["data"] / "evaluation.csv")

    fast_args = TrainingArguments(
        output_dir=str(paths["model"]),
        max_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to="none",
        save_strategy="no",
        use_cpu=True  # Force CPU to avoid CUDA errors
    )

    token = os.getenv("HF_TOKEN", "dummy_token_for_test")

    # Patch paths AND arguments
    with patch.dict(os.environ, {"HF_TOKEN": token}), \
         patch("src.pipeline.EdgeLLMPipeline.DATA_PATH", str(paths["data"])), \
         patch("src.pipeline.EdgeLLMPipeline.MODEL_PATH", str(paths["model"])), \
         patch("src.pipeline.EdgeLLMPipeline.PROMPT_PATH", str(paths["prompts"])), \
         patch("src.pipeline.EdgeLLMPipeline.DATA_FILE", data_file), \
         patch("src.pipeline.EdgeLLMPipeline.TEST_FILE", data_file), \
         patch("src.pipeline.EdgeLLMPipeline.RESULTS_FILE", temp_results_file), \
         patch("src.train.create_train_args", return_value=fast_args), \
         patch("src.pred.generate_response", return_value="Mocked Reply") as mock_gen:

        pipe = EdgeLLMPipeline(model_name="gemma")

        # TRAIN
        pipe.run_train()
        assert (paths["model"] / "tiny_gemma_merged").exists()

        # EVAL
        pipe.run_eval()
        assert (paths["data"] / "evaluation.csv").exists()

        # PRED
        pipe.run_pred()
        mock_gen.assert_called_once()
