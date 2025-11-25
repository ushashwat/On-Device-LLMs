"""test_pipeline.py for pipeline script's unit tests."""
import sys
import os
from unittest.mock import patch, mock_open, MagicMock
import pytest
sys.modules["src.logger_config"] = MagicMock() # Resolve logging setup config
from src.pipeline import EdgeLLMPipeline, load_prompt, main

# Fixtures
@pytest.fixture(name="mock_deps")
def fixture_dependencies():
    """Mocks external modules for assertion checking."""
    with patch("src.pipeline.train") as m_train, \
         patch("src.pipeline.val") as m_val, \
         patch("src.pipeline.pred") as m_pred, \
         patch("src.pipeline.DataCollatorForLanguageModeling"):

        # Setup return structures to allow chained calls
        m_train.process_data.return_value.map.return_value = {"train": [], "val": []}
        m_val.process_test_data.return_value.map.return_value = {"prompt": ["test"]}

        mock_objects = (MagicMock(), MagicMock())
        m_train.load_model_and_tokenizer.return_value = mock_objects
        m_val.load_model_and_tokenizer.return_value = mock_objects
        m_pred.load_model_and_tokenizer.return_value = mock_objects
        yield {"train": m_train, "val": m_val, "pred": m_pred}

@pytest.fixture(name="gemma_pipe")
def fixture_auth():
    """Returns a ready-to-use pipeline instance with env vars set."""
    with patch.dict(os.environ, {"HF_TOKEN": "test_token"}), \
         patch("builtins.open", mock_open(read_data="mock_sys_prompt")):
        return EdgeLLMPipeline("gemma")

# Unit Tests
def test_prompt_logic():
    """Verifies file reading and error handling."""
    with patch("builtins.open", mock_open(read_data="mock_prompt_content")):
        assert load_prompt("valid.txt") == "mock_prompt_content"

    with patch("builtins.open", side_effect=FileNotFoundError), pytest.raises(SystemExit):
        load_prompt("missing.txt")

@pytest.mark.parametrize("model, expected_id", [
    ("gemma", "google/gemma-3-1b-it"),
    ("qwen", "Qwen/Qwen3-4B-Instruct-2507")
])
def test_init_configs(model, expected_id):
    """Verifies correct attribute assignment for different models."""
    with patch.dict(os.environ, {"HF_TOKEN": "test"}), \
         patch("builtins.open", mock_open(read_data="prompt")):
        pipe = EdgeLLMPipeline(model)
        assert pipe.model_id == expected_id
        assert pipe.model_name == model

def test_init_missing_token():
    """Verifies security check for gated models."""
    with patch.dict(os.environ, {}, clear=True), \
         patch("builtins.open", mock_open(read_data="prompt")), \
         pytest.raises(EnvironmentError):
        EdgeLLMPipeline("gemma")

def test_run_train_execution(gemma_pipe, mock_deps):
    """Verifies the training sequence: setup -> train -> save -> merge."""
    gemma_pipe.run_train()
    mock_deps["train"].set_seed.assert_called_once()
    mock_deps["train"].load_model_and_tokenizer.assert_called_once()
    mock_deps["train"].train_sft.assert_called_once()
    mock_deps["train"].merge_model.assert_called_once()

def test_run_eval_execution(gemma_pipe, mock_deps):
    """Verifies evaluation sequence: load -> map -> predict -> save."""
    gemma_pipe.run_eval()
    mock_deps["val"].generate_preds.assert_called_once()
    mock_deps["val"].save_eval_results.assert_called_once()

def test_run_pred_execution(gemma_pipe, mock_deps):
    """Verifies inference sequence."""
    gemma_pipe.run_pred()
    mock_deps["pred"].generate_response.assert_called_once()

@pytest.mark.parametrize("cmd_arg, method", [
    ("train", "run_train"),
    ("eval", "run_eval"),
    ("pred", "run_pred"),
])
def test_main(cmd_arg, method):
    """Verifies CLI args map to the correct pipeline method."""
    # Patch EdgeLLMPipeline
    with patch.object(sys, "argv", ["script", "--script", cmd_arg, "--model_name", "gemma"]), \
         patch("src.pipeline.EdgeLLMPipeline") as mockpipe, \
         patch.dict(os.environ, {"HF_TOKEN": "t"}):

        main()
        getattr(mockpipe.return_value, method).assert_called_once()
