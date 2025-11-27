"""test_val.py for validation script's unit tests."""
import random
from unittest.mock import patch, MagicMock, ANY
import pytest
import pandas as pd
from datasets import Dataset
from src import val

# Fixtures
@pytest.fixture(name="mock_tokenizer")
def fixture_tokenizer():
    """Mocks the AutoTokeniser."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="Formatted Prompt")
    tokenizer.eos_token_id = 12345
    return tokenizer

@pytest.fixture(name="mock_test_dataset")
def fixture_test_dataset():
    """Mocks a loaded Dataset object."""
    data = {"question": ["q1"], "answer": ["a1"], "type": ["main"]}
    return Dataset.from_dict(data)

# Unit Tests
def test_set_seed():
    """Verifies seed execution and reproducibility."""
    val.set_seed(18)
    val_1 = random.random()
    val.set_seed(18)
    val_2 = random.random()

    assert val_1 == val_2

@patch("src.val.pd.read_json")
def test_process_test_data(mock_read_json, mock_test_dataset):
    """Verifies data loading and Dataset conversion."""
    mock_read_json.return_value = pd.DataFrame()
    with patch("src.val.Dataset.from_pandas", return_value=mock_test_dataset) as mock_from_pandas:
        result = val.process_test_data("fake.jsonl")
        mock_read_json.assert_called_with("fake.jsonl", lines=True)
        mock_from_pandas.assert_called_once()
        assert isinstance(result, Dataset)

@pytest.mark.parametrize("model_name, expected_content_start", [
    ("gemma", "SYS\nq1"),
    ("qwen", "q1")
])
def test_apply_chat_template_logic(mock_tokenizer, model_name, expected_content_start):
    """Verifies model-specific chat template creation and prompt generation."""
    example = {"question": "q1"}
    result = val.apply_chat_template("SYS", example, mock_tokenizer, model_name)

    # Check if the primary user content is structured correctly based on model
    call_messages = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert call_messages[-1]['content'].startswith(expected_content_start)

    mock_tokenizer.apply_chat_template.assert_called_with(
        ANY, tokenize=False, add_generation_prompt=True,
    )
    assert result == {"prompt": "Formatted Prompt"}

@patch("src.val.AutoModelForCausalLM.from_pretrained")
@patch("src.val.AutoTokenizer.from_pretrained")
def test_load_model_and_tokenizer(mock_auto_tokenizer, mock_auto_model):
    """Verifies model and tokeniser loading."""
    mock_auto_model.return_value = "model"
    mock_auto_tokenizer.return_value = "tokenizer"

    model, tokenizer = val.load_model_and_tokenizer("fake/path", token="abc")

    mock_auto_model.assert_called_with(
        "fake/path", dtype=ANY, token="abc"
    )
    mock_auto_tokenizer.assert_called_with(
        "fake/path", padding_side="left", token="abc"
    )
    assert model == "model"
    assert tokenizer == "tokenizer"

@patch("src.val.pipeline")
def test_generate_preds_execution(mock_pipeline, mock_tokenizer):
    """Verifies pipeline initialisation, execution args, and prediction stripping."""
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [[{"generated_text": " pred "}],]
    mock_pipeline.return_value = mock_pipe_instance
    mock_model = MagicMock()

    preds = val.generate_preds(mock_model, mock_tokenizer, ["prompt1"])
    mock_model.eval.assert_called_once()
    mock_pipeline.assert_called_with("text-generation", model=mock_model, tokenizer=mock_tokenizer)

    mock_pipe_instance.assert_called_with(
        ["prompt1"], max_new_tokens=256, eos_token_id=12345, return_full_text=False, batch_size=32
    )
    assert preds == ["pred"]

def test_save_eval_results_io(mock_test_dataset, tmp_path):
    """Verifies results are formatted into a DataFrame and saved to CSV."""
    results_file = tmp_path / "results.csv"
    val.save_eval_results(mock_test_dataset, ["p1"], str(results_file))

    assert results_file.exists()
