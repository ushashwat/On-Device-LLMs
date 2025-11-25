"""test_pred.py for prediction script's unit tests."""
from unittest.mock import patch, MagicMock, ANY
import pytest
import torch
from src import pred

# Fixtures
@pytest.fixture(name="mock_tokenizer")
def fixture_tokenizer():
    """Mocks tokeniser to return a real tensor dictionary."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 12345

    mock_encoding = MagicMock()
    mock_encoding.to.return_value = {
        "input_ids": torch.zeros((1, 5), dtype=torch.long),
        "attention_mask": torch.zeros((1, 5), dtype=torch.long)
    }

    tokenizer.apply_chat_template.return_value = mock_encoding
    return tokenizer

@pytest.fixture(name="mock_model")
def fixture_model():
    """Mocks model to return a fixed output tensor."""
    model = MagicMock()
    model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
    return model

# Unit Tests
@patch("src.pred.AutoModelForCausalLM.from_pretrained")
@patch("src.pred.AutoTokenizer.from_pretrained")
def test_load_model_and_tokenizer(mock_auto_tokenizer, mock_auto_model):
    """Verifies model and tokeniser loading."""
    pred.load_model_and_tokenizer("fake/path")

    mock_auto_tokenizer.assert_called_with(
        "fake/path", padding_side="left"
    )
    mock_auto_model.assert_called_with(
        "fake/path", device_map="auto", dtype=ANY
    )
    mock_auto_model.return_value.eval.assert_called_once()

@pytest.mark.parametrize("model_name", ["gemma", "qwen"])
def test_generate_response_logic(mock_model, mock_tokenizer, model_name):
    """Verifies template logic and output slicing efficiency."""
    pred.generate_response(
        "Sys",
        "User",
        mock_model,
        mock_tokenizer,
        model_name
    )
    # Check template was applied correctly based on model type
    messages = mock_tokenizer.apply_chat_template.call_args[0][0]
    expected_role = "user" if model_name == "gemma" else "system"
    assert messages[0]["role"] == expected_role

    decoded_tensor = mock_tokenizer.decode.call_args[0][0]
    assert decoded_tensor.shape[0] == 5
