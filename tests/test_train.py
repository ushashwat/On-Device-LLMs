"""test_train.py for training script's unit tests."""
from unittest.mock import MagicMock, patch, ANY
import random
import pytest
import pandas as pd
from datasets import DatasetDict
from transformers import TrainingArguments
from peft import LoraConfig
from src import train

# Fixtures
@pytest.fixture(name="mock_tokenizer")
def tokenizer_fixture():
    """Mocks the tokeniser to avoid loading heavy files."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    # Mock chat template application & tokeniser return value
    tokenizer.apply_chat_template.return_value = "Formatted Prompt"
    tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    return tokenizer

@pytest.fixture(name="sample_data_file")
def data_file_fixture(tmp_path):
    """Creates a dummy JSONL file for data testing."""
    data = [
        {"type": "x", "question": "q1", "answer": "a1"},
        {"type": "x", "question": "q2", "answer": "a2"},
        {"type": "y", "question": "q3", "answer": "a3"},
        {"type": "y", "question": "q4", "answer": "a4"},
        {"type": "z", "question": "q5", "answer": "a5"},
        {"type": "z", "question": "q6", "answer": "a6"},
    ]
    # Create more rows for splits
    df = pd.DataFrame(data * 5)
    file_path = tmp_path / "data.jsonl"
    df.to_json(file_path, orient="records", lines=True)
    return str(file_path)

# Unit Tests
def test_set_seed():
    """Verifies seed runs."""
    train.set_seed(18)
    train_1 = random.random()
    train.set_seed(18)
    train_2 = random.random()

    assert train_1 == train_2

def test_process_data(sample_data_file, tmp_path):
    """Verifies data loading, splitting, and return types."""
    save_dir = tmp_path / "splits"
    result = train.process_data(sample_data_file, str(save_dir), random_state=18)

    assert isinstance(result, DatasetDict)
    assert "train" in result and "val" in result and "test" in result

    # Check files were saved
    assert (save_dir / "train.jsonl").exists()
    assert (save_dir / "val.jsonl").exists()
    assert (save_dir / "test.jsonl").exists()

@pytest.mark.parametrize("model_name, expected_role", [
    ("gemma", "model"),
    ("qwen", "assistant")
])
def test_apply_chat_template(mock_tokenizer, model_name, expected_role):
    """Verifies logic for different model families."""
    example = {"question": "Hello", "answer": "Hi"}

    train.apply_chat_template("SysPrompt", example, mock_tokenizer, model_name)

    # Extract the messages passed to the tokenizer
    call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert call_args[-1]['role'] == expected_role
    assert call_args[-1]['content'] == "Hi"

def test_configurations():
    """Verifies configuration objects are created correctly."""
    peft_conf = train.create_peft_config()
    assert isinstance(peft_conf, LoraConfig)
    assert peft_conf.task_type == "CAUSAL_LM"

    train_args = train.create_train_args("output_dir")
    assert isinstance(train_args, TrainingArguments)
    assert train_args.bf16 is True

@patch("src.train.get_peft_model")
def test_prepare_model_for_peft(mock_get_peft):
    """Verifies PEFT model wrapper is applied."""
    mock_model = MagicMock()
    train.prepare_model_for_peft(mock_model, "config")

    mock_get_peft.assert_called_with(mock_model, "config")
    mock_get_peft.return_value.print_trainable_parameters.assert_called_once()

@patch("src.train.AutoModelForCausalLM.from_pretrained")
@patch("src.train.AutoTokenizer.from_pretrained")
def test_load_model(mock_tok, mock_model):
    """Verifies model and tokeniser loaders are called with correct args."""
    mock_instance = mock_tok.return_value
    mock_instance.pad_token = None
    model, tokenizer = train.load_model_and_tokenizer("model-id")

    mock_model.assert_called_with("model-id", dtype=ANY)
    mock_tok.assert_called_with("model-id")

    # Verify critical config settings
    assert model.config.use_cache is False
    assert model.config.pretraining_tp == 1
    assert tokenizer.pad_token == tokenizer.eos_token

@patch("src.train.SFTTrainer")
def test_train_sft_execution(mock_sft_trainer):
    """Verifies trainer initialisation and execution."""
    train.train_sft(
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )

    mock_sft_trainer.assert_called_once()
    mock_sft_trainer.return_value.train.assert_called_once()

def test_merge_model_execution(mock_tokenizer):
    """Verifies merge logic and file saving."""
    mock_trainer = MagicMock()

    train.merge_model(mock_trainer, mock_tokenizer, "output_path")

    merged_model = mock_trainer.model.merge_and_unload.return_value
    merged_model.resize_token_embeddings.assert_called_with(262144)
    merged_model.save_pretrained.assert_called_with(
        "output_path",
        safe_serialization=True,
        max_shard_size="2GB"
    )
    mock_tokenizer.save_pretrained.assert_called_with("output_path")
