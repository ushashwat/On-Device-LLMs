"""pred.py script for inference."""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

def load_model_and_tokenizer(
        model_path: str,
        **model_kwargs,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the merged model and tokeniser for inference."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        **model_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        **model_kwargs
    )
    model.eval()
    return model, tokenizer

def generate_response(
        sys_prompt: str,
        user_prompt: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_name: str,
    ) -> str:
    """Generates a response based on user prompt."""
    if model_name == "gemma":
        messages = [
            {"role": "user", "content": f"{sys_prompt}\n{user_prompt}"}
        ]
    else:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response
