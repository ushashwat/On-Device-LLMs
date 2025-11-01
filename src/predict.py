import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

model_path = "../model/tiny_qwen_merged"
device = "cpu"

# Define system instruction
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
    <task>Provide diagnostic steps.</task>
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

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    quantization_config=bnb_config,
)
model.eval()

def generate_response(prompt):
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )

    return response

# Run inference
question = "My window is stuck, what do i do?"
print(generate_response(question))