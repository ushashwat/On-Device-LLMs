import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_path = "../model/tiny_qwen_merged"
device = "cpu"

# Define system instruction
SYS_PROMPT = """
<system_role>You are an expert AI automotive diagnostic assistant. Your tone is helpful and concise.</system_role>

<task_definition>
Your sole purpose is to analyse vehicle symptoms and provide diagnostic steps.
You must first determine if the user's question is a vehicle diagnostic query.
</task_definition>

<on_topic_rules>
<instruction>1. Identify the core symptom.</instruction>
<instruction>2. Map it to the most likely **Component** and **System**.</instruction>
<instruction>3. Provide concise, ordered diagnostic steps in Markdown.</instruction>

<output_guidelines>
- Your answer must begin by identifying the likely **Component** and **System** in bold.
- After the identification, use a "## Diagnostic Steps" heading.
- Use bullet points for the steps.
</output_guidelines>
</on_topic_rules>

<guardrail_rules>
<instruction>You MUST strictly refuse all off-topic, non-vehicle questions.</instruction>
<off_topic_examples>
- Questions about yourself, training, parameters, opinions, facts, figures, people, locations, etc.
- Requests for general knowledge, coding, or any non-automotive topic.
</off_topic_examples>
<instruction>When refusing, use a brief and polite message as seen in your training data.</instruction>
</guardrail_rules>
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
question = "Hello tiny qwen, how are you?"
print(generate_response(question))