from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load tokenizer and model
model_path = "models/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)  # ✅ Auto-detects correct tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)  # ✅ Auto-detects correct model


# Function to generate code
def generate_code(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test the model
test_prompt = "def add(a, b):"
generated_code = generate_code(test_prompt)

print("Generated Code:\n", generated_code)
