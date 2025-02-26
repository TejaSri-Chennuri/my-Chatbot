from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load tokenizer and model from the local folder
model_path = "Salesforce/codet5-base"  # ✅ Correct

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
print("Tokenizer loaded successfully!")

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(model_path)
print("Model loaded successfully!")

def generate_code(prompt):
    print(f"Generating code for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    print(f"Tokenized input: {inputs}")
    
    output = model.generate(**inputs, max_length=50)
    print(f"Generated token output: {output}")
    
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Decoded output: {decoded_output}")
    
    return decoded_output

# Test the model
if __name__ == "__main__":
    test_prompt = "def add(a, b):"
    generated_code = generate_code(test_prompt)
    print("Generated Code:\n", generated_code)