import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
MODEL_PATH = "models/codet5-base"  # Ensure the model files are in this directory

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Function to generate code from input prompt
def generate_code(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Main chatbot loop
if __name__ == "__main__":
    print("\n💡 AI Code Generator Chatbot - Powered by CodeT5")
    print("🔹 Type 'exit' to quit\n")

    while True:
        user_input = input("👩‍💻 Enter a code prompt: ")
        if user_input.lower() == "exit":
            print("👋 Exiting chatbot. Goodbye!")
            break

        print("\n🤖 AI-Generated Code:\n")
        try:
            generated_code = generate_code(user_input)
            print(generated_code)
        except Exception as e:
            print(

