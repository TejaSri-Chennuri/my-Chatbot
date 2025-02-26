from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()  # ✅ Make sure this is here

# ✅ Use a pre-trained model instead of fine-tuned
model_name = "meta-llama/Meta-Llama-3-8B"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# ✅ Define input format
class ChatRequest(BaseModel):
    input_text: str

# ✅ Create chatbot endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    inputs = tokenizer(request.input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}

# ✅ Run API with: uvicorn app:app --host 0.0.0.0 --port 8000

