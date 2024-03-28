# import python-dotenv
import os
from dotenv import load_dotenv

# import from huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# import regex for clean response
import re

# import gradio for gui
import gradio as gr


# take environment variables from .env file
load_dotenv()
token = os.getenv('TOKEN')

# set some stuffs
model_id = "google/gemma-2b-it"
dtype = torch.bfloat16

# start with chat
def gemma_chat(message, history):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        hidden_activation="gelu_pytorch_tanh",
        device_map="cuda",
        torch_dtype=dtype,
    )

    chat = [
        { "role": "user", "content": message },
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=2048)
    
    response = tokenizer.decode(outputs[0])
    
    
    # clean the response
    response_cleaned = re.split("model", response)

    # return the response
    return response_cleaned[1]


gr.ChatInterface(gemma_chat).launch()
