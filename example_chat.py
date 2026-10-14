import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from hydrogen import HydrogenConfig, HydrogenForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread

def load_model(path):
    config = HydrogenConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HydrogenForCausalLM.from_pretrained(path, exit_flag_channel=False)

    model.to(device)
    model.eval()
    return model, tokenizer, device

messages = []

def generate(model, tokenizer, device, prompt):
    messages.append({"role": "user", "content": prompt})

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    output = ""

    for chunk in streamer:
        print(chunk, end="", flush=True)
        output += chunk

    thread.join()
    print()

    messages.append({"role": "assistant", "content": output})

def main():
    # path = "takenusername32/Hydrogen-v0.5-25M-Base"
    path = input("Path: ")
    model, tokenizer, device = load_model(path)

    while True:
        prompt = input("User: ")
        if prompt.strip().lower() in ["exit", "quit"]:
            break

        generate(model, tokenizer, device, prompt)

if __name__ == "__main__":
    main()
