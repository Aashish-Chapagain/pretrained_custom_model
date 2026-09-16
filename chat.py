import argparse
import sys
import sentencepiece as spm
import torch
from config.settings import GENERATE, TOKENIZER
from generate import load_model, generate
from train import resolve_device

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initializing chat interface...")
    device = resolve_device()
    print(f"Using device: {device}")
    
    tokenizer = spm.SentencePieceProcessor(
        model_file=f"{TOKENIZER['model_prefix']}.model"
    )
    model = load_model(device)
    
    print("\nModel loaded! You can now chat with your LLM.")
    print("Type 'quit' or 'exit' to stop the chat.\n")
    
    context = ""
    
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break
            
        if user_input.strip().lower() in ["quit", "exit"]:
            print("Exiting chat...")
            break
            
        if not user_input.strip():
            continue
            
        # Optional: formatting it as a simple chat turn if it was trained as one,
        # but since it's likely a standard completion model, we just append to context.
        # Let's just pass the user input as prompt or append it to context.
        prompt = context + user_input + "\n"
        
        # Generate text
        print("Bot: ", end="", flush=True)
        # We don't stream in the current generate function, we just get the full response.
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=GENERATE["max_new_tokens"],
            temperature=GENERATE["temperature"],
            device=device,
        )
        
        # The generated response will include the prompt. 
        # We want to extract only the newly generated part to display.
        new_text = response[len(prompt):]
        print(new_text)
        print()
        
        # Update context (optional, may want to trim if too long)
        context = response + "\n"
        
        # The model has max_seq_len, so we might not want context to grow indefinitely.
        # But `generate` only takes the last max_seq_len tokens anyway.

if __name__ == "__main__":
    main()
