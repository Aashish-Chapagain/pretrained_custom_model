import sys
import sentencepiece as spm
from config.settings import TOKENIZER
from generate import load_model, chat_loop
from train import resolve_device


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = resolve_device()
    print(f"Using device: {device}")

    tokenizer = spm.SentencePieceProcessor(
        model_file=f"{TOKENIZER['model_prefix']}.model"
    )
    model = load_model(device)

    chat_loop(model=model, tokenizer=tokenizer, device=device)


if __name__ == "__main__":
    main()

