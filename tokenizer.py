import os
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from config.settings import TOKENIZER


def main() -> None:
    corpus_path = TOKENIZER["corpus_path"]
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Corpus file '{corpus_path}' not found.\n"
            "Please run 'python createdataset.py' first to download and prepare the dataset."
        )

    print(f"Training SentencePiece tokenizer on '{corpus_path}'...")
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=TOKENIZER["model_prefix"],
        vocab_size=TOKENIZER["vocab_size"],
        model_type=TOKENIZER["model_type"],
        character_coverage=TOKENIZER["character_coverage"],
        normalization_rule_name=TOKENIZER["normalization_rule_name"],
        pad_id=TOKENIZER["pad_id"],
        unk_id=TOKENIZER["unk_id"],
        bos_id=TOKENIZER["bos_id"],
        eos_id=TOKENIZER["eos_id"],
        eos_piece=TOKENIZER["eos_piece"],
        max_sentence_length=32768,
    )

    model_path = f"{TOKENIZER['model_prefix']}.model"
    print(f"Tokenizer trained. Saved to '{model_path}'.")
    sp = spm.SentencePieceProcessor(model_file=model_path)
    all_tokens = []

    print("Encoding corpus into tokens...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            text = line.strip()
            if text:
                tokens = sp.encode(text)
                all_tokens.extend(tokens)

    print(f"Total tokens encoded: {len(all_tokens):,}")
    token_array = np.array(all_tokens, dtype=np.uint16)
    np.save(TOKENIZER["tokenized_output_path"], token_array)
    print(f"Saved tokenized dataset to '{TOKENIZER['tokenized_output_path']}'.")


if __name__ == "__main__":
    main()
