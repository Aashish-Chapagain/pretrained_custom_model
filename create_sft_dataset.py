import argparse
import os
import sys
from datasets import load_dataset
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from config.settings import SFT, TOKENIZER

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_sft_dataset(
    dataset_name: str = SFT["dataset_name"],
    output_path: str = SFT["output_dataset_path"],
    max_samples: int = SFT["max_samples"],
    max_seq_len: int = SFT["max_seq_len"],
    tokenizer_path: str = f"{TOKENIZER['model_prefix']}.model",
) -> str:
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer model '{tokenizer_path}' not found.")

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0
    eos_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else 3

    print(f"Loading instruction dataset '{dataset_name}'...")
    ds = load_dataset(dataset_name, split="train")
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    print(f"Processing {len(ds):,} instruction-response pairs...")

    all_input_ids = []
    all_labels = []

    skipped_too_long = 0
    skipped_empty = 0

    for item in tqdm(ds, desc="Tokenizing & masking SFT data"):
        instruction = (item.get("instruction") or "").strip()
        inp = (item.get("input") or "").strip()
        response = (item.get("output") or "").strip()

        if not instruction or not response:
            skipped_empty += 1
            continue

        if inp:
            prompt = f"User: {instruction}\n{inp}\nAssistant: "
        else:
            prompt = f"User: {instruction}\nAssistant: "

        prompt_tokens = tokenizer.encode(prompt)
        full_tokens = tokenizer.encode(prompt + response) + [eos_id]

        # Ensure sequence fits within max_seq_len (we need input of max_seq_len)
        if len(full_tokens) > max_seq_len + 1:
            skipped_too_long += 1
            continue

        # Next-token prediction: input is full[:-1], target is full[1:]
        x = full_tokens[:-1]
        y = full_tokens[1:]

        # Loss masking: mask prompt tokens with -100 so loss is only computed on Assistant's response
        prompt_len = len(prompt_tokens)
        labels = list(y)
        mask_len = max(0, prompt_len - 1)
        labels[:mask_len] = [-100] * mask_len

        # Pad sequences up to max_seq_len
        pad_len = max_seq_len - len(x)
        if pad_len > 0:
            x = x + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

        all_input_ids.append(x)
        all_labels.append(labels)

    input_ids_arr = np.array(all_input_ids, dtype=np.int32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    print(f"\nSFT Dataset prepared:")
    print(f"  - Valid samples: {len(input_ids_arr):,}")
    print(f"  - Skipped (too long > {max_seq_len} tokens): {skipped_too_long:,}")
    print(f"  - Skipped (empty): {skipped_empty:,}")
    print(f"  - input_ids shape: {input_ids_arr.shape}")
    print(f"  - labels shape: {labels_arr.shape}")

    np.savez_compressed(
        output_path,
        input_ids=input_ids_arr,
        labels=labels_arr,
    )
    print(f"Saved SFT dataset to: '{output_path}'")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create tokenized SFT dataset with loss masking.")
    parser.add_argument("--dataset-name", default=SFT["dataset_name"])
    parser.add_argument("--output", default=SFT["output_dataset_path"])
    parser.add_argument("--max-samples", type=int, default=SFT["max_samples"])
    parser.add_argument("--max-seq-len", type=int, default=SFT["max_seq_len"])
    args = parser.parse_args()

    build_sft_dataset(
        dataset_name=args.dataset_name,
        output_path=args.output,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
