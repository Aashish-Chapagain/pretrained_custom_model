import argparse
import os
import re

from datasets import load_dataset
from tqdm import tqdm

from config.settings import CORPUS


def main() -> None:
    parser = argparse.ArgumentParser(description="Create text corpus from Hugging Face dataset")
    parser.add_argument("--dataset", type=str, default=CORPUS["dataset_name"], help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default=CORPUS.get("split", "train"), help="Dataset split")
    parser.add_argument("--text-column", type=str, default=CORPUS.get("text_column", "text"), help="Text column name")
    parser.add_argument("--output", type=str, default=CORPUS["output_path"], help="Output file path")
    parser.add_argument("--max-samples", type=int, default=CORPUS.get("max_samples"), help="Max samples to extract")
    args = parser.parse_args()

    print(f"Loading dataset '{args.dataset}' (split='{args.split}')...")
    ds = load_dataset(args.dataset, split=args.split)
    
    total = len(ds)
    if args.max_samples and args.max_samples < total:
        total = args.max_samples

    print(f"Writing text corpus to '{args.output}' ({total:,} samples)...")
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for example in tqdm(ds, total=total, desc="Extracting corpus"):
            if args.max_samples and count >= args.max_samples:
                break
            raw_text = example.get(args.text_column, "")
            if not raw_text:
                continue
            text = raw_text.strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                f.write(text + "\n")
                count += 1

    print(f"Corpus creation complete. Saved {count:,} documents to '{args.output}'.")


if __name__ == "__main__":
    main()

