import os

from config.settings import CHECKDATA, CORPUS


def main() -> None:
    size_mb = os.path.getsize(CORPUS["output_path"]) / (1024 * 1024)
    print(f"Size of {CORPUS['output_path']}: {size_mb:.2f} MB")

    with open(CORPUS["output_path"], "r", encoding="utf-8") as f:
        text = f.read()
        num_tokens = len(text.split())
        char_count = len(text)

    print(f"Number of tokens: {num_tokens // CHECKDATA['token_estimate_divisor']:,}")
    print(f"Number of characters: {char_count}")


if __name__ == "__main__":
    main()
