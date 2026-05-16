import re

from datasets import load_dataset

DATASET_NAME = "lilithyu/kaggle-child-stories"
OUTPUT_PATH = "corpus.txt"

ds = load_dataset(DATASET_NAME)


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for example in ds["train"]:
        text = example["text"].strip()
        text = re.sub(r"\s+", " ", text)  

        if text : 
         f.write(text + "\n")

            