from datasets import load_dataset
import re

ds = load_dataset("lilithyu/kaggle-child-stories")


with open("dataset.txt", "w", encoding="utf-8") as f:
    for example in ds["train"]:
        text = example["text"].strip()
        text = re.sub(r"\s+", " ", text)  # Replace multiple whitespace with a single space
    

        if text : 
         f.write(text + "\n")

            