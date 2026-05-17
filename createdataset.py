import re

from datasets import load_dataset

from config.settings import CORPUS

ds = load_dataset(CORPUS["dataset_name"])


with open(CORPUS["output_path"], "w", encoding="utf-8") as f:
    for example in ds["train"]:
        text = example["text"].strip()
        text = re.sub(r"\s+", " ", text)  

        if text : 
         f.write(text + "\n")

            