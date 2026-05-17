import numpy as np
import sentencepiece as spm

from config.settings import TOKENIZER

spm.SentencePieceTrainer.Train(
    input=TOKENIZER["corpus_path"],
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
)

sp = spm.SentencePieceProcessor(model_file=f"{TOKENIZER['model_prefix']}.model")

all_tokens = []


with open(TOKENIZER["corpus_path"], "r", encoding="utf-8") as f:
    for line in f: 
       tokens = sp.encode(line.strip())
       all_tokens.extend(tokens)


print(f"Total tokens: {len(all_tokens)}")


all_tokens = np.array(all_tokens, dtype=np.uint16)
np.save(TOKENIZER["tokenized_output_path"], all_tokens)
