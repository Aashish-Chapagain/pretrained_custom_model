import sentencepiece as spm 
from torch.utils.data import Dataset
import torch 
import numpy as np 

spm.SentencePieceTrainer.Train(
    input = 'corpus.txt',
    model_prefix = 'tokenizer',
    vocab_size = 5000, 
    model_type = 'bpe',
    character_coverage = 0.9995,
    normalization_rule_name = 'identity',
    pad_id = 0,
    unk_id = 1,
    bos_id = 2,
    eos_id = 3, 
    eos_piece = '</s>'
)


sp = spm.SentencePieceProcessor(model_file = "tokenizer.model")

all_tokens = []


with open("corpus.txt", 'r', encoding='utf-8') as f :
    for line in f: 
       tokens = sp.encode(line.strip())
       all_tokens.extend(tokens)


print(f"Total tokens: {len(all_tokens)}")


all_tokens = np.array(all_tokens, dtype = np.uint16)
np.save("tokenized_corpus.npy", all_tokens)
