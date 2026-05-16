import numpy as np
import sentencepiece as spm

CORPUS_PATH = "corpus.txt"
MODEL_PREFIX = "tokenizer"
VOCAB_SIZE = 5000
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = 0.9995
NORMALIZATION_RULE_NAME = "identity"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
EOS_PIECE = "</s>"
TOKENIZED_OUTPUT_PATH = "tokenized_corpus.npy"

spm.SentencePieceTrainer.Train(
    input=CORPUS_PATH,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,
    character_coverage=CHARACTER_COVERAGE,
    normalization_rule_name=NORMALIZATION_RULE_NAME,
    pad_id=PAD_ID,
    unk_id=UNK_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    eos_piece=EOS_PIECE,
)


sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")

all_tokens = []


with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f: 
       tokens = sp.encode(line.strip())
       all_tokens.extend(tokens)


print(f"Total tokens: {len(all_tokens)}")


all_tokens = np.array(all_tokens, dtype=np.uint16)
np.save(TOKENIZED_OUTPUT_PATH, all_tokens)
