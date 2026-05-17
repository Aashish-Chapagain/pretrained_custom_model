MODEL = {
    "vocab_size": 5000,
    "embedding_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "max_seq_len": 256,
}

DATASET = {
    "path": "tokenized_corpus.npy",
    "sequence_length": 256,
    "stride": 64,
}

TRAINING = {
    "checkpoint_path": "minillm_checkpoint.pth",
    "epoch_save_interval": 20,
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 16,
}

CORPUS = {
    "dataset_name": "lilithyu/kaggle-child-stories",
    "output_path": "corpus.txt",
}

CHECKDATA = {
    "token_estimate_divisor": 4,
}

TOKENIZER = {
    "corpus_path": "corpus.txt",
    "model_prefix": "tokenizer",
    "vocab_size": 5000,
    "model_type": "bpe",
    "character_coverage": 0.9995,
    "normalization_rule_name": "identity",
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "eos_piece": "</s>",
    "tokenized_output_path": "tokenized_corpus.npy",
}
