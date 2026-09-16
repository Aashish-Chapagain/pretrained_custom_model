MODEL = {
    "vocab_size": 5000,
    "embedding_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "max_seq_len": 256,
    "ffn_dim": 1024,
    "dropout": 0.1,
}

DATASET = {
    "path": "tokenized_corpus.npy",
    "sequence_length": 256,
    "stride": 64,
}

TRAINING = {
    "checkpoint_path": "minillm_checkpoint.pth",
    "final_model_path": "final_model.pth",
    "epoch_save_interval": 3,
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 16,
    "resume": True,
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

GENERATE = {
    "prompt": "Once upon a time",
    "max_new_tokens": 80,
    "temperature": 0.8,
}
