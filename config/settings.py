# -----------------------------------------------------------------------------
# Hardware Presets & Sizing Quick Guide:
# - Low VRAM / CPU (2-4GB):  batch_size: 4-8,  sequence_length: 128, dim: 192, layers: 4
# - Mainstream (6-8GB VRAM): batch_size: 16,   sequence_length: 256, dim: 256, layers: 6 (Default)
# - High-End (12-16GB VRAM): batch_size: 32+,  sequence_length: 512, dim: 384, layers: 8-12
# - Cloud / DC (24GB+ VRAM): batch_size: 128+, sequence_length: 512-1024, dim: 768, layers: 12-16
# -----------------------------------------------------------------------------

MODEL = {
    "vocab_size": 5000,
    "embedding_dim": 256,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 256,
    "ffn_dim": 1024,
    "dropout": 0.1,
}

DATASET = {
    "path": "tokenized_corpus.npy",
    "sequence_length": 256,
    "stride": 256,  # Keep equal to sequence_length for clean non-overlapping chunks
}

TRAINING = {
    "checkpoint_path": "minillm_checkpoint.pth",
    "final_model_path": "final_model.pth",
    "epoch_save_interval": 1,
    "learning_rate": 3e-4,
    "epochs": 10,
    "batch_size": 16,  # Tune according to available VRAM (e.g. 4-8 for <6GB, 16 for 8GB, 32-64 for >=12GB)
    "resume": True,
}

CORPUS = {
    "dataset_name": "HuggingFaceTB/cosmopedia-100k",
    "split": "train",
    "text_column": "text",
    "output_path": "corpus.txt",
    "max_samples": None,
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
    "prompt": "Artificial intelligence is",
    "max_new_tokens": 100,
    "temperature": 0.35,
    "top_k": 10,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}

