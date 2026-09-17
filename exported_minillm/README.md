# Exported MiniLLM Package

This directory contains the trained **MiniLLM** model weights, architecture, and tokenizer bundled as a self-contained runtime package.

## Quick Start in Any Python Project

Copy this entire folder (or add it to your Python path), then:

```python
from minillm import load_model

# Load model, weights, and tokenizer automatically
llm = load_model("path/to/this/folder")

# Generate text:
print(llm.generate("Artificial intelligence is", max_new_tokens=80))

# Or launch interactive chat:
llm.chat()
```

## Contents
- `weights.pth`: Trained state dictionary weights.
- `full_model.pth`: Full PyTorch serialized model object.
- `tokenizer.model`: SentencePiece BPE tokenizer.
- `config.json`: Model architecture hyperparameters.
- `model.py`: Standalone PyTorch architecture definition of `MiniLLM`.
- `minillm.py`: Single-class runtime with generation, sampling, and chat.
