# MiniLLM: Custom Pretrained Language Model

A lightweight, from-scratch Transformer Language Model built with **PyTorch** and **SentencePiece**, engineered for universal training and inference across **NVIDIA (CUDA), AMD (DirectML / ROCm), Intel (DirectML / XPU), Apple Silicon (MPS), and CPU**.

Pretrained on high-quality synthetic educational and scientific content from [HuggingFaceTB/cosmopedia-100k](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k) (~88.8M tokens).

---

## 📌 Architecture & Default Specifications

The default configuration is optimized for fast, stable training on consumer hardware while remaining fully scalable to workstations, cloud instances, or CPU-only setups.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Type** | Decoder-style Autoregressive Transformer | Causal masked multi-head self-attention |
| **Parameters** | ~6.08M | Lightweight, high-throughput micro-LLM |
| **Layers (`num_layers`)** | 6 | Transformer layers with pre-LayerNorm |
| **Embedding Dimension (`embedding_dim`)** | 256 | Hidden state dimension |
| **Attention Heads (`num_heads`)** | 8 | 32 dimensions per head |
| **FFN Dimension (`ffn_dim`)** | 1024 | Feed-forward intermediate projection |
| **Max Sequence Length (`max_seq_len`)** | 256 | Context window size |
| **Activation** | GELU | Gaussian Error Linear Unit |
| **Vocabulary Size (`vocab_size`)** | 5,000 | BPE tokens trained with SentencePiece |
| **Universal Hardware Support** | CUDA / DirectML / MPS / XPU / CPU | Zero code modification required |

---

## 💻 Hardware Targets & Sizing Guide

MiniLLM is hardware-agnostic and dynamically adapts to any compute device. Select a hardware profile below and adjust [`config/settings.py`](config/settings.py) accordingly:

| Target Tier | Example Hardware | VRAM / RAM | `batch_size` | `max_seq_len` / `stride` | `embedding_dim` | `num_layers` | Est. Epoch Time (100k data) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. CPU / Ultra-Light** | Intel Core, AMD Ryzen, Thin-and-Light laptops | 4 GB – 8 GB System RAM | `4` – `8` | `128` | `192` | `4` | ~45 – 90 mins |
| **2. Mainstream Consumer GPU** *(Default Preset)* | NVIDIA RTX 2060/3060/4060/5050, AMD RX 6600/7600, Intel Arc A580/A750, Apple M1/M2/M3 (8–16GB) | 6 GB – 8 GB VRAM | `16` | `256` | `256` | `6` | ~14 mins |
| **3. Enthusiast / Workstation** | NVIDIA RTX 3080/4070/4080/5080, AMD RX 7800/7900 XT, Apple M Pro/Max (32GB+) | 12 GB – 16 GB VRAM | `32` – `64` | `512` | `384` – `512` | `8` – `12` | ~5 – 8 mins |
| **4. Cloud & Data Center** | NVIDIA RTX 3090/4090, A100, H100, L4, AMD MI250/MI300 | 24 GB – 80 GB VRAM | `128` – `256` | `512` – `1024` | `512` – `768` | `12` – `16` | ~1 – 3 mins |

> [!TIP]
> **VRAM Tuning Rule**: If you encounter Out-Of-Memory (OOM) errors on lower VRAM GPUs, reduce `TRAINING["batch_size"]` (e.g. from `16` to `8` or `4`). Ensure `DATASET["stride"]` matches `DATASET["sequence_length"]` to maintain optimal non-redundant training throughput.

---

## ⚡ Platform Acceleration Matrix & Setup

The training and inference pipelines automatically probe and utilize the best accelerator available. Follow the setup matching your operating system and hardware:

### 1. Windows (DirectX 12 / DirectML)
> **Target**: AMD Radeon, Intel Arc / Iris Xe, and NVIDIA GeForce on Windows.
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. NVIDIA CUDA (Windows or Linux)
> **Target**: GeForce RTX/GTX, Quadro, Tesla, A100, H100.
```bash
python -m venv env
# On Windows: .\env\Scripts\Activate.ps1 | On Linux: source env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Apple Silicon (macOS Metal / MPS)
> **Target**: Mac M1, M2, M3, M4 (Base, Pro, Max, Ultra).
```bash
python3 -m venv env
source env/bin/activate
pip install torch
pip install -r requirements.txt
```

### 4. AMD ROCm (Linux)
> **Target**: AMD Radeon RX 7000 series, Radeon Pro, Instinct accelerators on Linux.
```bash
python3 -m venv env
source env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

### 5. Universal CPU-Only
> **Target**: Systems without a dedicated GPU accelerator.
```bash
python -m venv env
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### Accelerator Priority Order:
When launching any script (`train.py`, `generate.py`, `chat.py`, `benchmark.py`), device selection is automatically resolved in order of performance:
```
1. NVIDIA CUDA / AMD ROCm  (torch.cuda)
2. Windows DirectML / DX12 (torch_directml - AMD, Intel, NVIDIA)
3. Apple Silicon Metal     (torch.backends.mps)
4. Intel XPU / oneAPI      (torch.xpu)
5. CPU Fallback            (Standard PyTorch engine)
```

---

## 📁 Repository Structure

```text
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration (Model, Dataset, Training, Tokenizer)
├── tests/
│   ├── test_dataset_validation.py   # Unit tests for dataset slicing and validation
│   └── test_inference_validation.py # Unit tests for logits accuracy and generation
├── benchmark.py             # HellaSwag benchmark evaluation script
├── chat.py                  # Terminal interactive chatbot interface
├── checkdatasize.py         # Inspects corpus size, character count, and token estimates
├── createdataset.py         # Downloads and extracts text from Hugging Face datasets
├── datasetclass.py          # PyTorch LLMDataset with sequence stride windowing
├── generate.py              # Text generation with Top-K, Top-P, repetition penalty & streaming
├── hellaswag.py             # DeepEval HellaSwag benchmark evaluation
├── model.py                 # MiniLLM PyTorch neural network module
├── requirements.txt         # Cross-platform project dependencies
├── run_chat.bat             # One-click batch launcher for interactive chat
├── tokenizer.py             # SentencePiece BPE tokenizer trainer & corpus encoder
└── train.py                 # Main training loop with progress tracking and checkpointing
```

---

## 🔄 End-to-End Pipeline

The workflow runs in 3 sequential steps:

### Step 1: Prepare the Text Corpus
Downloads `HuggingFaceTB/cosmopedia-100k` and extracts clean educational and scientific text into `corpus.txt`:
```powershell
python createdataset.py
```
*Optional: You can limit samples for quick smoke testing:*
```powershell
python createdataset.py --max-samples 1000
```

### Step 2: Train Tokenizer & Encode Data
Trains a SentencePiece BPE tokenizer with 5,000 vocab and saves token IDs to `tokenized_corpus.npy` (~88.8M tokens):
```powershell
python tokenizer.py
```

### Step 3: Train the Model
Trains `MiniLLM` using cross-entropy loss and AdamW, with live `tqdm` progress tracking:
```powershell
python train.py
```
* Checkpoints are automatically saved to `minillm_checkpoint.pth` after each epoch.
* The final trained model weights are saved to `final_model.pth`.
* If interrupted, set `"resume": True` in `config/settings.py` to seamlessly resume from the latest checkpoint.

## 💬 Inference & Generation

### Continuous Multi-Turn Chat
Launch continuous back-and-forth conversation with sliding context memory:
```powershell
python generate.py --chat
```
* **Commands in chat**:
  * Type your prompt/question and press Enter to stream the model's reply.
  * Type `clear` or `reset` to wipe conversation memory and start fresh.
  * Type `exit` or `quit` to leave.

### Single-Prompt Generation
Generate completions with real-time token streaming and advanced sampling controls:
```powershell
python generate.py --stream --prompt "The solar system consists of"
```

#### Available CLI Options:
* `--chat`: Launches continuous multi-turn interactive chat mode.
* `--prompt`: Initial text prompt (default configured in `config/settings.py`).
* `--max-new-tokens`: Maximum number of tokens to generate (e.g. `80`).
* `--temperature`: Sampling temperature (default `0.35` for micro-LLM coherence).
* `--top-k`: Limits selection to top K tokens (default `10`).
* `--top-p`: Nucleus sampling probability cutoff (default `0.9`).
* `--repetition-penalty`: Penalizes repeated tokens to prevent looping (default `1.2`).
* `--stream`: Streams tokens to the console in real-time.

### Quick Chat Launchers
You can also launch chat via:
```powershell
# Directly:
python chat.py

# Or via batch launcher:
.\run_chat.bat
```

---

## 🧪 Testing & Benchmarks

Run automated unit tests:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Evaluate model accuracy on HellaSwag:
```powershell
# Lightweight standalone perplexity/choice benchmark:
python benchmark.py

# DeepEval HellaSwag benchmark:
python hellaswag.py
```

---

## 📦 Exporting for Other Projects

You can export the trained model architecture, weights, tokenizer, and runtime into a standalone, portable folder:

```powershell
python export.py --output exported_minillm
```

This creates a self-contained `exported_minillm/` directory containing:
* `config.json`: Hyperparameters (`vocab_size`, `embedding_dim`, `num_heads`, `num_layers`, `max_seq_len`).
* `weights.pth`: Trained state dictionary.
* `full_model.pth`: Serialized PyTorch model object.
* `tokenizer.model`: SentencePiece BPE tokenizer.
* `model.py`: Standalone architecture definition without project-root dependencies.
* `minillm.py`: Clean runtime engine.

### Using MiniLLM in Any External Python Project:
Simply copy `exported_minillm/` into your other project and use it in 2 lines:

```python
from exported_minillm.minillm import load_model

# 1. Load the model, tokenizer, and weights
llm = load_model("path/to/exported_minillm")

# 2. Generate text
print(llm.generate("Artificial intelligence is", max_new_tokens=80))

# 3. Or launch an interactive terminal chat
llm.chat()
```

---

## ⚙️ Configuration Reference (`config/settings.py`)

All hyperparameters and operational paths are centralized in [`config/settings.py`](config/settings.py):

* **`MODEL`**: Vocabulary size, embedding dimension, number of layers, attention heads, context length.
* **`DATASET`**: Sequence length and stride.
* **`TRAINING`**: Epochs, learning rate, batch size, checkpoint intervals, resume toggle.
* **`TOKENIZER`**: SentencePiece BPE parameters, character coverage, and special token IDs (`<pad>`, `<unk>`, `<s>`, `</s>`).
* **`GENERATE`**: Default generation prompt, temperature, top-k, top-p, repetition penalty.
