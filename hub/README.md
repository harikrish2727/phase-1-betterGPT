---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
spaces:
- Harikrish2727/BetterGPT-demo
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/finemath
- bigcode/starcoderdata
- HuggingFaceTB/cosmopedia
metrics:
- accuracy
- perplexity
tags:
- pytorch
- transformers
- causal-lm
- decoder-only
- small-language-model
- pretrained
- from-scratch
base_model:
- Harikrish2727/BetterGPT-150M
---
# BetterGPT-150M

BetterGPT-150M is a **150 million parameter decoder-only Transformer** language model pretrained from scratch using PyTorch.

It is a **base language model** and has **not** been instruction tuned. The model is intended for continued pretraining, supervised fine-tuning, research, and downstream adaptation.

BetterGPT is developed as an end-to-end engineering project that implements the complete lifecycle of building a modern small language model, including tokenizer training, dataset preparation, large-scale pretraining, and Hugging Face Transformers integration.

Despite training on just ~15B tokens, BetterGPT-150M outperforms several established ~110–160M parameter baselines (GPT-2 Small, OPT-125M, Pythia-160M) on science-reasoning benchmarks like ARC — see [Evaluation](#evaluation) for full results.

## 🚀 Live Demo

You can try out **BetterGPT-150M** directly in your browser:
👉 [**Try the Interactive Space Demo**](https://huggingface.co/spaces/Harikrish2727/BetterGPT-Demo)

---
# [Repo](https://github.com/harikrish2727/BetterGPT)

# Model Details

| Property | Value |
|----------|-------|
| Model Type | Decoder-only Transformer |
| Parameters | 150M |
| Layers | 18 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Context Length | 2048 |
| Vocabulary Size | 32,768 |
| Positional Encoding | Rotary Position Embeddings (RoPE) |
| Normalization | RMSNorm |
| Feed Forward | SwiGLU |
| Attention | PyTorch Scaled Dot Product Attention (SDPA) |
| Weight Tying | Yes |
| Framework | PyTorch |
| Library | Hugging Face Transformers |

# Evaluation

BetterGPT-150M was evaluated zero-shot (0-shot) on standard academic benchmarks using `lm-evaluation-harness`. Notably, on ARC-Easy and ARC-Challenge, BetterGPT-150M outperforms GPT-2 Small, OPT-125M, Pythia-160M, and Cerebras-GPT-111M despite being trained on roughly 15B tokens — a fraction of the 40B–300B+ tokens used for those baselines — suggesting the FineMath/Cosmopedia-heavy data mix is particularly effective for science-reasoning-style tasks relative to raw token count. Results are grouped by capability; `acc_norm` is reported where answer options vary in length (this corrects for a length bias in raw log-likelihood scoring — see [lm-eval-harness docs](https://github.com/EleutherAI/lm-evaluation-harness) for details).

### Commonsense & Physical Reasoning

| Benchmark | Metric | Score |
|---|---|---|
| PIQA | acc | 64.58% |
| WinoGrande | acc | 52.41% |
| HellaSwag | acc_norm | 36.46% |

### Knowledge & Science Reasoning

| Benchmark | Metric | Score |
|---|---|---|
| SciQ | acc | 80.70% |
| ARC-Easy | acc_norm | 48.27% |
| ARC-Challenge | acc_norm | 27.30% |
| OpenBookQA | acc_norm | 31.60% |

### Language Modeling Quality

| Benchmark | Metric | Score |
|---|---|---|
| LAMBADA (OpenAI) | acc | 27.79% |
| LAMBADA (OpenAI) | perplexity | 70.88 |

### Math & Logical Reasoning

*In progress — MathQA and LogiQA results will be added once complete.*

### Baseline Comparison (~110–160M Parameter Scale)

| Model | Params | ARC-E | ARC-C | HellaSwag | PIQA | WinoGrande | SciQ |
|---|---|---|---|---|---|---|---|
| **BetterGPT-150M (ours)** | 150M | **48.27** | **27.30** | **36.46** | **64.58** | **52.41** | **80.70** |
| GPT-2 Small | 124M | 39.7 | 22.6 | 31.4 | 62.1 | 50.7 | — |
| OPT-125M | 125M | 39.9 | 22.1 | 31.6 | 62.0 | 51.8 | — |
| Pythia-160M | 160M | 36.4–46.3* | 23.1 | 30.3 | 59.8–62.5* | 50.8–51.2 | 76.4 |
| Cerebras-GPT-111M | 111M | 35.1 | 21.0 | 27.2 | 58.1 | 49.0 | — |

*Baseline figures are drawn from published papers/reproductions using varying `lm-eval-harness` versions, which can introduce small discrepancies (ranges shown reflect this). BetterGPT-150M's own figures above are from a single consistent run.*

# Training

BetterGPT-150M was pretrained on approximately **15 billion tokens** using a two-stage curriculum together with a **Warmup–Stable–Decay (WSD)** learning rate schedule implemented using PyTorch's `LambdaLR`.

### Stage 1 — Stable Phase (~13B Tokens)

The first stage focuses on broad language acquisition using a diverse mixture of educational, web, mathematical, and programming datasets.

### Stage 2 — Annealing Phase (~2B Tokens)

The second stage increases the sampling probability of mathematics, reasoning, instructional text, and Python programming data while training with a reduced learning rate.

This curriculum is designed to adapt the model toward reasoning-intensive domains while preserving the language capabilities learned during the stable phase.

---

# Training Data

BetterGPT-150M was pretrained using publicly available datasets, including:

- FineWeb-Edu
- Cosmopedia
- FineMath
- StarCoder-Python

The datasets were streamed, interleaved, and converted into binary training shards for efficient large-scale pretraining.

Please refer to the original dataset repositories for licensing information, intended uses, and any applicable restrictions.

---

# Architecture

BetterGPT-150M implements a modern decoder-only Transformer architecture including:

- Multi-Head Self Attention
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed Forward Networks
- Weight Tying
- PyTorch Scaled Dot Product Attention (SDPA)

The repository provides two Hugging Face compatible model classes:

- **BetterGPTModel** – Base transformer model returning hidden states (`AutoModel`)
- **BetterGPTForCausalLM** – Causal language model for autoregressive text generation (`AutoModelForCausalLM`)

---

# Intended Uses

BetterGPT-150M is intended for:

- Continued pretraining
- Supervised fine-tuning
- Preference optimization
- Research
- Education
- Building downstream NLP applications

---

# Out-of-Scope Uses

BetterGPT-150M is **not** instruction tuned and is **not intended to be used directly as a conversational assistant**.

Users requiring instruction-following behavior should fine-tune the model using supervised instruction tuning or other alignment techniques.

---

# Limitations

As a relatively small pretrained language model, BetterGPT-150M has several limitations:

- May generate factually incorrect information.
- May produce hallucinated or inconsistent responses.
- Limited reasoning ability compared to significantly larger language models.
- Limited multilingual capability.
- Limited coding performance compared to larger code-specialized models.

Evaluation benchmarks are currently in progress and will be released separately.

---

# Usage

## Load for text generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True,
    device_map="auto"
)
```

### Generate text

```python
prompt = "The future of artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
## Load as a base model

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True
)
```

---

# Citation

If you use BetterGPT-150M in your work, please cite the project.

```bibtex
@software{bettergpt2026,
  title={BetterGPT: Building a Small Language Model from Scratch},
  author={Harikrishnan Vijayan},
  year={2026},
  url={https://github.com/Harikrish2727/BetterGPT}
}
```

---

# License

This model is released under the Apache License 2.0.

Please ensure that any downstream use also complies with the licenses of the datasets used during pretraining.

---

# Acknowledgements

BetterGPT is an independent engineering project inspired by modern open-source language models and the Hugging Face Transformers ecosystem.

The project draws inspiration from the broader open-source LLM community, including work such as nanoGPT, llm.c, Llama, Gemma, and Qwen.