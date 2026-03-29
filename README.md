# GoSeek-V3

> A high-performance Go runtime for DeepSeek-V3 models.

---

> **Disclaimer:** This project is an independent implementation and is **not affiliated with, endorsed by, or sponsored by DeepSeek**. DeepSeek is a trademark of its respective owners. This project is a Go rewrite of software related to DeepSeek models.
>
> Original work: Copyright (c) 2023 DeepSeek
> 
> Modifications: Rewritten from Python to Go, 2026.


> Forked from: [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) / [README.md](https://github.com/deepseek-ai/DeepSeek-V3)
---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Summary](#2-model-summary)
3. [Model Downloads](#3-model-downloads)
4. [How to Run Locally](#4-how-to-run-locally)
5. [License](#5-license)
6. [Contact](#6-contact)

---

## 1. Introduction

**GoSeek-V3** is a Go reimplementation of the inference engine for DeepSeek-V3, a powerful Mixture-of-Experts (MoE) language model with 671B total parameters (37B activated per token).

This project provides a DeepSeek-compatible inference runtime written entirely in Go, targeting production deployments that benefit from Go's concurrency model, low memory overhead, and static compilation.

GoSeek-V3 preserves all capabilities of the original model:

- **Multi-head Latent Attention (MLA)** for efficient inference
- **DeepSeekMoE architecture** for cost-effective computation
- **Auxiliary-loss-free load balancing** for stable training
- **Multi-Token Prediction (MTP)** objective for stronger performance and speculative decoding

The underlying model was pre-trained on 14.8 trillion diverse, high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages. Reasoning capabilities were further enhanced via knowledge distillation from the DeepSeek-R1 series.

> This project focuses on the **inference runtime**. Model weights are sourced from the original DeepSeek-V3 release and are subject to the [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL).

---

## 2. Model Summary

### Architecture

| Property | Value |
|---|---|
| Architecture | Mixture-of-Experts (MoE) |
| Total Parameters | 671B |
| Activated Parameters (per token) | 37B |
| Context Length | 128K tokens |
| Attention | Multi-head Latent Attention (MLA) |

### Key Features

**Load Balancing:** An auxiliary-loss-free strategy minimizes performance degradation while encouraging balanced expert utilization.

**Multi-Token Prediction:** MTP improves model performance and enables speculative decoding for faster inference.

**FP8 Training:** The model was trained using an FP8 mixed precision framework, validating FP8 effectiveness at extreme scale. FP8 weights are provided natively; a conversion script for BF16 is available (see [How to Run Locally](#5-how-to-run-locally)).

**Knowledge Distillation:** Reasoning patterns from DeepSeek-R1's long Chain-of-Thought are distilled into DeepSeek-V3, improving its reasoning while preserving output style and length control.

---

## 3. Model Downloads

| Model | Total Params | Activated Params | Context Length | Download |
|---|---|---|---|---|
| DeepSeek-V3-Base | 671B | 37B | 128K | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) |
| DeepSeek-V3 | 671B | 37B | 128K | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) |

> **Note:** The total size on Hugging Face is 685B, which includes 671B of main model weights and 14B of Multi-Token Prediction (MTP) module weights.

---

## 4. How to Run Locally

GoSeek-V3 supports multiple inference backends. Since FP8 training is natively adopted in the original framework, only FP8 weights are provided. To convert to BF16:

```shell
cd inference
goseek --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights
```

### Quick Start (SGLang)

```shell
git clone https://github.com/leycm/GoSeek-V3.git
cd GoSeek-V3/interface/
go run main.go
```

Download model weights from Hugging Face into `/path/to/DeepSeek-V3`, then convert:

```shell
go run convert.go --hf-ckpt-path /path/to/DeepSeek-V3 \
  --save-path /path/to/DeepSeek-V3-Demo \
  --n-experts 256 --model-parallel 16
```

Run interactive chat:
```shell
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR \
  generate.go --ckpt-path /path/to/DeepSeek-V3-Demo \
  --config configs/config_671B.json \
  --interactive --temperature 0.7 --max-new-tokens 200
```

> **System Requirements:** Linux, Python 3.10. macOS and Windows are not supported. Hugging Face Transformers is not directly supported.
---

## 5. License

This code repository is licensed under the [MIT License](LICENSE-CODE).

The use of DeepSeek-V3 Base/Chat model weights is subject to the [DeepSeek Model License](LICENSE-MODEL). DeepSeek-V3 (Base and Chat) supports commercial use.

This project (GoSeek-V3) is an independent Go reimplementation and is not affiliated with or endorsed by DeepSeek. The DeepSeek Model License applies to the model weights and derivatives.

---

## 6. Contact

For questions about this Go runtime, please open an issue in this repository or write [leycm@proton.me](mailto:leycm@proton.me).

For questions about the underlying DeepSeek-V3 model, contact the original authors at their [repository](https://github.com/deepseek-ai/DeepSeek-V3)  or at [service@deepseek.com](mailto:service@deepseek.com).