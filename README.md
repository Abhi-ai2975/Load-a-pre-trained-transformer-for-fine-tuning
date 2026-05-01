# Load-a-pre-trained-transformer-for-fine-tuning
# Local QLoRA Fine-Tuning Pipeline

An optimized, local machine learning environment for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models using QLoRA. 

## Overview
This repository contains the infrastructure and training scripts to fine-tune lightweight foundation models (like Qwen2.5-0.5B) directly on consumer hardware. It utilizes 4-bit quantization to drastically reduce VRAM requirements while maintaining training precision.

## Hardware & Software Architecture
- **OS:** Windows 11 with WSL2 (Ubuntu Subsystem)
- **Compute:** NVIDIA RTX 3050 (6GB VRAM)
- **Environment:** Python 3.10 (venv)
- **Core Stack:** PyTorch (CUDA 12.1), Hugging Face `transformers`, `peft`, `trl`, and `bitsandbytes`

## Features
- **4-Bit Quantization (NF4):** Loads base models in 4-bit precision to fit within a 6GB VRAM constraint.
- **Low-Rank Adaptation (LoRA):** Targets attention blocks (`q_proj`, `v_proj`, `k_proj`, `o_proj`) for efficient parameter updates.
- **Gradient Checkpointing & Accumulation:** Simulates larger batch sizes without triggering Out-Of-Memory (OOM) errors.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Abhi-ai2975/Load-a-pre-trained-transformer-for-fine-tuning.git](https://github.com/Abhi-ai2975/Load-a-pre-trained-transformer-for-fine-tuning.git)
   cd Load-a-pre-trained-transformer-for-fine-tuning
