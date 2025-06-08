# Fair Diffusion: Mitigating Demographic Bias in Text-to-Image Models

This project explores demographic bias in AI-generated images produced by **Stable Diffusion**, with a focus on **race and gender representation**. We implement a **four-phase pipeline** that identifies, evaluates, and mitigates societal biases using a combination of **prompt engineering, output filtering**, and **fine-tuning via DreamBooth + LoRA**. The final outcome demonstrates how careful intervention can reduce visual stereotypes in generated content.

## 🚩 Problem Statement

Text-to-image diffusion models such as *Stable Diffusion* often replicate or amplify real-world stereotypes. Neutral prompts like “a CEO” or “a criminal” frequently yield biased imagery (e.g., white males or Black individuals, respectively), reinforcing harmful narratives.

## 🎯 Objectives

* Detect and quantify demographic bias in generated images.
* Apply lightweight mitigation techniques (prompt rephrasing, filtering).
* Fine-tune the model to intrinsically reduce bias.
* Evaluate model outputs using demographic classifiers and statistical fairness metrics.

## 🧩 Pipeline Overview

### Phase 1: Preprocessing & Image Generation

* Generated 10 images per neutral/stereotype-sensitive prompt using Stable Diffusion v1.4.
* Prompts include: `"a nurse"`, `"a criminal"`, `"a software engineer"`, etc.

### Phase 2: Demographic Detection

* Used [FairFace](https://github.com/joojs/fairface) to detect perceived race and gender in generated faces.
* Stored structured outputs in CSV/JSON for analysis.

### Phase 3: Bias Optimization

* **Prompt Engineering**: Soft cues (e.g., “diverse ethnicity”) to encourage balanced outputs.
* **Output Filtering**: Discarded non-human or demographically misaligned results.
* **Combined**: Sequential use of both techniques for improved results.

### Phase 4: Model Fine-Tuning

* Fine-tuned Stable Diffusion using **DreamBooth + LoRA** for minimal compute cost.
* Curated training pairs (e.g., “a criminal” → images of Black women) to counteract bias.
* Trained for 800 steps using PEFT and Hugging Face’s Diffusers.

## 📊 Evaluation

| Method             | KL (Gender) | KL (Race) | Observations                               |
| ------------------ | ----------- | --------- | ------------------------------------------ |
| Baseline           | 0.1201      | 0.6375    | Female and White overrepresentation        |
| Prompt Engineering | 0.2882      | 0.7699    | Gender overcorrected; race bias persisted  |
| Output Filtering   | 0.1529      | 1.0986    | Ineffective at addressing racial imbalance |
| Combined           | 0.0566      | 0.3790    | Strongest among light methods              |
| Fine-Tuned Model   | 0.0201      | 0.3749    | Best gender parity and race fairness       |

Metrics:

* **KL Divergence** from ideal uniform demographic distribution
* Human audits for realism, fairness, and stereotype reinforcement

## 📁 Project Structure

```
├── 526_final_完整版.ipynb         # Full pipeline code (Colab-ready)
├── train_dreambooth_lora.py      # Fine-tuning data and weights
├── fairface_label_train.csv      # Demographic labels (race, gender) for training set
├── fairface_label_val.csv        # Demographic labels (race, gender) for validation set
├── README.md                     # This file
```

## 🔍 Limitations & Future Work

* Manual prompt engineering is time-consuming.
* Current evaluation only covers race and gender.
* Future work includes:

  * Auto prompt generation via LLMs
  * RLHF-based fine-tuning for fairness
  * A benchmark dataset for fairness in generative models

## 📎 References

1. AlDahoul, N., Rahwan, T., & Zaki, Y. (2025). *AI-generated faces influence gender stereotypes and racial homogenization*. Scientific Reports, 15. [https://doi.org/10.1038/s41598-025-99623-3](https://doi.org/10.1038/s41598-025-99623-3)
2. Lu, Y. et al. (2023). *LLMScore: Unveiling the power of large language models in text-to-image synthesis evaluation*. NeurIPS. [https://openreview.net/forum?id=OJ0c6um1An](https://openreview.net/forum?id=OJ0c6um1An)
