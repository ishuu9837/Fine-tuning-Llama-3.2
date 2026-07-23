# Fine-Tuning Llama 3.2 for Domain-Specific (Finance) Question Answering using LoRA

## Overview
This project fine-tunes **Llama 3.2-3B-Instruct** using **LoRA (Low-Rank Adaptation)** and **QLoRA (4-bit quantization)** on a finance-domain instruction dataset, with a full evaluation pipeline using BLEU, ROUGE, and perplexity metrics.

## Tech Stack
- **Model:** Llama 3.2-3B-Instruct (Meta)
- **Frameworks:** PyTorch, Hugging Face Transformers, PEFT (LoRA), TRL (SFTTrainer)
- **Quantization:** QLoRA via `bitsandbytes` (4-bit NF4)
- **Training hardware:** Google Colab (free T4 GPU)
- **Dataset:** [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca) (~69K instruction-tuned QA pairs)

## Method
1. Loaded Llama 3.2-3B-Instruct in 4-bit precision (QLoRA) to fit training on a 16GB GPU.
2. Attached LoRA adapters (`r=16`, `alpha=32`) to all attention and MLP projection layers.
3. Fine-tuned on a 3,000-example subset for 1 epoch using `SFTTrainer`.
4. Only **0.75%** of total parameters (24.3M of 3.24B) were trained.

## Results

| Metric | Base Model | Fine-Tuned | Change |
|---|---|---|---|
| **Perplexity** (held-out finance text) | 30.59 | 3.93 | **↓ 87.16%** |
| **BLEU** | — | 0.0733 | — |
| **ROUGE-1** | — | 0.4018 | — |
| **ROUGE-2** | — | 0.1774 | — |
| **ROUGE-L** | — | 0.3372 | — |

**Perplexity** is the headline result: an 87% reduction shows the model became substantially more fluent and confident on finance-domain text after fine-tuning.

**BLEU/ROUGE context:** BLEU scores in the 0.05–0.15 range are typical (not a red flag) for open-ended QA, since valid answers can be phrased many different ways without matching a single reference verbatim. ROUGE-1/L scores of ~0.34–0.40 indicate reasonable lexical and structural overlap with reference answers.

Qualitative spot-checks confirmed the fine-tuned model produces coherent, on-topic, non-repetitive answers.

## Known Limitation
The `gbharti/finance-alpaca` dataset contains a mix of genuine finance-domain QA and generic instruction-following examples (e.g., synonym lists, general how-to questions), rather than being purely finance-specific. This was discovered during qualitative evaluation. **Future work** would filter the dataset to finance-only examples (e.g., via keyword/topic filtering) to strengthen domain specificity and likely improve the reported metrics further.


## Key Learnings
- QLoRA (bitsandbytes) requires a CUDA GPU — does not run on Apple Silicon (MPS); Colab's free T4 was used instead.
- LoRA reduces trainable parameters by ~100x by decomposing weight updates into low-rank matrices (`ΔW ≈ B×A`), enabling fine-tuning of billion-parameter models on consumer/free-tier hardware.
- Perplexity and BLEU/ROUGE measure different things — perplexity reflects model confidence on held-out text, while BLEU/ROUGE measure similarity between generated and reference answers. Both are useful, neither is sufficient alone.
