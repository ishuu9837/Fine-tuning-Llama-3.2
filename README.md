# Fine-Tuning Llama 3.2 for Finance Question Answering with LoRA

This project fine-tunes **Meta Llama 3.2-3B-Instruct** with LoRA and 4-bit QLoRA on a subset of [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca). The dataset and the evaluated sample contain both finance questions and general instruction tasks.

## Training and evaluation

Use **[fine.ipynb](fine.ipynb)** for the current training and evaluation workflow. It deduplicates normalized instruction/input pairs before splitting the data, creates separate training, validation, and test partitions, and selects checkpoints using validation loss. The older, long-named notebook is retained as a legacy experiment; its evaluation procedure and scores differ.

- **Reported run:** 3,000 training examples, 200 validation examples, 50 final test examples; seed 42.
- **Training configuration in `fine.ipynb`:** 1 epoch, learning rate `2e-4`, batch size 1 with 8 gradient accumulation steps, training context length 512, completion-only loss, 4-bit NF4 quantization, and paged AdamW 8-bit optimizer. The saved notebook output identifies a Colab Tesla T4 GPU.
- **Verified adapter configuration:** rank 16, alpha 32, dropout 0.05, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. The supplied Safetensors header contains **24,313,856 adapter parameters** across 392 float32 tensors.
- **Evaluation configuration:** greedy decoding (`do_sample=False`), at most 256 generated tokens; reference-answer likelihood uses a 1,024-token conversation limit. The base comparison disables the LoRA adapter on the same quantized model.

## Results from the supplied archive

These results come from `llama3-finance-results.zip` and match the saved evaluation output in `fine.ipynb`. Aggregate values are preserved in [metrics.json](evaluation_results/metrics.json) and the original [comparison.csv](evaluation_results/comparison.csv).

| Metric | Base | Fine-tuned | Change |
|---|---:|---:|---:|
| Exact match (%) | 0.00 | 2.00 | +2.00 pp |
| Answer token F1 (%) | 21.03 | 39.18 | +18.14 pp |
| Answer loss | 2.3886 | 1.9390 | −0.4496 |
| Answer perplexity | 10.8986 | 6.9520 | −3.9466 (36.21% lower) |
| Next-token accuracy (%) | 49.40 | 55.17 | +5.76 pp |
| BLEU (×100) | 3.61 | 7.30 | +3.68 pp |
| ROUGE-1 (×100) | 25.29 | 44.48 | +19.19 pp |
| ROUGE-2 (×100) | 8.83 | 22.93 | +14.10 pp |
| ROUGE-L (×100) | 18.37 | 37.40 | +19.03 pp |
| ROUGE-Lsum (×100) | 20.06 | 38.06 | +18.00 pp |

`pp` means percentage points; changes use the full-precision values. Each model scored **3,678 reference-answer tokens**, with **0 skipped examples** and **1 truncated example**. Exact match was 0/50 for the base model and 1/50 for the fine-tuned model after lowercasing and whitespace normalization; punctuation and number formatting are preserved.

Answer perplexity is `exp(answer_loss)`, with prompt tokens masked and answer tokens (including the chat ending token) scored under teacher forcing. Next-token accuracy also uses the correct preceding reference tokens: it is not the percentage of questions answered correctly. BLEU, ROUGE, and whitespace-token F1 measure overlap with a reference, rather than factual correctness.

The previous README's 30.59 → 3.93 perplexity and 87.16% reduction belong to a different evaluation and are superseded here. They are not comparable to this answer-only evaluation.

## Validation and limits

The archive's CSV and JSON agree for every metric and delta. Exact match and answer token F1 were independently recomputed from all 50 saved prediction/reference pairs; both match the reported values. Perplexity was checked against `exp(loss)`. The supplied adapter's tensor shapes, byte ranges, and parameter count were checked without loading model code or deserializing `training_args.bin`.

**Training and model inference were not rerun.** Loss, next-token accuracy, BLEU, and ROUGE are reported from the supplied run; CSV agreement and notebook outputs corroborate them, but do not constitute an independent inference evaluation. The archive does not pin the dataset/model revisions, record partition IDs, or contain complete training/checkpoint-selection logs, so exact reproduction and partition membership cannot be independently established from it alone.

The 50-example test set is small and contains many general tasks, including stories, translation, and programming. These results show improved reference overlap and lower reference-answer loss on that mixed sample; they do not establish finance-specific factual accuracy, reliability for financial decisions, or statistical significance. Future evaluation should use a larger, independently reviewed finance-only test set, pinned revisions, saved split IDs, and checks for numerical and factual correctness.

## Saved artifacts

- [Evaluation notes](evaluation_results/README.md): metric definitions, provenance, and validation scope.
- [Aggregate metrics](evaluation_results/metrics.json) and [comparison CSV](evaluation_results/comparison.csv): full-precision results. Raw prompts, references, and generated answers from the archive are omitted from the published JSON to minimize disclosure of free-text financial narratives.
- [Adapter configuration](artifacts/adapter_config.json) and [archive manifest](artifacts/manifest.json): original configuration plus SHA-256 hashes and sizes of every archived file.
- [Artifact notes](artifacts/README.md): inventory and exclusions. The 92.80 MiB adapter weights, tokenizer assets, pickle-based training arguments, and placeholder model card are not committed. The adapter configuration alone cannot load the fine-tuned model; obtain the original adapter weights separately before using it.

Validate the published summaries with Python's standard library:

```bash
python3 scripts/validate_results.py
```

If you have the original archive, also verify its hashes and recompute exact match/F1:

```bash
python3 scripts/validate_results.py --archive /path/to/llama3-finance-results.zip
```

The run's recorded versions are PyTorch `2.11.0+cu130`, Transformers `4.57.6`, TRL `0.23.1`, PEFT `0.18.1`, and Datasets `4.8.5`. These are provenance values from the archive, not a newly tested environment.
