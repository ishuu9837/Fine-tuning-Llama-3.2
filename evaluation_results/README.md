# Evaluation artifacts

Source: the supplied `llama3-finance-results.zip`, whose SHA-256 and complete inventory are in [the manifest](../artifacts/manifest.json). The values match the stored evaluation outputs in [`fine.ipynb`](../fine.ipynb), evaluated at the source commit recorded in the manifest.

- `comparison.csv` is byte-for-byte identical to the archive. The unnamed first column contains metric names. `change = fine_tuned - base`; changes in fields ending in `_pct` are percentage points.
- `metrics.json` retains all original top-level metadata and both result dictionaries, with unchanged numeric values. The original `examples` array (50 instruction/input/reference/base/fine-tuned records) is omitted to minimize public disclosure of free text. No other original field is changed.

## Protocol in `fine.ipynb`

The notebook removes empty questions/answers and deduplicates lowercased, whitespace-normalized instruction/input pairs. It splits into 80% training and 20% held-out data, then splits the held-out partition equally into validation and test data, using seed 42. It selects up to 3,000/200/50 examples respectively. The reported archive counts are 3,000/200/50. Checkpoint selection uses validation loss. The archive lacks dataset revisions and saved partition identifiers, so the recorded counts cannot independently prove the absence of cross-partition leakage.

Base and fine-tuned generation use the same prompts and greedy decoding with `max_new_tokens=256`; base evaluation temporarily disables the adapter on the same quantized model. Reference likelihood masks prompt labels, limits the full chat to 1,024 tokens, and scores the assistant completion including its chat ending token. Loss is weighted by the number of scored answer tokens, and perplexity is its exponential. Both models report 3,678 scored tokens, no skipped examples, and one truncated example.

Exact match lowercases and collapses whitespace, preserving punctuation and decimal points. Answer token F1 averages per-example multiset overlap F1 over whitespace tokens. BLEU and ROUGE come from the notebook's `evaluate` calls and are multiplied by 100. ROUGE uses that evaluator's default aggregation; the archive does not record the evaluator revision or a confidence interval.

## What was validated

- Every CSV base/fine-tuned value and delta against the JSON.
- Both perplexities against `exp(answer_loss)` and the 36.2119162% relative reduction.
- Exact match and token F1 recomputed from all 50 archived examples using the notebook's normalization. The results match: base EM 0%, F1 21.031611767686762%; fine-tuned EM 2%, F1 39.175486804196176%.
- Every archived file's size and SHA-256, plus ZIP integrity.
- Adapter header structure, float32 byte lengths, contiguous offsets, LoRA shapes, target modules, and 24,313,856 parameters.

No training or model inference was run during this validation. Loss and next-token accuracy cannot be recomputed from saved answer text. BLEU and ROUGE were not independently recomputed; they agree between the archive's JSON/CSV and notebook output. The test sample contains general instruction tasks as well as finance questions and is not a finance-only factual benchmark.

Run `python3 scripts/validate_results.py` from the repository root. Add `--archive /path/to/llama3-finance-results.zip` to reproduce the archive hash, example-count, exact-match/F1, and Safetensors checks without executing pickle content or loading the model.
