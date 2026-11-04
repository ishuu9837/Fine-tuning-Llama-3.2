# Fine-tuning Llama 3.2 for Finance Question Answering

Fine-tune `meta-llama/Llama-3.2-3B-Instruct` with 4-bit QLoRA, then compare the base and adapted model on held-out questions.

[Open the notebook in Colab](https://colab.research.google.com/github/ishuu9837/Fine-tuning-Llama-3.2/blob/main/Fine_Tuning_Llama_3_2_for_Domain_Specific_(Finance)_Question_Answering_using_LoRA.ipynb)

## Run

1. Select a CUDA GPU runtime in Colab (T4 supported).
2. Run the dependency installation cell, restart the runtime, then continue from the login cell. Accept Meta's model license on Hugging Face first and put a read token in the Colab secret named `collab`.
3. Run the remaining cells in order. The notebook selects FP16 on T4 and BF16 on supported GPUs.
4. Read the base/fine-tuned comparison table and download the adapter plus `evaluation_results/metrics.json` and `comparison.csv`.

Training uses up to 3,000 examples for one epoch, with LoRA rank 16 and alpha 32. Only answer completions contribute to training loss. Normalized duplicate questions are removed before separate training, validation, and final test splits. Validation selects the checkpoint; the final test is used only for reporting.

## Numbers shown after a run

| Metric | Meaning | Better direction |
|---|---|---|
| Exact-match accuracy (%) | Generated answer equals reference after case/whitespace normalization; signs and decimals are preserved | Higher |
| Answer token F1 (%) | Mean word overlap F1 against reference answers | Higher |
| Next-token accuracy (%) | Correct next-token predictions on reference answers given their correct preceding tokens | Higher |
| Answer loss / perplexity | Token-weighted likelihood on answer tokens, excluding the question | Lower |
| BLEU / ROUGE (%) | Generated answer overlap with reference text | Higher |

There is no single factual accuracy percentage for this open-ended dataset. Exact match can penalize valid paraphrases, and next-token accuracy is teacher-forced, not a measure of independently generated financial correctness. Review the exported answers as well as the scores. BLEU and ROUGE are displayed on a 0–100 scale; table changes for percentage metrics are percentage points.

Final evaluation defaults to 50 held-out questions with deterministic generation (up to 256 new tokens). Increase `raw_test` for a larger sample. Likelihood evaluation uses a 1,024-token window and reports truncated/skipped examples and the number of scored answer tokens. Long references and generation limits affect the scores. The dataset (`gbharti/finance-alpaca`) also contains general instruction examples, so results do not establish finance-only performance.

The base comparison disables the LoRA adapter on the same quantized model, avoiding a second model allocation on the T4. Both models use identical test examples and decoding settings. Results include package versions, sample counts, settings, references, and both models' generated answers.

## Earlier notebook output (historical)

The original committed outputs reported perplexity **30.5891 → 3.9267**, BLEU **0.0733**, ROUGE-1 **0.4018**, ROUGE-2 **0.1774**, and ROUGE-L **0.3372**. These used the old evaluation procedure, including full-chat perplexity and a shared validation/evaluation pool. They are not measurements from the corrected pipeline and do not establish an accuracy level. Fresh accuracy numbers appear only after running the revised notebook; none are fabricated here.

## Validation

Run `python3 -m unittest discover -s tests -v` for CPU checks of the scoring rules and notebook structure. A full training/evaluation run requires CUDA, model access, and dataset downloads; CPU checks do not validate GPU execution.

API references: [TRL SFTTrainer](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer), [PEFT adapter disabling](https://huggingface.co/docs/peft/main/package_reference/peft_model).
