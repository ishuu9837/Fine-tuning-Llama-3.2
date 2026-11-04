# Archive inventory

`adapter_config.json` is copied unchanged from the uploaded archive. It identifies `meta-llama/Llama-3.2-3B-Instruct`, PEFT LoRA, rank 16, alpha 32, dropout 0.05, and seven attention/MLP projection targets. `manifest.json` records the original archive checksum, each member's size and checksum, publication decision, and inspected tensor statistics. It contains no local filesystem paths.

The following files remain in the original archive and are excluded from this Git commit:

| File | Reason |
|---|---|
| `adapter_model.safetensors` | 97,307,544 bytes (92.80 MiB); large model weights are kept out of ordinary Git history. Header and tensor layout were validated, but no inference was run. |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja` | Tokenizer assets are kept with the unpublished model bundle; the repository is publishing evaluation results and adapter metadata. |
| `training_args.bin` | Pickle-based serialized training arguments; not deserialized or published. |
| Model `README.md` | Auto-generated placeholder model card with missing-information fields, rather than a completed results report. |
| `metrics.json` → `examples` field | Omitted from the published derivative to minimize disclosure of raw prompts, references, and generated free text. Aggregate results and run metadata are unchanged. |

Neither the model bundle nor the ZIP is uploaded by this commit. The config alone is insufficient to load the adapter. The original archive is required to run the private-example validation and to obtain weights for inference. No model-hosting URL is claimed or created.
