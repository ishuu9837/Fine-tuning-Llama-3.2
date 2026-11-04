#!/usr/bin/env python3
"""Validate published results; optionally check the original private archive.

Uses only the standard library. Never unpickles training arguments or loads a model.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            f"Value mismatch: {label}")


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def answer_scores(examples, name):
    exact, f1 = 0, 0.0
    for example in examples:
        prediction = " ".join(example[name].lower().split())
        reference = " ".join(example["reference"].lower().split())
        exact += prediction == reference
        a, b = prediction.split(), reference.split()
        overlap = sum((Counter(a) & Counter(b)).values())
        f1 += 2 * overlap / (len(a) + len(b)) if a or b else 1.0
    return {"exact_match_accuracy_pct": 100 * exact / len(examples),
            "answer_token_f1_pct": 100 * f1 / len(examples)}


def inspect_adapter(content, config):
    require(len(content) >= 8, "Missing Safetensors header")
    header_size = struct.unpack("<Q", content[:8])[0]
    require(0 < header_size <= len(content) - 8, "Invalid header size")
    header = json.loads(content[8:8 + header_size])
    header.pop("__metadata__", None)
    rank = config["r"]
    targets, layers, ranges, parameters = set(), set(), [], 0
    pattern = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.(?:mlp|self_attn)\."
                         r"([a-z_]+)\.lora_([AB])\.weight")
    for key, tensor in header.items():
        match = pattern.fullmatch(key)
        require(match is not None, "Unexpected adapter tensor name")
        layer, module, side = match.groups()
        targets.add(module)
        layers.add(int(layer))
        shape = tensor["shape"]
        require(tensor["dtype"] == "F32" and len(shape) == 2
                and all(isinstance(n, int) and n > 0 for n in shape),
                "Unexpected tensor dtype or shape")
        require(shape[0 if side == "A" else 1] == rank, "LoRA rank mismatch")
        partner = key.replace(f".lora_{side}.", f".lora_{'B' if side == 'A' else 'A'}.")
        require(partner in header, "Missing LoRA partner")
        start, end = tensor["data_offsets"]
        size = math.prod(shape)
        require(0 <= start < end and end - start == size * 4,
                "Invalid tensor byte range")
        ranges.append((start, end))
        parameters += size
    ranges.sort()
    require(bool(ranges) and ranges[0][0] == 0, "Missing tensor data")
    require(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])),
            "Noncontiguous tensor byte ranges")
    require(ranges[-1][1] == len(content) - 8 - header_size,
            "Tensor file size mismatch")
    require(targets == set(config["target_modules"]), "Target module mismatch")
    require(layers == set(range(len(layers)))
            and len(header) == len(layers) * len(targets) * 2,
            "Incomplete layer/module coverage")
    return {"tensor_count": len(header), "parameter_count": parameters,
            "dtype": "F32", "layer_count": len(layers),
            "target_modules": sorted(targets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    report = json.loads((ROOT / "evaluation_results/metrics.json").read_text())
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    config = json.loads((ROOT / "artifacts/adapter_config.json").read_text())
    require("examples" not in report, "Raw examples must not be in public metrics")
    require(config["base_model_name_or_path"] == report["model_id"],
            "Model identifier mismatch")
    rows = list(csv.DictReader(io.StringIO(
        (ROOT / "evaluation_results/comparison.csv").read_text())))
    results = report["results"]
    names = [row[""] for row in rows]
    require(len(names) == len(set(names)) and set(names) == set(results["base"])
            == set(results["fine_tuned"]), "Metric keys mismatch")
    for row in rows:
        metric = row[""]
        for name in ("base", "fine_tuned"):
            close(float(row[name]), results[name][metric], f"{name}/{metric}")
        close(float(row["change"]), results["fine_tuned"][metric]
              - results["base"][metric], f"delta/{metric}")
    for name, scores in results.items():
        close(scores["answer_perplexity"], math.exp(scores["answer_loss"]), name)
        for metric, value in scores.items():
            if metric.endswith("_pct"):
                require(0 <= value <= 100, f"Out-of-range metric: {metric}")
        for metric in ("evaluated_answer_tokens", "skipped_examples", "truncated_examples"):
            require(isinstance(scores[metric], int) and scores[metric] >= 0,
                    f"Invalid counter: {metric}")
        exact_count = scores["exact_match_accuracy_pct"] * report["test_examples"] / 100
        close(exact_count, round(exact_count), f"{name}/exact count")
    for member in manifest["members"]:
        if member["publication"] == "copied":
            content = (ROOT / member["repository_path"]).read_bytes()
            require(len(content) == member["bytes"] and sha256(content) == member["sha256"],
                    "Published copy differs from archive inventory")
    print("PASS: published metrics, CSV deltas, exp(loss), counters, and copied-file hashes")

    if args.archive:
        require(sha256(args.archive.read_bytes()) == manifest["archive_sha256"],
                "Archive SHA-256 mismatch")
        with zipfile.ZipFile(args.archive) as archive:
            expected_names = {m["archive_path"] for m in manifest["members"]}
            actual_names = [m.filename for m in archive.infolist() if not m.is_dir()]
            require(set(actual_names) == expected_names
                    and len(actual_names) == len(expected_names), "Archive members mismatch")
            for member in manifest["members"]:
                content = archive.read(member["archive_path"])
                require(len(content) == member["bytes"]
                        and sha256(content) == member["sha256"], "Archive member hash mismatch")
            original = json.loads(archive.read("evaluation_results/metrics.json"))
            examples = original.pop("examples")
            require(original == report, "Published metrics differ from original metadata/results")
            require(len(examples) == report["test_examples"] > 0, "Test example count mismatch")
            for name in ("base", "fine_tuned"):
                for metric, value in answer_scores(examples, name).items():
                    close(value, results[name][metric], f"recomputed {name}/{metric}")
            stats = inspect_adapter(archive.read(
                "llama3-finance-lora-final/adapter_model.safetensors"), config)
            require(stats == manifest["adapter_inspection"], "Adapter statistics mismatch")
        print("PASS: archive/member integrity, unchanged aggregate metadata, all 50 EM/F1 pairs, adapter structure")
    print("No training/inference rerun; BLEU/ROUGE and model likelihoods were not independently recomputed.")


if __name__ == "__main__":
    main()
