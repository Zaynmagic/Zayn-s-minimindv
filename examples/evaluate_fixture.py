"""Check an exported model on the four training fixtures, not a VQA benchmark."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor

from inference import predict
from training.data import build_batch, read_records

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate(model_path):
    model_path = Path(model_path)
    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    processor = SiglipImageProcessor.from_pretrained(model_path, local_files_only=True)
    data_path = ROOT / "data/synthetic/sft.jsonl"
    rows = read_records(data_path)
    batch = build_batch(rows, tokenizer, processor, model.config, 64, data_path.parent)
    output = model(**batch)
    token_count = int(batch["labels"][:, 1:].ne(-100).sum())
    loss = float(output.loss)

    examples = []
    for row in rows:
        question = row["conversations"][0]["content"].replace("<image>", "").strip()
        target = row["conversations"][1]["content"]
        with Image.open(data_path.parent / row["image"]) as image:
            answer = predict(image, question, 32, model, tokenizer, processor)
        examples.append(
            {
                "image": f"data/synthetic/{row['image']}",
                "question": question,
                "answer": answer,
                "target": target,
                "exact_match": answer == target,
            }
        )

    return {
        "scope": "In-sample check on four colored-square training fixtures; not a held-out VQA evaluation.",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_sha256": hashlib.sha256(
            (model_path / "model.safetensors").read_bytes()
        ).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "device": "cpu",
        "precision": "fp32",
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "assistant_tokens": token_count,
        "assistant_token_cross_entropy": loss,
        "assistant_token_perplexity": math.exp(loss),
        "exact_matches": sum(example["exact_match"] for example in examples),
        "example_count": len(examples),
        "examples": examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted local hf/ export")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(evaluate(args.model), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
