"""Train the image projector, then tune the language model on image/chat pairs."""

import argparse
from contextlib import nullcontext
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from model.model_vlm import VisionLanguageConfig, VisionLanguageModel
from .data import build_batch, read_records

ROOT = Path(__file__).resolve().parents[1]


def digest(module):
    checksum = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        checksum.update(name.encode())
        raw_bytes = tensor.detach().cpu().contiguous().view(torch.uint8)
        checksum.update(raw_bytes.numpy().tobytes())
    return checksum.hexdigest()


def primary():
    return not dist.is_initialized() or dist.get_rank() == 0


def configure_stage(model, stage):
    if stage not in {"align", "sft"}:
        raise ValueError("supported stages: align,sft")
    model.requires_grad_(False)
    model.vision_proj.requires_grad_(True)
    if stage == "sft":
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    model.vision_encoder.requires_grad_(False).eval()


def move_to_device(value, device):
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value.to(device)


def precision_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype)


def export(model, tokenizer, destination):
    VisionLanguageConfig.register_for_auto_class()
    VisionLanguageModel.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    model.processor.save_pretrained(destination)


def build_model(args):
    settings = json.loads(Path(args.config).read_text())
    if args.vision_source:
        model = VisionLanguageModel.with_pretrained_vision(
            settings, args.vision_source, args.vision_revision
        )
    elif args.allow_random_vision:
        model = VisionLanguageModel(VisionLanguageConfig(**settings))
    else:
        raise ValueError("provide --vision-source or --allow-random-vision")

    if args.llm_checkpoint:
        checkpoint = torch.load(
            args.llm_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_ties_embeddings = checkpoint_config.get("tie_word_embeddings")
        if (
            checkpoint_ties_embeddings is not None
            and checkpoint_ties_embeddings != model.config.tie_word_embeddings
        ):
            raise ValueError(
                "LLM checkpoint tie_word_embeddings does not match VLM config"
            )
        weights = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        language_missing = [
            name
            for name in missing
            if not name.startswith(("vision_encoder.", "vision_proj."))
        ]
        if language_missing or unexpected:
            raise ValueError("LLM checkpoint architecture does not match VLM config")
    elif not args.allow_random_llm:
        raise ValueError("provide --llm-checkpoint or --allow-random-llm")

    if args.length > model.config.max_position_embeddings:
        raise ValueError("sequence length exceeds model context")
    return model


def train_stage(model, tokenizer, stage, args, device):
    configure_stage(model, stage)
    vision_before = digest(model.vision_encoder)
    language_before = digest(model.model)
    projector_before = digest(model.vision_proj)

    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    active_model = model
    if distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        active_model = DistributedDataParallel(model, device_ids=device_ids)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    use_scaler = device.type == "cuda" and args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    data_path = Path(args.data) / f"{stage}.jsonl"
    rows = read_records(data_path)
    losses = []
    active_model.train()
    started = time.perf_counter()
    for first_step in range(0, args.steps, args.accumulation):
        # Normalize the final update by its actual number of microbatches.
        microbatch_count = min(args.accumulation, args.steps - first_step)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(microbatch_count):
            step = first_step + offset
            first_row = (step * world_size + rank) * args.batch_size
            selected_rows = [
                rows[(first_row + index) % len(rows)]
                for index in range(args.batch_size)
            ]
            inputs = build_batch(
                selected_rows,
                tokenizer,
                model.processor,
                model.config,
                args.length,
                data_path.parent,
            )
            inputs = move_to_device(inputs, device)
            sync_context = nullcontext()
            if distributed and offset < microbatch_count - 1:
                sync_context = active_model.no_sync()
            with sync_context, precision_context(device, args.precision):
                output = active_model(**inputs)
                loss = output.loss + output.aux_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite VLM loss")
                scaler.scale(loss / microbatch_count).backward()
            losses.append(loss.item())
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite gradient norm")
        scaler.step(optimizer)
        scaler.update()

    result = {
        "stage": stage,
        "losses": losses,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "vision_unchanged": digest(model.vision_encoder) == vision_before,
        "language_unchanged": digest(model.model) == language_before,
        "projector_changed": digest(model.vision_proj) != projector_before,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    if not result["vision_unchanged"]:
        raise AssertionError("frozen vision weights changed")
    if stage == "align" and not result["language_unchanged"]:
        raise AssertionError("language weights changed during alignment")
    if not result["projector_changed"]:
        raise AssertionError("projector did not update")
    return result, optimizer, scaler


def execute(args):
    if args.steps < 1 or args.accumulation < 1 or args.batch_size < 1:
        raise ValueError("positive step, batch, and accumulation counts required")
    if args.lr <= 0 or args.length < 2:
        raise ValueError(
            "positive learning rate and sequence length of at least two required"
        )
    stages = args.stages.split(",")
    if any(stage not in {"align", "sft"} for stage in stages):
        raise ValueError("supported stages: align,sft")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 requires CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if device.type == "cuda":
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
            torch.cuda.set_device(device)
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend)

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    model = build_model(args).to(device)
    destination = Path(args.output)
    report = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kind": "vision-language-training",
        "torch": torch.__version__,
        "device": str(device),
        "precision": args.precision,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "config": model.config.to_dict(),
        "seed": args.seed,
        "vision_pretrained": bool(args.vision_source),
        "llm_checkpoint": args.llm_checkpoint,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vision_parameters": sum(
            parameter.numel() for parameter in model.vision_encoder.parameters()
        ),
        "language_parameters": sum(
            parameter.numel() for parameter in model.model.parameters()
        ),
        "projector_parameters": sum(
            parameter.numel() for parameter in model.vision_proj.parameters()
        ),
        "stages": [],
    }
    for stage in stages:
        result, optimizer, scaler = train_stage(model, tokenizer, stage, args, device)
        report["stages"].append(result)
        if primary():
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config.to_dict(),
                    "stage": stage,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                destination / f"{stage}.pt",
            )
            (destination / "metrics.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
    if primary():
        export(model, tokenizer, destination / "hf")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniMind-V")
    parser.add_argument("--config", default=str(ROOT / "configs/tiny.json"))
    parser.add_argument("--data", default=str(ROOT / "data/synthetic"))
    parser.add_argument("--output", default=str(ROOT / "runs/smoke"))
    parser.add_argument("--stages", default="align,sft")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-checkpoint")
    parser.add_argument("--vision-source")
    parser.add_argument("--vision-revision")
    parser.add_argument("--allow-random-vision", action="store_true")
    parser.add_argument("--allow-random-llm", action="store_true")
    execute(parser.parse_args())
