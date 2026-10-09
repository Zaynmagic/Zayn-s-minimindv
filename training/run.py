# Align the projector, tune image/chat responses, and export.

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
from model.model_vlm import VisionLanguageModel, VisionLanguageConfig
from .data import records, batch

ROOT = Path(__file__).resolve().parents[1]


def digest(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


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


def move(value, device):
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value.to(device)


def export(model, tokenizer, destination):
    VisionLanguageConfig.register_for_auto_class()
    VisionLanguageModel.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    model.processor.save_pretrained(destination)


def execute(args):
    if args.steps < 1 or args.accumulation < 1 or args.batch_size < 1:
        raise ValueError("positive step, batch, and accumulation counts required")
    if args.lr <= 0 or args.length < 2:
        raise ValueError("positive learning rate and sequence length of at least two are required")
    stages = args.stages.split(",")
    if any(stage not in {"align", "sft"} for stage in stages):
        raise ValueError("supported stages: align,sft")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 requires CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if device.type == "cuda":
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    values = json.loads(Path(args.config).read_text())
    if args.vision_source:
        values.pop("vision_config", None)
        values.pop("vision_source", None)
        model = VisionLanguageModel.with_pretrained_vision(values, args.vision_source, args.vision_revision)
    elif args.allow_random_vision:
        model = VisionLanguageModel(VisionLanguageConfig(**values))
    else:
        raise ValueError(
            "provide --vision-source or explicitly select --allow-random-vision for fixture validation"
        )
    if args.llm_checkpoint:
        state = torch.load(args.llm_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
        if unexpected or any(not key.startswith(("vision_encoder.", "vision_proj.")) for key in missing):
            raise ValueError("LLM checkpoint architecture does not match VLM config")
    elif not args.allow_random_llm:
        raise ValueError("provide --llm-checkpoint or explicitly select --allow-random-llm")
    if args.length > model.config.max_position_embeddings:
        raise ValueError("sequence length exceeds model context")
    model.to(device)
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
        "parameters": sum(p.numel() for p in model.parameters()),
        "vision_parameters": sum(
            p.numel() for p in model.vision_encoder.parameters()
        ),
        "language_parameters": sum(
            p.numel() for p in model.model.parameters()
        ),
        "projector_parameters": sum(
            p.numel() for p in model.vision_proj.parameters()
        ),
        "stages": [],
    }
    for stage in stages:
        configure_stage(model, stage)
        vision_before = digest(model.vision_encoder)
        language_before = digest(model.model)
        projector_before = digest(model.vision_proj)
        active = (
            DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None
            )
            if dist.is_initialized()
            else model
        )
        optimizer = torch.optim.AdamW([p for p in active.parameters() if p.requires_grad], lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
        data_path = Path(args.data) / f"{stage}.jsonl"
        rows = records(data_path)
        losses = []
        active.train()
        started = time.perf_counter()
        for start in range(0, args.steps, args.accumulation):
            window = min(args.accumulation, args.steps - start)
            optimizer.zero_grad(set_to_none=True)
            for index in range(window):
                microstep = start + index
                rank = dist.get_rank() if dist.is_initialized() else 0
                size = dist.get_world_size() if dist.is_initialized() else 1
                selected = [
                    rows[((microstep * size + rank) * args.batch_size + j) % len(rows)]
                    for j in range(args.batch_size)
                ]
                inputs = move(
                    batch(
                        selected,
                        tokenizer,
                        model.processor,
                        model.config,
                        args.length,
                        data_path.parent,
                    ),
                    device,
                )
                precision = (
                    nullcontext()
                    if args.precision == "fp32"
                    else torch.autocast(
                        device.type,
                        dtype=torch.bfloat16
                        if args.precision == "bf16"
                        else torch.float16,
                    )
                )
                sync = (active.no_sync() if dist.is_initialized() and index < window - 1 else nullcontext())
                with sync, precision:
                    output = active(**inputs)
                    loss = output.loss + output.aux_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite VLM loss")
                    scaler.scale(loss / window).backward()
                losses.append(loss.item())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for p in active.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient norm")
            scaler.step(optimizer)
            scaler.update()
        result = {
            "stage": stage,
            "losses": losses,
            "seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vision_unchanged": digest(model.vision_encoder) == vision_before,
            "language_unchanged": digest(model.model) == language_before,
            "projector_changed": digest(model.vision_proj) != projector_before,
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        }
        if not result["vision_unchanged"] or (stage == "align" and not result["language_unchanged"]):
            raise AssertionError("frozen parameters changed")
        if not result["projector_changed"]:
            raise AssertionError("projector did not update")
        report["stages"].append(result)
        if primary():
            Path(args.output).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config.to_dict(),
                    "stage": stage,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                Path(args.output) / f"{stage}.pt",
            )
            (Path(args.output) / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    if primary():
        export(model, tokenizer, Path(args.output) / "hf")
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
