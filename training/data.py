"""Load image/chat records and build assistant-only training targets."""

import json
from pathlib import Path

from PIL import Image
import torch


def read_records(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise ValueError("empty multimodal dataset")
    return rows


def prepare_messages(conversation, config):
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("each image record requires a conversation")
    reserved_tokens = ("<|im_start|>", "<|im_end|>", config.image_special_token)
    messages = []
    for message in conversation:
        if not isinstance(message, dict):
            raise ValueError("invalid chat message")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            raise ValueError("invalid chat message")
        if any(token in content for token in reserved_tokens):
            raise ValueError("reserved chat or image token in message content")
        messages.append(dict(message))

    if messages[0]["role"] != "user":
        raise ValueError("image records must start with a user message")
    first_placeholder_count = messages[0]["content"].count("<image>")
    later_placeholder = any("<image>" in message["content"] for message in messages[1:])
    if first_placeholder_count != 1 or later_placeholder:
        raise ValueError(
            "exactly one <image> placeholder is required in the first user message"
        )
    image_span = config.image_special_token * config.image_token_len
    messages[0]["content"] = messages[0]["content"].replace("<image>", image_span)
    return messages


def assistant_targets(token_ids, assistant_prefix, eos_token_id):
    targets = [-100] * len(token_ids)
    prefix_length = len(assistant_prefix)
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor : cursor + prefix_length] != assistant_prefix:
            cursor += 1
            continue
        cursor += prefix_length
        while cursor < len(token_ids):
            targets[cursor] = token_ids[cursor]
            cursor += 1
            if token_ids[cursor - 1] == eos_token_id:
                break
    if not any(target != -100 for target in targets[1:]):
        raise ValueError("no assistant labels after truncation")
    return targets


def build_batch(rows, tokenizer, processor, config, length, base):
    if not rows or not 1 <= length <= config.max_position_embeddings:
        raise ValueError(
            "a nonempty batch and a sequence length within model context are required"
        )
    image_ids = tokenizer.encode(config.image_special_token, add_special_tokens=False)
    if image_ids != config.image_ids:
        raise ValueError("tokenizer image token does not match model configuration")
    assistant_prefix = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    token_batches = []
    target_batches = []
    attention_masks = []
    images = []
    for record in rows:
        if not isinstance(record, dict) or not isinstance(record.get("image"), str):
            raise ValueError("each image record requires an image path")
        messages = prepare_messages(record.get("conversations"), config)
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)[:length]
        if token_ids.count(config.image_ids[0]) != config.image_token_len:
            raise ValueError("truncated image span; increase sequence length")
        targets = assistant_targets(token_ids, assistant_prefix, tokenizer.eos_token_id)
        padding = length - len(token_ids)
        attention_masks.append([1] * len(token_ids) + [0] * padding)
        token_batches.append(token_ids + [tokenizer.pad_token_id] * padding)
        target_batches.append(targets + [-100] * padding)
        with Image.open(Path(base) / record["image"]) as source:
            images.append(source.convert("RGB"))
    return {
        "input_ids": torch.tensor(token_batches),
        "labels": torch.tensor(target_batches),
        "attention_mask": torch.tensor(attention_masks),
        "pixel_values": dict(processor(images=images, return_tensors="pt")),
    }
