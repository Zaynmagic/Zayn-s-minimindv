"""Read one-image chat records and construct assistant-only training targets."""

import json
from pathlib import Path

from PIL import Image
import torch


def records(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty multimodal dataset")
    return rows


def batch(rows, tokenizer, processor, config, length, base):
    if not rows or length < 1 or length > config.max_position_embeddings:
        raise ValueError("a nonempty batch and a sequence length within model context are required")
    if tokenizer.encode(config.image_special_token, add_special_tokens=False) != config.image_ids:
        raise ValueError("tokenizer image token does not match model configuration")
    token_batches, target_batches, attention_masks, images = [], [], [], []
    assistant_prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    for record in rows:
        if not isinstance(record, dict) or not isinstance(record.get("image"), str):
            raise ValueError("each image record requires an image path")
        conversation = record.get("conversations")
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("each image record requires a conversation")
        messages = []
        for message in conversation:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "system"} or not isinstance(message.get("content"), str):
                raise ValueError("invalid chat message")
            if any(token in message["content"] for token in ("<|im_start|>", "<|im_end|>", config.image_special_token)):
                raise ValueError("reserved chat or image token in message content")
            messages.append(dict(message))
        if messages[0]["role"] != "user":
            raise ValueError("image records must start with a user message")
        if messages[0]["content"].count("<image>") != 1 or any("<image>" in message["content"] for message in messages[1:]):
            raise ValueError("exactly one <image> placeholder is required in the first user message")
        messages[0]["content"] = messages[0]["content"].replace(
            "<image>", config.image_special_token * config.image_token_len
        )
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)[:length]
        if token_ids.count(config.image_ids[0]) != config.image_token_len:
            raise ValueError("truncated image span; increase sequence length")
        targets = [-100] * len(token_ids)
        cursor = 0
        while cursor < len(token_ids):
            if token_ids[cursor:cursor + len(assistant_prefix)] == assistant_prefix:
                cursor += len(assistant_prefix)
                while cursor < len(token_ids):
                    targets[cursor] = token_ids[cursor]
                    cursor += 1
                    if token_ids[cursor - 1] == tokenizer.eos_token_id:
                        break
            else:
                cursor += 1
        if not any(target != -100 for target in targets[1:]):
            raise ValueError("no assistant labels after truncation")
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
