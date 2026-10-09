"""Run a local image/question prompt against a self-contained model export."""

import argparse

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor


@torch.inference_mode()
def predict(image, question, max_new_tokens, model, tokenizer, processor):
    if image is None or not isinstance(question, str) or not question.strip():
        raise ValueError("provide an image and a question")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if any(marker in question for marker in ("<|im_start|>", "<|im_end|>", "<image>", model.config.image_special_token)):
        raise ValueError("question must not contain reserved chat or image tokens")
    marker = model.config.image_special_token * model.config.image_token_len
    conversation = [{"role": "user", "content": marker + "\n" + question.strip()}]
    prompt_ids = torch.tensor(
        [tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)],
        device=model.device,
    )
    if prompt_ids.size(1) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus requested output exceeds model context")
    pixels = {
        key: value.to(model.device)
        for key, value in processor(images=image.convert("RGB"), return_tensors="pt").items()
    }
    model.eval()
    answer_ids, cache, current_ids = [], None, prompt_ids
    for _ in range(max_new_tokens):
        output = model(
            input_ids=current_ids,
            pixel_values=pixels if cache is None else None,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        next_id = output.logits[:, -1].argmax(-1, keepdim=True)
        answer_ids.append(next_id.item())
        if next_id.item() == tokenizer.eos_token_id:
            break
        current_ids, cache = next_id, output.past_key_values
    return tokenizer.decode(answer_ids, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask an exported model about an image")
    parser.add_argument("--model", required=True, help="Path to a trusted local hf/ export")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        options.model, trust_remote_code=True, local_files_only=True
    ).to(options.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    processor = SiglipImageProcessor.from_pretrained(options.model, local_files_only=True)
    with Image.open(options.image) as image:
        print(predict(image, options.question, options.max_new_tokens, model, tokenizer, processor))
