# Check image inputs, frozen weights, cached inference, and self-contained exports.

from pathlib import Path
import json
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model_vlm import VisionLanguageModel, VisionLanguageConfig
from training.run import configure_stage, export
from training.data import batch
from training.fixtures import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(42)
    torch.set_num_threads(2)


@pytest.fixture
def active():
    return VisionLanguageModel(
        VisionLanguageConfig(**json.loads((ROOT / "configs/tiny.json").read_text()))
    )


@pytest.fixture
def inputs(active, tmp_path):
    generate(tmp_path)
    row = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    return batch([row], tokenizer, active.processor, active.config, 128, tmp_path)


def test_encoder_stays_frozen_and_in_eval(active):
    active.train()
    assert active.training and not active.vision_encoder.training
    assert all(not p.requires_grad for p in active.vision_encoder.parameters())


def test_alignment_only_updates_projector(active, inputs):
    configure_stage(active, "align")
    before = {name: p.clone() for name, p in active.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in active.parameters() if p.requires_grad], lr=0.01)
    output = active(**inputs)
    (output.loss + output.aux_loss).backward()
    optimizer.step()
    assert any(
        not torch.equal(p, before[name])
        for name, p in active.named_parameters()
        if name.startswith("vision_proj.")
    )
    assert all(
        torch.equal(p, before[name])
        for name, p in active.named_parameters()
        if not name.startswith("vision_proj.")
    )


def test_instruction_tuning_updates_llm_and_projector(active, inputs):
    configure_stage(active, "sft")
    active.train()
    output = active(**inputs)
    (output.loss + output.aux_loss).backward()
    assert active.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert active.vision_proj[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in active.vision_encoder.parameters())


def test_images_change_logits(active, inputs):
    active.eval()
    altered = {**inputs, "pixel_values": {"pixel_values": -inputs["pixel_values"]["pixel_values"]}}
    with torch.no_grad():
        first, second = active(**inputs).logits, active(**altered).logits
    assert (first - second).abs().max() > 1e-5


def test_invalid_image_span_rejected(active, inputs):
    inputs["input_ids"][inputs["input_ids"] == 12] = 41
    with pytest.raises(ValueError, match="image_token_len"):
        active(**inputs)


def test_cached_decode_matches_full_multimodal_forward(active, inputs):
    active.eval()
    prefix = {key: value for key, value in inputs.items() if key != "labels"}
    with torch.no_grad():
        output = active(**prefix, use_cache=True)
        following = torch.tensor([[41]])
        mask = torch.cat((inputs["attention_mask"], torch.ones(1, 1, dtype=torch.long)), 1)
        cached = active(
            input_ids=following,
            attention_mask=mask,
            past_key_values=output.past_key_values,
            use_cache=True,
        )
        full = active(
            input_ids=torch.cat((inputs["input_ids"], following), 1),
            attention_mask=mask,
            pixel_values=inputs["pixel_values"],
        )
    torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-5)


def test_self_contained_hf_export_roundtrip(active, inputs, tmp_path):
    active.eval()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    export(active, tokenizer, tmp_path / "hf")
    restored = AutoModelForCausalLM.from_pretrained(
        tmp_path / "hf", trust_remote_code=True, local_files_only=True
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(active(**inputs).logits, restored(**inputs).logits, rtol=1e-4, atol=1e-5)
    assert restored.vision_encoder is not None
    assert (tmp_path / "hf" / "preprocessor_config.json").is_file()


def test_image_padding_never_contributes_to_language_loss(active, inputs):
    assert torch.all(inputs["labels"][inputs["input_ids"] == 12] == -100)
    assert torch.all(inputs["labels"][inputs["attention_mask"] == 0] == -100)


def test_truncated_image_span_rejected_in_preprocessing(active, tmp_path):
    generate(tmp_path)
    row = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    with pytest.raises(ValueError, match="truncated image"):
        batch([row], tokenizer, active.processor, active.config, 4, tmp_path)


def test_tensor_and_processor_dictionary_inputs_match(active, inputs):
    active.eval()
    with torch.no_grad():
        wrapped = active(**inputs)
        direct = active(**{**inputs, "pixel_values": inputs["pixel_values"]["pixel_values"]})
    torch.testing.assert_close(wrapped.logits, direct.logits)


def test_noncontiguous_image_span_is_rejected(active, inputs):
    token_ids = inputs["input_ids"].clone()
    positions = token_ids[0].eq(active.config.image_ids[0]).nonzero().flatten()
    displaced = positions[-1] + 2
    token_ids[0, positions[-1]], token_ids[0, displaced] = token_ids[0, displaced].clone(), token_ids[0, positions[-1]].clone()
    with pytest.raises(ValueError, match="contiguous"):
        active(**{**inputs, "input_ids": token_ids})


def test_image_tokens_require_pixels(active, inputs):
    without_image = {key: value for key, value in inputs.items() if key != "pixel_values"}
    with pytest.raises(ValueError, match="require pixel_values"):
        active(**without_image)


@pytest.mark.parametrize("shape", [(2, 3, 32, 32), (1, 1, 3, 32, 32), (1, 3, 16, 16)])
def test_invalid_image_tensor_shape_is_rejected(active, inputs, shape):
    with pytest.raises(ValueError, match="one RGB image"):
        active(**{**inputs, "pixel_values": torch.zeros(shape)})


def test_model_masks_image_and_padding_targets(active, inputs):
    active.eval()
    targets = inputs["input_ids"].clone()
    masked_targets = targets.masked_fill(
        targets.eq(active.config.image_ids[0]) | inputs["attention_mask"].eq(0), -100
    )
    with torch.no_grad():
        raw = active(**{**inputs, "labels": targets})
        masked = active(**{**inputs, "labels": masked_targets})
    torch.testing.assert_close(raw.loss, masked.loss)
    assert torch.equal(targets, inputs["input_ids"])


def test_all_ignored_labels_are_rejected(active, inputs):
    with pytest.raises(ValueError, match="language target"):
        active(**{**inputs, "labels": torch.full_like(inputs["labels"], -100)})


def test_cached_generation_only_encodes_image_once(active, inputs):
    calls = []
    handle = active.vision_encoder.register_forward_hook(lambda *args: calls.append(1))
    try:
        generated = active.generate(
            input_ids=inputs["input_ids"][:, :32],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=3,
            eos_token_id=None,
            do_sample=False,
        )
    finally:
        handle.remove()
    assert len(calls) == 1
    assert generated.shape == (1, 35)


def test_multiple_generated_sequences_repeat_corresponding_images(active, inputs):
    generated = active.generate(
        input_ids=inputs["input_ids"][:, :32],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1,
        num_return_sequences=2,
        do_sample=False,
    )
    assert generated.shape == (2, 33)
    torch.testing.assert_close(generated[0], generated[1])


@pytest.mark.parametrize("field,value", [("image_token_len", 99), ("image_hidden_size", 99), ("image_ids", []), ("projector_hidden_size", 0)])
def test_mismatched_vision_config_rejected(field, value):
    settings = json.loads((ROOT / "configs/tiny.json").read_text())
    settings[field] = value
    with pytest.raises(ValueError):
        VisionLanguageConfig(**settings)


def test_duplicate_image_placeholder_is_rejected(active, tmp_path):
    generate(tmp_path)
    row = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    row["conversations"][1]["content"] += " <image>"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    with pytest.raises(ValueError, match="exactly one"):
        batch([row], tokenizer, active.processor, active.config, 128, tmp_path)


def test_local_prediction(active):
    from PIL import Image
    from inference import predict

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    response = predict(Image.new("RGB", (32, 32), "red"), "What color is this?", 2, active, tokenizer, active.processor)
    assert isinstance(response, str)
    with pytest.raises(ValueError, match="positive integer"):
        predict(Image.new("RGB", (32, 32)), "Describe this.", 0, active, tokenizer, active.processor)
