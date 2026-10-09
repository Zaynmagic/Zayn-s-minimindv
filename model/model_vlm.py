"""Single-image language model with a frozen, bundled SigLIP vision tower.

Derived from MiniMind-V (Apache-2.0); modified in 2026 for validated image
spans, vectorized feature insertion, and self-contained model exports.
"""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .model_minimind import MiniMindConfig, MiniMindForCausalLM


class VisionLanguageConfig(MiniMindConfig):
    model_type = "zayn_minimind_v"

    def __init__(
        self,
        vision_config=None,
        processor_config=None,
        projector_hidden_size=2048,
        vision_source="random-test-fixture",
        vision_revision=None,
        image_special_token="<|image_pad|>",
        image_ids=None,
        **kwargs,
    ):
        if not vision_config:
            vision_config = {
                "image_size": 256,
                "patch_size": 32,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
            }
        vision = SiglipVisionConfig(**vision_config)
        valid_patch_size = vision.image_size > 0 and vision.patch_size > 0
        if not valid_patch_size or vision.image_size % vision.patch_size:
            raise ValueError(
                "vision image size must be a positive multiple of patch size"
            )
        patch_count = (vision.image_size // vision.patch_size) ** 2
        if kwargs.pop("image_token_len", patch_count) != patch_count:
            raise ValueError("image_token_len must match the vision patch count")
        if kwargs.pop("image_hidden_size", vision.hidden_size) != vision.hidden_size:
            raise ValueError("image_hidden_size must match the vision hidden size")
        if not isinstance(projector_hidden_size, int) or projector_hidden_size < 1:
            raise ValueError("projector_hidden_size must be a positive integer")
        super().__init__(**kwargs)
        self.image_ids = list(image_ids) if image_ids is not None else [12]
        if len(self.image_ids) != 1:
            raise ValueError("image_ids must contain one valid vocabulary ID")
        image_token_id = self.image_ids[0]
        if (
            not isinstance(image_token_id, int)
            or not 0 <= image_token_id < self.vocab_size
        ):
            raise ValueError("image_ids must contain one valid vocabulary ID")
        if not isinstance(image_special_token, str) or not image_special_token:
            raise ValueError("image_special_token must be a nonempty string")
        self.image_special_token = image_special_token
        self.image_hidden_size = vision.hidden_size
        self.image_token_len = patch_count
        self.vision_config = vision.to_dict()
        self.processor_config = processor_config or {
            "size": {"height": vision.image_size, "width": vision.image_size},
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        self.projector_hidden_size = projector_hidden_size
        self.vision_source = vision_source
        self.vision_revision = vision_revision


class VisionLanguageModel(MiniMindForCausalLM):
    config_class = VisionLanguageConfig

    def __init__(self, config=None):
        super().__init__(config or VisionLanguageConfig())
        self.vision_encoder = SiglipVisionModel(
            SiglipVisionConfig(**self.config.vision_config)
        )
        self.processor = SiglipImageProcessor(**self.config.processor_config)
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.config.image_hidden_size),
            nn.Linear(self.config.image_hidden_size, self.config.projector_hidden_size),
            nn.GELU(),
            nn.Linear(self.config.projector_hidden_size, self.config.hidden_size),
        )
        # Freezing the tower keeps its feature space stable across alignment and SFT.
        self.vision_encoder.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vision_encoder.eval()
        return self

    def _merge_image_features(self, token_ids, embeddings, pixel_values):
        """One contiguous span keeps image patch order aligned with token order."""
        marker_mask = token_ids.eq(self.config.image_ids[0])
        span_length = self.config.image_token_len
        if not torch.all(marker_mask.sum(dim=1) == span_length):
            raise ValueError("one image requires exactly image_token_len image tokens")
        span_start = marker_mask.long().argmax(dim=1)
        token_positions = torch.arange(token_ids.size(1), device=token_ids.device)
        patch_offsets = token_positions[None, :] - span_start[:, None]
        expected_span = (patch_offsets >= 0) & (patch_offsets < span_length)
        if not torch.equal(marker_mask, expected_span):
            raise ValueError("image tokens must form one contiguous span")
        if isinstance(pixel_values, Mapping):
            if set(pixel_values) != {"pixel_values"}:
                raise ValueError("image inputs must contain only pixel_values")
            pixels = pixel_values["pixel_values"]
        else:
            pixels = pixel_values
        vision_size = self.config.vision_config["image_size"]
        expected_shape = (token_ids.size(0), 3, vision_size, vision_size)
        if (
            not isinstance(pixels, torch.Tensor)
            or tuple(pixels.shape) != expected_shape
        ):
            raise ValueError(
                f"pixel_values must have shape {expected_shape}; one RGB image per row"
            )
        with torch.no_grad():
            image_features = self.vision_encoder(pixel_values=pixels).last_hidden_state
        projected = self.vision_proj(image_features).to(embeddings.dtype)
        if projected.shape[:2] != (token_ids.size(0), span_length):
            raise ValueError("vision output does not match configured image span")
        # Non-image positions need valid gather indices; the mask discards them below.
        patch_indices = patch_offsets.clamp(0, span_length - 1)
        gather_indices = patch_indices.unsqueeze(-1).expand(-1, -1, projected.size(-1))
        image_embeddings = projected.gather(1, gather_indices)
        return torch.where(marker_mask.unsqueeze(-1), image_embeddings, embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        if input_ids is None or input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must be a nonempty batch of token sequences")
        has_cached_prefix = (
            past_key_values is not None
            and not hasattr(past_key_values, "layers")
            and len(past_key_values) > 0
            and past_key_values[0] is not None
        )
        embeddings = self.model.embed_tokens(input_ids)
        if not has_cached_prefix:
            if pixel_values is not None:
                embeddings = self._merge_image_features(
                    input_ids, embeddings, pixel_values
                )
            elif input_ids.eq(self.config.image_ids[0]).any():
                raise ValueError(
                    "image tokens require pixel_values on the initial forward pass"
                )
        hidden_states, cache, auxiliary_loss = self.model(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        if labels is not None and (
            not isinstance(logits_to_keep, int) or logits_to_keep != 0
        ):
            raise ValueError("training labels require the full sequence of logits")
        if isinstance(logits_to_keep, int):
            token_selection = slice(-logits_to_keep, None)
        else:
            token_selection = logits_to_keep
        logits = self.lm_head(hidden_states[:, token_selection, :])
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            targets = labels.clone().masked_fill(
                input_ids.eq(self.config.image_ids[0]), -100
            )
            if attention_mask is not None:
                targets.masked_fill_(attention_mask[:, -input_ids.size(1) :] == 0, -100)
            if not (targets[:, 1:] != -100).any():
                raise ValueError("at least one language target is required")
            next_token_logits = logits[:, :-1].contiguous()
            next_token_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                next_token_logits.view(-1, logits.size(-1)),
                next_token_targets.view(-1),
                ignore_index=-100,
            )
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=auxiliary_loss,
            logits=logits,
            past_key_values=cache,
            hidden_states=hidden_states,
        )

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and kwargs.get("pixel_values") is not None:
            pixels = kwargs["pixel_values"]
            if isinstance(pixels, Mapping):
                kwargs["pixel_values"] = {
                    key: value.repeat(num_return_sequences, *([1] * (value.ndim - 1)))
                    for key, value in pixels.items()
                }
            else:
                kwargs["pixel_values"] = pixels.repeat(
                    num_return_sequences, *([1] * (pixels.ndim - 1))
                )
        return super().generate(
            *args, num_return_sequences=num_return_sequences, **kwargs
        )

    @classmethod
    def with_pretrained_vision(cls, text_config, source, revision=None):
        vision = SiglipVisionModel.from_pretrained(source, revision=revision)
        processor = SiglipImageProcessor.from_pretrained(source, revision=revision)
        settings = dict(text_config)
        for key in (
            "vision_config",
            "processor_config",
            "vision_source",
            "vision_revision",
        ):
            settings.pop(key, None)
        config = VisionLanguageConfig(
            **settings,
            vision_config=vision.config.to_dict(),
            processor_config=processor.to_dict(),
            vision_source=source,
            vision_revision=revision,
        )
        model = cls(config)
        model.vision_encoder = vision.eval().requires_grad_(False)
        model.processor = processor
        return model
