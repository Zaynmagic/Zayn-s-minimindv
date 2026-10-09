# Adapted from jingyaogong/minimind under Apache-2.0; see NOTICE.
# Modified: validated configuration, explicit masked attention, and embedding inputs.

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        if (
            not isinstance(self.num_attention_heads, int)
            or isinstance(self.num_attention_heads, bool)
            or self.num_attention_heads < 1
        ):
            raise ValueError("num_attention_heads must be a positive integer")
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        # Expert settings only affect mixture-of-experts models.
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get(
            "moe_intermediate_size", self.intermediate_size
        )
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        self.validate_dimensions()

    def validate_dimensions(self):
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
            "head_dim",
            "intermediate_size",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        kv_heads = self.num_key_value_heads or self.num_attention_heads
        if (
            not isinstance(kv_heads, int)
            or isinstance(kv_heads, bool)
            or kv_heads < 1
            or self.num_key_value_heads == 0
            or self.num_attention_heads % kv_heads
        ):
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta and rms_norm_eps must be positive")
        if self.hidden_act not in ACT2FN:
            raise ValueError(f"unsupported activation: {self.hidden_act}")
        if self.use_moe:
            for name in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
                value = getattr(self, name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            if self.num_experts_per_tok > self.num_experts:
                raise ValueError("num_experts_per_tok cannot exceed num_experts")
            if self.router_aux_loss_coef < 0:
                raise ValueError("router_aux_loss_coef cannot be negative")


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        mean_square = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    dimensions = torch.arange(0, dim, 2)[: dim // 2].float()
    inverse_frequencies = 1.0 / (rope_base ** (dimensions / dim))
    attention_factor = 1.0
    if rope_scaling is not None:
        original_context = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attention_factor = rope_scaling.get("attention_factor", 1.0)
        if end / original_context > 1.0:

            def correction_dimension(rotations):
                wavelength = original_context / (rotations * 2 * math.pi)
                return dim * math.log(wavelength) / (2 * math.log(rope_base))

            lower = max(math.floor(correction_dimension(beta_fast)), 0)
            upper = min(math.ceil(correction_dimension(beta_slow)), dim // 2 - 1)
            positions = torch.arange(
                dim // 2, device=inverse_frequencies.device
            ).float()
            ramp = ((positions - lower) / max(upper - lower, 0.001)).clamp(0, 1)
            inverse_frequencies = inverse_frequencies * (1 - ramp + ramp / factor)
    positions = torch.arange(end, device=inverse_frequencies.device)
    angles = torch.outer(positions, inverse_frequencies).float()
    freqs_cos = (
        torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attention_factor
    )
    freqs_sin = (
        torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attention_factor
    )
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(tensor):
        midpoint = tensor.shape[-1] // 2
        return torch.cat((-tensor[..., midpoint:], tensor[..., :midpoint]), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, sequence_length, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    repeated = x[:, :, :, None, :].expand(
        batch_size, sequence_length, kv_heads, n_rep, head_dim
    )
    return repeated.reshape(batch_size, sequence_length, kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        batch_size, query_length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).view(
            batch_size, query_length, self.n_local_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).view(
            batch_size, query_length, self.n_local_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).view(
            batch_size, query_length, self.n_local_kv_heads, self.head_dim
        )
        queries, keys = self.q_norm(queries), self.k_norm(keys)
        cos, sin = position_embeddings
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        if past_key_value is not None:
            keys = torch.cat([past_key_value[0], keys], dim=1)
            values = torch.cat([past_key_value[1], values], dim=1)
        present = (keys, values) if use_cache else None
        queries = queries.transpose(1, 2)
        keys = repeat_kv(keys, self.n_rep).transpose(1, 2)
        values = repeat_kv(values, self.n_rep).transpose(1, 2)
        key_length = keys.size(-2)

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, key_length):
                raise ValueError(
                    "attention_mask must cover all cached and current tokens"
                )
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask must contain only zero or one")
        unpadded = attention_mask is None or torch.all(attention_mask == 1)
        if self.flash and unpadded and (past_key_value is None or query_length == 1):
            # Preserve the fused causal path without allocating a square mask.
            output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal and past_key_value is None,
            )
        else:
            # Cached queries occupy the final positions in the full key sequence.
            query_positions = torch.arange(
                key_length - query_length, key_length, device=queries.device
            )
            key_positions = torch.arange(key_length, device=queries.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            if not self.is_causal:
                allowed = torch.ones_like(allowed)
            allowed = allowed[None, None, :, :]
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].bool()
            if self.flash:
                output = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    attn_mask=allowed,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            else:
                scores = (queries @ keys.transpose(-2, -1)).float() / math.sqrt(
                    self.head_dim
                )
                scores = scores.masked_fill(~allowed, float("-inf"))
                # A left-padding query can have no visible keys. Its context is zero.
                scores = torch.where(
                    allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores)
                )
                weights = (
                    F.softmax(scores, dim=-1).masked_fill(~allowed, 0).type_as(queries)
                )
                output = self.attn_dropout(weights) @ values
        output = output.transpose(1, 2).reshape(batch_size, query_length, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                FeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, sequence_length, hidden_dim = x.shape
        token_states = x.view(-1, hidden_dim)
        router_scores = F.softmax(self.gate(token_states), dim=-1)
        routing_weights, expert_indices = torch.topk(
            router_scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                selected_mass = routing_weights.sum(dim=-1, keepdim=True)
                routing_weights = routing_weights / (selected_mass + 1e-20)
            else:
                detached_scores = F.softmax(self.gate(token_states.detach()), dim=-1)
                top_weight = torch.topk(detached_scores, k=1, dim=-1, sorted=False)[0]
                # Keep the top-1 output unscaled while still training its router weight.
                routing_weights = top_weight - top_weight.detach() + 1.0
        combined = torch.zeros_like(token_states)
        for expert_index, expert in enumerate(self.experts):
            selected = expert_indices == expert_index
            if selected.any():
                token_indices = selected.any(dim=-1).nonzero().flatten()
                weights = routing_weights[selected].view(-1, 1)
                expert_output = expert(token_states[token_indices]) * weights
                combined.index_add_(0, token_indices, expert_output.to(combined.dtype))
            elif self.training:
                # DDP still needs gradient hooks for experts receiving no tokens.
                combined[0, 0] += 0 * sum(
                    parameter.sum() for parameter in expert.parameters()
                )
        if self.training and self.config.router_aux_loss_coef > 0:
            expert_load = (
                F.one_hot(expert_indices, self.config.num_experts).float().mean(0)
            )
            mean_scores = router_scores.mean(0)
            self.aux_loss = (
                (expert_load * mean_scores).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss = router_scores.new_zeros(1).squeeze()
        return combined.view(batch_size, sequence_length, hidden_dim)


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                MiniMindBlock(layer_index, config)
                for layer_index in range(self.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        inputs_embeds=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            inputs_embeds = self.embed_tokens(input_ids)
        elif (
            inputs_embeds.ndim != 3
            or inputs_embeds.shape[-1] != self.config.hidden_size
        ):
            raise ValueError(
                "inputs_embeds must have shape [batch, sequence, hidden_size]"
            )
        batch_size, seq_length = inputs_embeds.shape[:2]
        if batch_size == 0 or seq_length == 0:
            raise ValueError("input batches and sequences cannot be empty")
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one entry per layer")
        cached_length = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        end_position = cached_length + seq_length
        if end_position > self.config.max_position_embeddings:
            raise ValueError("sequence and cache exceed max_position_embeddings")
        hidden_states = self.dropout(inputs_embeds)
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.freqs_cos[cached_length:end_position],
            self.freqs_sin[cached_length:end_position],
        )
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        expert_losses = [
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        ]
        aux_loss = sum(expert_losses, hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        inputs_embeds=None,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            next_token_logits = logits[..., :-1, :].contiguous()
            next_token_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                next_token_logits.view(-1, next_token_logits.size(-1)),
                next_token_labels.view(-1),
                ignore_index=-100,
            )
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            cached_length = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, cached_length:],
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            if attention_mask is not None:
                next_mask = attention_mask.new_ones(attention_mask.shape[0], 1)
                attention_mask = torch.cat([attention_mask, next_mask], -1)
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for row_index in range(input_ids.shape[0]):
                    seen_tokens = torch.unique(input_ids[row_index])
                    seen_logits = logits[row_index, seen_tokens]
                    logits[row_index, seen_tokens] = torch.where(
                        seen_logits > 0,
                        seen_logits / repetition_penalty,
                        seen_logits * repetition_penalty,
                    )
            if top_k > 0:
                cutoff = torch.topk(logits, top_k)[0][..., -1, None]
                logits[logits < cutoff] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_mass = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                excluded = cumulative_mass > top_p
                excluded[..., 1:] = excluded[..., :-1].clone()
                excluded[..., 0] = 0
                logits[excluded.scatter(1, sorted_indices, excluded)] = -float("inf")
            if do_sample:
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                eos_tokens = next_token.new_full((next_token.shape[0], 1), eos_token_id)
                next_token = torch.where(finished.unsqueeze(-1), eos_tokens, next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids
