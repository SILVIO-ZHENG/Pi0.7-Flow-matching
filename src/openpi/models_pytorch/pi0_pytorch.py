"""Implement the PyTorch Pi0 vision-language-action model."""

import logging
import math
import time

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar or token positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim not in (1, 2):
        raise ValueError("The time tensor is expected to have shape `(batch_size,)` or `(batch_size, horizon)`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Flatten first and restore later to support batch-level and action-token-level timesteps.
    original_shape = time.shape
    flat_time = time.reshape(-1).to(dtype=dtype)
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * flat_time[:, None]
    emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return emb.reshape(*original_shape, dimension)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def _sync_if_cuda(device):
    """Synchronize the CUDA queue so asynchronous execution does not under-report timing."""
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    """PyTorch Pi0 implementation for multimodal Flow Matching inference."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self._last_sample_timing = {}

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def _sample_rtc_prefix_steps(self, bsize, device):
        """Sample a training-time RTC prefix length from the configuration."""
        rtc_config = getattr(self.config, "rtc_training", None)
        if rtc_config is None or not rtc_config.enabled:
            return None

        execution_horizon = rtc_config.execution_horizon
        if execution_horizon is None:
            execution_horizon = self.config.action_horizon // 2

        legal_max_prefix = self.config.action_horizon - execution_horizon
        if legal_max_prefix < 0:
            raise ValueError(
                "RTC training requires execution_horizon <= action_horizon, "
                f"got execution_horizon={execution_horizon}, action_horizon={self.config.action_horizon}."
            )

        configured_max = rtc_config.max_prefix_steps
        if configured_max is None:
            configured_max = legal_max_prefix
        max_prefix_steps = min(configured_max, legal_max_prefix)
        min_prefix_steps = min(rtc_config.min_prefix_steps, max_prefix_steps)

        if max_prefix_steps == min_prefix_steps:
            prefix_steps = torch.full((bsize,), max_prefix_steps, dtype=torch.long, device=device)
        else:
            prefix_steps = torch.randint(min_prefix_steps, max_prefix_steps + 1, (bsize,), device=device)

        if rtc_config.prefix_probability < 1.0:
            use_prefix = torch.rand((bsize,), device=device) < rtc_config.prefix_probability
            prefix_steps = torch.where(use_prefix, prefix_steps, torch.zeros_like(prefix_steps))

        return prefix_steps

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def vlm_parameters(self):
        """Return the PaliGemma parameters protected by Knowledge Insulation."""

        return self.paligemma_with_expert.paligemma.parameters()

    def compute_fast_token_loss(self, observation) -> Tensor:
        """Compute hierarchical subtask + FAST action-token teacher-forcing CE.

        The method uses the dedicated ``fast_*`` fields and therefore cannot
        leak expert action tokens into the continuous flow branch's prefix.
        """

        processed = _preprocessing.preprocess_observation_pytorch(observation, train=True)
        tokens = processed.fast_tokenized_prompt
        token_mask = processed.fast_tokenized_prompt_mask
        token_ar_mask = processed.fast_token_ar_mask
        loss_mask = processed.fast_token_loss_mask
        if any(value is None for value in (tokens, token_mask, token_ar_mask, loss_mask)):
            raise ValueError("joint FAST objective requires all fast_token* observation fields")

        embeddings = []
        padding = []
        autoregressive = []
        for image, image_mask in zip(processed.images.values(), processed.image_masks.values(), strict=True):
            image_embedding = self._apply_checkpoint(self.paligemma_with_expert.embed_image, image)
            batch_size, image_tokens = image_embedding.shape[:2]
            embeddings.append(image_embedding)
            padding.append(image_mask[:, None].expand(batch_size, image_tokens))
            autoregressive.append(torch.zeros(batch_size, image_tokens, dtype=torch.bool, device=image.device))

        language_embedding = self.paligemma_with_expert.embed_language_tokens(tokens)
        language_embedding = language_embedding * math.sqrt(language_embedding.shape[-1])
        embeddings.append(language_embedding)
        padding.append(token_mask.to(dtype=torch.bool))
        autoregressive.append(token_ar_mask.to(dtype=torch.bool))

        full_embedding = torch.cat(embeddings, dim=1)
        full_padding = torch.cat(padding, dim=1)
        full_ar = torch.cat(autoregressive, dim=1)
        attention = self._prepare_attention_masks_4d(make_att_2d_masks(full_padding, full_ar))
        position_ids = torch.cumsum(full_padding, dim=1) - 1
        (hidden, _), _ = self.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[full_embedding, None],
            use_cache=False,
            adarms_cond=[None, None],
        )

        image_token_count = full_embedding.shape[1] - tokens.shape[1]
        predictor_hidden = hidden[:, image_token_count : image_token_count + tokens.shape[1] - 1]
        target_tokens = tokens[:, 1:].to(dtype=torch.long)
        target_mask = loss_mask[:, 1:].to(dtype=torch.bool) & token_mask[:, 1:].to(dtype=torch.bool)
        if not torch.any(target_mask):
            # A batch can consist entirely of episode-tail samples.  Returning
            # a graph-connected zero keeps DDP/backward valid without learning
            # the repeat-last padding encoded by FAST.
            return hidden.sum() * 0.0

        selected_hidden = predictor_hidden[target_mask]
        lm_head = self.paligemma_with_expert.paligemma.lm_head
        selected_hidden = selected_hidden.to(dtype=lm_head.weight.dtype)
        logits = lm_head(selected_hidden).to(dtype=torch.float32)
        return F.cross_entropy(logits, target_tokens[target_mask], reduction="mean")

    @torch.no_grad()
    def generate_fast_tokens(
        self,
        device,
        observation,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Tensor:
        """Autoregressively generate the hierarchical FAST suffix for batch=1.

        This deliberately recomputes the short planner sequence instead of
        sharing the flow KV cache; it keeps the research path simple and makes
        the planner/Action-Expert boundary explicit.
        """

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        tokens = processed.fast_tokenized_prompt
        token_mask = processed.fast_tokenized_prompt_mask
        if tokens is None or token_mask is None:
            raise ValueError("hierarchical generation requires fast tokenized inputs")
        if tokens.shape[0] != 1:
            raise ValueError("hierarchical generation currently supports batch size 1")
        valid_length = int(token_mask[0].sum().item())
        if valid_length <= 0:
            raise ValueError("hierarchical generation requires a non-empty FAST prompt")
        remaining_budget = self.config.fast_max_token_len - valid_length
        if remaining_budget <= 0:
            raise ValueError(
                "FAST prompt already fills fast_max_token_len; increase the configured token budget before generation"
            )
        max_new_tokens = min(max_new_tokens, remaining_budget)
        sequence = tokens[:, :valid_length].to(device=device, dtype=torch.long)

        image_embeddings = []
        image_masks = []
        for image, image_mask in zip(processed.images.values(), processed.image_masks.values(), strict=True):
            embedding = self.paligemma_with_expert.embed_image(image)
            image_embeddings.append(embedding)
            image_masks.append(image_mask[:, None].expand(1, embedding.shape[1]))
        image_embedding = torch.cat(image_embeddings, dim=1)
        image_padding = torch.cat(image_masks, dim=1).to(dtype=torch.bool)
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        for _ in range(max_new_tokens):
            language = self.paligemma_with_expert.embed_language_tokens(sequence)
            language = language * math.sqrt(language.shape[-1])
            full_embedding = torch.cat([image_embedding, language], dim=1)
            language_padding = torch.ones_like(sequence, dtype=torch.bool)
            full_padding = torch.cat([image_padding, language_padding], dim=1)
            # Original prompt is one bidirectional block; generated tokens are causal.
            language_ar = torch.zeros_like(sequence, dtype=torch.bool)
            language_ar[:, valid_length:] = True
            full_ar = torch.cat([torch.zeros_like(image_padding), language_ar], dim=1)
            attention = self._prepare_attention_masks_4d(make_att_2d_masks(full_padding, full_ar))
            position_ids = torch.cumsum(full_padding, dim=1) - 1
            (hidden, _), _ = self.paligemma_with_expert.forward(
                attention_mask=attention,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[full_embedding, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
            last_hidden = hidden[:, -1].to(dtype=lm_head.weight.dtype)
            logits = lm_head(last_hidden).to(dtype=torch.float32)
            if temperature > 0:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
        return sequence

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions=None, noise=None, time=None, objective="flow") -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if objective == "fast_ce":
            return self.compute_fast_token_loss(observation)
        if objective != "flow":
            raise ValueError(f"Unknown objective: {objective}")
        if actions is None:
            raise ValueError("actions are required for the flow objective")
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B,{self.config.action_horizon},{self.config.action_dim}]")
        expected_shape = (actions.shape[0], self.config.action_horizon, self.config.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"actions must have shape [B,{self.config.action_horizon},{self.config.action_dim}]")
        if not torch.isfinite(actions).all():
            raise ValueError("actions contain NaN or Inf")
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        # LeRobot normalization may produce float64; cast to float32 before the PyTorch model.
        actions = actions.to(dtype=torch.float32)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        else:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=actions.device)
            if tuple(noise.shape) != tuple(actions.shape) or not torch.isfinite(noise).all():
                raise ValueError("noise must be finite and have the same shape as actions")

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        else:
            time = torch.as_tensor(time, dtype=torch.float32, device=actions.device)
            if time.numel() != actions.shape[0]:
                raise ValueError(f"time must contain exactly {actions.shape[0]} values")
            time = time.reshape(actions.shape[0])
            if not torch.isfinite(time).all() or torch.any(time < 0) or torch.any(time > 1):
                raise ValueError("time must contain finite values in [0, 1]")

        time_expanded = time[:, None, None]
        # xt=tao*noise + (1-tao)*At
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # ut=noise-At
        u_t = noise - actions
        loss_scale = None
        model_time = time

        prefix_steps = self._sample_rtc_prefix_steps(actions.shape[0], actions.device)
        if prefix_steps is not None:
            step_ids = torch.arange(self.config.action_horizon, device=actions.device)[None, :]
            prefix_mask = step_ids < prefix_steps[:, None]
            postfix_mask = ~prefix_mask
            postfix_time = time[:, None].expand(-1, self.config.action_horizon)
            # The OpenPI flow convention uses time=0 for clean actions and time=1 for noise.
            model_time = torch.where(prefix_mask, torch.zeros_like(postfix_time), postfix_time)
            x_t = model_time[:, :, None] * noise + (1 - model_time[:, :, None]) * actions

            if observation.action_step_mask is None:
                valid_step_mask = torch.ones_like(postfix_mask)
            else:
                valid_step_mask = observation.action_step_mask.to(device=x_t.device, dtype=torch.bool)
                if tuple(valid_step_mask.shape) != tuple(postfix_mask.shape):
                    raise ValueError(
                        f"action_step_mask must have shape {tuple(postfix_mask.shape)}, got {tuple(valid_step_mask.shape)}"
                    )
                # Episode-tail batches may contain fewer valid future steps than
                # the sampled RTC prefix. Keep at least one valid postfix step
                # whenever a sample has any supervision at all.
                max_valid_prefix = (valid_step_mask.sum(dim=1) - 1).clamp_min(0)
                prefix_steps = torch.minimum(prefix_steps, max_valid_prefix)
                prefix_mask = step_ids < prefix_steps[:, None]
                postfix_mask = ~prefix_mask
                model_time = torch.where(prefix_mask, torch.zeros_like(postfix_time), postfix_time)
                x_t = model_time[:, :, None] * noise + (1 - model_time[:, :, None]) * actions
            valid_steps = valid_step_mask.sum(dim=1).clamp_min(1).to(dtype=x_t.dtype)
            valid_postfix_steps = (postfix_mask & valid_step_mask).sum(dim=1).clamp_min(1).to(dtype=x_t.dtype)
            loss_scale = postfix_mask[:, :, None].to(dtype=x_t.dtype)
            loss_scale = loss_scale * (valid_steps / valid_postfix_steps)[:, None, None]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, model_time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        if loss_scale is not None:
            losses = losses * loss_scale
        if observation.action_step_mask is not None:
            losses = losses * observation.action_step_mask[:, :, None].to(device=losses.device, dtype=losses.dtype)
        if observation.action_dim_mask is not None:
            losses = losses * observation.action_dim_mask[:, None, :].to(device=losses.device, dtype=losses.dtype)
        return losses

    def sample_actions(
        self, device, observation, noise=None, num_steps=10, rtc_guidance=None, rtc_prefix=None
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        if rtc_guidance is None:
            with torch.no_grad():
                return self._sample_actions_impl(
                    device, observation, noise=noise, num_steps=num_steps, rtc_prefix=rtc_prefix
                )
        with torch.enable_grad():
            return self._sample_actions_impl(
                device,
                observation,
                noise=noise,
                num_steps=num_steps,
                rtc_guidance=rtc_guidance,
                rtc_prefix=rtc_prefix,
            )

    def _sample_actions_impl(
        self, device, observation, noise=None, num_steps=10, rtc_guidance=None, rtc_prefix=None
    ) -> Tensor:
        """Run Flow Matching sampling with optional RTC guidance or a training-time hard prefix."""
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        total_start = time.perf_counter()
        timing = {}
        bsize = observation.state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        if noise is None:
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=device)
            if tuple(noise.shape) != actions_shape:
                raise ValueError(f"noise must have shape {actions_shape}, got {tuple(noise.shape)}")
            if not torch.isfinite(noise).all():
                raise ValueError("noise contains NaN or Inf")

        preprocess_start = time.perf_counter()
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        timing["model_preprocess_ms"] = (time.perf_counter() - preprocess_start) * 1000

        _sync_if_cuda(device)
        vlm_start = time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        _sync_if_cuda(device)
        timing["vlm_prefix_forward_ms"] = (time.perf_counter() - vlm_start) * 1000

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        rtc_prefix_actions, rtc_prefix_mask = self._prepare_rtc_prefix(x_t, rtc_prefix)
        flow_time = torch.tensor(1.0, dtype=torch.float32, device=device)
        _sync_if_cuda(device)
        flow_start = time.perf_counter()
        denoise_steps = 0
        while flow_time >= -dt / 2:
            if rtc_prefix_mask is not None:
                x_t = torch.where(rtc_prefix_mask[:, :, None], rtc_prefix_actions, x_t)
            if rtc_guidance is not None:
                x_t = x_t.detach().requires_grad_(requires_grad=True)
            if rtc_prefix_mask is None:
                expanded_time = flow_time.expand(bsize)
            else:
                postfix_time = flow_time.expand(bsize, self.config.action_horizon)
                expanded_time = torch.where(rtc_prefix_mask, torch.zeros_like(postfix_time), postfix_time)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            if rtc_guidance is not None:
                v_t = v_t + self._rtc_guidance_correction(x_t, v_t, flow_time, rtc_guidance)

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            if rtc_guidance is not None:
                x_t = x_t.detach()
            flow_time += dt
            denoise_steps += 1
        # The final Euler update also changes the committed prefix. Restore it
        # once more so downstream execution receives the exact in-flight plan.
        if rtc_prefix_mask is not None:
            x_t = torch.where(rtc_prefix_mask[:, :, None], rtc_prefix_actions, x_t)
        _sync_if_cuda(device)
        timing["flow_denoise_ms"] = (time.perf_counter() - flow_start) * 1000
        timing["flow_denoise_steps"] = denoise_steps
        timing["model_sample_total_ms"] = (time.perf_counter() - total_start) * 1000
        self._last_sample_timing = timing
        return x_t

    def _prepare_rtc_prefix(self, x_t, rtc_prefix):
        """Prepare hard-prefix actions and masks for training-time RTC inference."""
        if rtc_prefix is None:
            return None, None
        action_prefix = rtc_prefix.get("action_prefix", rtc_prefix.get("target_actions"))
        if action_prefix is None:
            raise ValueError("rtc_prefix must contain `action_prefix` or `target_actions`.")
        action_prefix = action_prefix.to(device=x_t.device, dtype=x_t.dtype)
        if action_prefix.ndim == 2:
            action_prefix = action_prefix[None, ...]
        if action_prefix.shape[0] == 1 and x_t.shape[0] > 1:
            action_prefix = action_prefix.expand(x_t.shape[0], -1, -1)
        if action_prefix.shape != x_t.shape:
            raise ValueError(f"rtc_prefix action shape {action_prefix.shape} must equal sample shape {x_t.shape}.")
        if not torch.isfinite(action_prefix).all():
            raise ValueError("rtc_prefix actions contain NaN or Inf")

        delay = rtc_prefix.get("delay", rtc_prefix.get("prefix_steps"))
        if delay is None:
            raise ValueError("rtc_prefix must contain `delay` or `prefix_steps`.")
        raw_delay = torch.as_tensor(delay, device=x_t.device)
        if raw_delay.dtype == torch.bool or (
            torch.is_floating_point(raw_delay)
            and (not torch.isfinite(raw_delay).all() or not torch.equal(raw_delay, raw_delay.round()))
        ):
            raise ValueError("rtc_prefix delay must contain finite integer values")
        delay = raw_delay.to(dtype=torch.long)
        if delay.ndim == 0:
            delay = delay.expand(x_t.shape[0])
        if delay.shape[0] == 1 and x_t.shape[0] > 1:
            delay = delay.expand(x_t.shape[0])
        if delay.shape != (x_t.shape[0],):
            raise ValueError(f"rtc_prefix delay shape {delay.shape} must be `(batch_size,)`.")
        if torch.any(delay < 0) or torch.any(delay > self.config.action_horizon):
            raise ValueError(f"rtc_prefix delay must be in [0, {self.config.action_horizon}]")
        step_ids = torch.arange(self.config.action_horizon, device=x_t.device)[None, :]
        return action_prefix, step_ids < delay[:, None]

    def _rtc_guidance_correction(self, x_t, v_t, time, rtc_guidance):
        """Compute the PiGDM velocity correction for Real-Time Chunking."""
        target_actions = rtc_guidance["target_actions"].to(device=x_t.device, dtype=x_t.dtype)
        weights = rtc_guidance["weights"].to(device=x_t.device, dtype=x_t.dtype)
        beta = torch.as_tensor(rtc_guidance.get("beta", 0.0), dtype=x_t.dtype, device=x_t.device)
        eps = torch.as_tensor(rtc_guidance.get("eps", 1e-4), dtype=x_t.dtype, device=x_t.device)
        if float(beta.detach().cpu()) <= 0.0:
            return torch.zeros_like(v_t)

        tau = torch.clamp(time.to(dtype=x_t.dtype, device=x_t.device), min=eps, max=1.0 - eps)
        a_hat_1 = x_t + (1.0 - tau) * v_t
        weighted_error = (target_actions - a_hat_1) * weights

        # VJP: weighted_error @ d(a_hat_1) / d(x_t)
        objective = (a_hat_1 * weighted_error.detach()).sum()
        grad = torch.autograd.grad(objective, x_t, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(v_t)

        one_minus_tau = 1.0 - tau
        r_tau_sq = (one_minus_tau * one_minus_tau) / (tau * tau + one_minus_tau * one_minus_tau + eps)
        guidance_scale = torch.minimum(beta, (one_minus_tau / tau) * r_tau_sq)
        return guidance_scale * grad

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
