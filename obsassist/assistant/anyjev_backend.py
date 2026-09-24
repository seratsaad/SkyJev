"""AnyJev backends for this project.

`LocalHF` is AnyJev's transformers backend with a compatibility layer for transformers 5.x
(argument renames in the causal-mask helper and the decoder layers), Apple-GPU memory hygiene
(bucketed padding, cache release, a memory cap) and defaults (MPS, bfloat16). Nothing in AnyJev
itself is modified.
"""

from __future__ import annotations

import inspect
import os


def load_anyjev():
    """AnyJev is a dependency (pinned in pyproject.toml, extra `decide`)."""
    try:
        import anyjev
    except ImportError as e:
        raise ImportError('AnyJev is not installed: pip install -e ".[decide]" (see README)') from e
    return anyjev


def default_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class _BucketedTokenizer:
    """Pads every batch to a multiple of `multiple` tokens. On Apple GPUs (MPS) each new tensor
    shape compiles and caches a new graph; per-batch padding makes almost every batch a new
    shape, and the cache grows until the machine swaps. Everything else is forwarded."""

    def __init__(self, tok, multiple: int = 64):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_multiple", multiple)

    def __call__(self, *args, **kwargs):
        if kwargs.get("padding"):
            kwargs.setdefault("pad_to_multiple_of", self._multiple)
        return self._tok(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)


def make_local_backend(
    model_name: str = "Qwen/Qwen3-1.7B", device: str = None, dtype: str = "bfloat16", batch_size: int = 8
):
    # keep MPS allocations below physical memory: fail loudly instead of swapping (must precede torch init)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
    load_anyjev()
    from anyjev.backends.hf import HFBackend

    class LocalHF(HFBackend):
        """HFBackend + transformers 5.x compatibility for the L2 block loop + MPS memory hygiene."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if str(self.device).startswith("mps"):
                self.tokenizer = _BucketedTokenizer(self.tokenizer)

        def _release(self):
            if str(self.device).startswith("mps"):
                import torch

                torch.mps.empty_cache()

        def next_token_logprobs(self, prompts, token_ids):
            try:
                return super().next_token_logprobs(prompts, token_ids)
            finally:
                self._release()

        def hidden_states_to(self, *args, **kwargs):
            try:
                return super().hidden_states_to(*args, **kwargs)
            finally:
                self._release()

        def hidden_states(self, *args, **kwargs):
            try:
                return super().hidden_states(*args, **kwargs)
            finally:
                self._release()

        def _prepare(self, enc, pos):
            import torch

            inner = getattr(self.model, "model", None)
            if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
                raise NotImplementedError("no .model.layers / .model.embed_tokens on this architecture")
            cfg = getattr(self.model.config, "text_config", self.model.config)
            if any(t not in ("full_attention", "sliding_attention") for t in (getattr(cfg, "layer_types", None) or [])):
                # hybrid linear-attention stacks (Qwen3.5 Gated DeltaNet) need their own recurrent
                # cache; AnyJev falls back to a full forward with output_hidden_states
                raise NotImplementedError("hybrid linear-attention layers: block loop not supported")
            try:
                from transformers import DynamicCache
                from transformers.masking_utils import create_causal_mask
            except ImportError as e:
                raise NotImplementedError(str(e))
            ids, mask = enc["input_ids"], enc["attention_mask"]
            embeds = inner.embed_tokens(ids)
            cache = DynamicCache()
            cache_position = torch.arange(0, embeds.shape[1], device=embeds.device)
            params = inspect.signature(create_causal_mask).parameters
            emb_key = "inputs_embeds" if "inputs_embeds" in params else "input_embeds"
            kw = {
                "config": self.model.config,
                emb_key: embeds,
                "attention_mask": mask,
                "past_key_values": cache,
                "position_ids": pos,
            }
            if "cache_position" in params:
                kw["cache_position"] = cache_position
            masks = {"full_attention": create_causal_mask(**kw)}
            types = {getattr(layer, "attention_type", "full_attention") for layer in inner.layers}
            if "sliding_attention" in types:
                from transformers.masking_utils import create_sliding_window_causal_mask

                masks["sliding_attention"] = create_sliding_window_causal_mask(**kw)
            if not hasattr(inner, "rotary_emb"):
                raise NotImplementedError("no shared rotary embedding module on this architecture")
            rope = inner.rotary_emb(embeds, pos)
            if getattr(inner, "embed_scale", None) is not None:
                raise NotImplementedError("scaled embeddings are not supported by the block loop")
            return {
                "hidden": embeds,
                "masks": masks,
                "cache": cache,
                "cache_position": cache_position,
                "position_ids": pos,
                "rope": rope,
            }

        def _run_layers(self, ctx, start, stop, capture=None):
            inner = self.model.model
            h = ctx["hidden"]
            layer0 = inner.layers[0]
            params = inspect.signature(layer0.forward).parameters
            kv_key = "past_key_values" if "past_key_values" in params else "past_key_value"
            for i in range(start, stop):
                layer = inner.layers[i]
                kw = {
                    "attention_mask": ctx["masks"][getattr(layer, "attention_type", "full_attention")],
                    "position_ids": ctx["position_ids"],
                    kv_key: ctx["cache"],
                    "position_embeddings": ctx["rope"],
                }
                if "cache_position" in params:
                    kw["cache_position"] = ctx["cache_position"]
                out = layer(h, **kw)
                h = out[0] if isinstance(out, tuple) else out
                if capture is not None:
                    capture(i + 1, h)
            ctx["hidden"] = h
            ctx["layer"] = stop
            return h

    return LocalHF(model_name, device=device or default_device(), dtype=dtype, batch_size=batch_size)
