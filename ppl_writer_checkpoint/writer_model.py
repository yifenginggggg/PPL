"""Qwen-LoRA Writer that turns strict history into fixed-width latent slots.

The Writer is query independent.  It appends learned memory-token embeddings to
the registered history, runs the Qwen Transformer once, and projects the final
memory-token hidden states into the existing Profile6 slot interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


class QwenLoRALatentWriter(nn.Module):
    """Qwen backbone + q/v LoRA + learned memory tokens + slot projector."""

    SCHEMA = "qwen_lora_latent_writer_v1"

    def __init__(
        self,
        model: nn.Module,
        *,
        hidden_size: int,
        num_memory_tokens: int = 64,
        slot_dim: int = 512,
        lora_rank: int = 64,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_memory_tokens <= 0 or slot_dim <= 0:
            raise ValueError("Writer dimensions must be positive")
        if lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        self.model = model
        self.hidden_size = int(hidden_size)
        self.num_memory_tokens = int(num_memory_tokens)
        self.slot_dim = int(slot_dim)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else lora_rank)
        self.lora_dropout = float(lora_dropout)
        self.memory_embeddings = nn.Parameter(
            torch.empty(self.num_memory_tokens, self.hidden_size)
        )
        nn.init.normal_(self.memory_embeddings, std=self.hidden_size**-0.5)
        self.slot_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.slot_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.slot_dim),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        num_memory_tokens: int = 64,
        slot_dim: int = 512,
        lora_rank: int = 64,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.05,
        torch_dtype: torch.dtype | str | None = torch.bfloat16,
        device_map: str | Mapping[str, int] | None = None,
        attn_implementation: str | None = "flash_attention_2",
    ) -> "QwenLoRALatentWriter":
        """Load a local Qwen checkpoint and install q/v LoRA on every layer."""

        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        causal_lm = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
        config = getattr(causal_lm, "config", None)
        hidden_size = int(
            getattr(config, "hidden_size", getattr(config, "d_model", 0))
        )
        if hidden_size <= 0:
            raise ValueError("could not infer Qwen hidden size")
        for parameter in causal_lm.parameters():
            parameter.requires_grad_(False)
        lora_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=float(lora_alpha if lora_alpha is not None else lora_rank),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        peft_model = get_peft_model(causal_lm, lora_config)
        peft_model.config.use_cache = False
        writer = cls(
            peft_model,
            hidden_size=hidden_size,
            num_memory_tokens=num_memory_tokens,
            slot_dim=slot_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        # ``device_map`` dispatches the Qwen modules, but custom modules added
        # after loading stay on CPU unless placed explicitly.
        input_device = writer.model.get_input_embeddings().weight.device
        writer.memory_embeddings.data = writer.memory_embeddings.data.to(input_device)
        writer.slot_projector.to(input_device)
        return writer

    @property
    def dtype(self) -> torch.dtype:
        return self.memory_embeddings.dtype

    @property
    def transformer(self) -> nn.Module:
        """Return the Qwen Transformer carrying the installed LoRA modules."""

        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        transformer = getattr(base, "model", None)
        if transformer is None:
            raise TypeError("the causal model must expose a .model Transformer")
        return transformer

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Expose trainable components for logging and optimizer construction."""

        lora = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        custom = [self.memory_embeddings, *self.slot_projector.parameters()]
        return {"lora": lora, "memory_and_projector": custom}

    def trainable_parameter_count(self) -> dict[str, int]:
        return {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in self.trainable_parameter_groups().items()
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode history and return ``[batch, 64, slot_dim]`` slots.

        ``input_ids`` contains only the registered history.  Memory embeddings
        are appended after it; no answer, query, profile, or benchmark labels
        enter this forward path.
        """

        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer tensor [batch, history_tokens]")
        device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(device=device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids")
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        embeddings = self.model.get_input_embeddings()(input_ids)
        memory = self.memory_embeddings.to(device=embeddings.device, dtype=embeddings.dtype)
        memory = memory.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        inputs_embeds = torch.cat([embeddings, memory], dim=1)
        memory_mask = torch.ones(
            input_ids.size(0), self.num_memory_tokens,
            device=attention_mask.device, dtype=attention_mask.dtype,
        )
        full_mask = torch.cat([attention_mask, memory_mask], dim=1)
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1][:, -self.num_memory_tokens :, :]
        projector_dtype = next(self.slot_projector.parameters()).dtype
        hidden = hidden.to(dtype=projector_dtype)
        slots = self.slot_projector(hidden)
        if return_hidden:
            return slots, hidden
        return slots

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "hidden_size": self.hidden_size,
            "num_memory_tokens": self.num_memory_tokens,
            "slot_dim": self.slot_dim,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": ["q_proj", "v_proj"],
        }

    def custom_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "memory_embeddings": self.memory_embeddings.detach().cpu().contiguous(),
            **{
                f"slot_projector.{key}": value.detach().cpu().contiguous()
                for key, value in self.slot_projector.state_dict().items()
            },
        }

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only PEFT adapter weights, never the frozen 4B base."""

        from peft import get_peft_model_state_dict

        return {
            key: value.detach().cpu().contiguous()
            for key, value in get_peft_model_state_dict(self.model).items()
        }


__all__ = ["QwenLoRALatentWriter"]
