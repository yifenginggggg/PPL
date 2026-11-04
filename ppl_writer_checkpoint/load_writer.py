#!/usr/bin/env python3
"""Load the released PPL-Writer checkpoint and run a forward smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from writer_model import QwenLoRALatentWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="local Qwen3-4B-Instruct-2507 directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("writer.pt"))
    parser.add_argument(
        "--device-map",
        default="",
        help="transformers device map, e.g. cuda:0 or cpu; defaults to cuda:0 when available",
    )
    parser.add_argument(
        "--attn-implementation",
        default="",
        help="optional transformers attention implementation, e.g. flash_attention_2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_map = args.device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device_map == "cpu" else torch.bfloat16

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["writer_config"]
    writer = QwenLoRALatentWriter.from_pretrained(
        args.model_path,
        num_memory_tokens=config["num_memory_tokens"],
        slot_dim=config["slot_dim"],
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=args.attn_implementation or None,
    )
    expected_adapter = get_peft_model_state_dict(writer.model)
    adapter_missing = sorted(
        set(expected_adapter) - set(checkpoint["adapter_state"])
    )
    adapter_unexpected = sorted(
        set(checkpoint["adapter_state"]) - set(expected_adapter)
    )
    if adapter_missing:
        raise RuntimeError(
            f"checkpoint is missing {len(adapter_missing)} adapter tensors"
        )
    if adapter_unexpected:
        raise RuntimeError(
            f"checkpoint has {len(adapter_unexpected)} unexpected adapter tensors"
        )
    set_peft_model_state_dict(writer.model, checkpoint["adapter_state"])

    custom_state = checkpoint["custom_state"]
    memory_key = "memory_embeddings"
    projector_prefix = "slot_projector."
    if memory_key not in custom_state:
        raise RuntimeError(f"checkpoint is missing {memory_key}")
    projector_keys = {
        key for key in custom_state if key.startswith(projector_prefix)
    }
    projector_state = {
        key[len(projector_prefix):]: custom_state[key]
        for key in projector_keys
    }
    unknown = set(custom_state) - {memory_key} - projector_keys
    if unknown:
        raise RuntimeError(
            f"checkpoint has {len(unknown)} unknown writer tensors"
        )
    memory_result = writer.load_state_dict(
        {memory_key: custom_state[memory_key]}, strict=False
    )
    projector_result = writer.slot_projector.load_state_dict(
        projector_state, strict=False
    )
    custom_missing = [
        key for key in memory_result.missing_keys if key == memory_key
    ] + list(projector_result.missing_keys)
    custom_unexpected = (
        list(memory_result.unexpected_keys)
        + list(projector_result.unexpected_keys)
    )
    if custom_missing:
        raise RuntimeError(
            f"checkpoint is missing {len(custom_missing)} writer tensors"
        )
    if custom_unexpected:
        raise RuntimeError(
            f"checkpoint has {len(custom_unexpected)} unexpected writer tensors"
        )
    writer.eval()

    input_ids = torch.randint(0, 1000, (1, 16), dtype=torch.long)
    with torch.no_grad():
        slots = writer(input_ids)

    print(
        json.dumps(
            {
                "epoch": checkpoint.get("epoch"),
                "adapter_tensors": len(checkpoint["adapter_state"]),
                "adapter_missing": len(adapter_missing),
                "adapter_unexpected": len(adapter_unexpected),
                "custom_tensors": len(custom_state),
                "custom_missing": len(custom_missing),
                "custom_unexpected": len(custom_unexpected),
                "slots_shape": list(slots.shape),
                "slots_finite": bool(torch.isfinite(slots).all().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
