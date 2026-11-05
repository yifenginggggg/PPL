#!/usr/bin/env python3
"""Load the released PPL-Writer LoRA adapter and run a forward smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="local Qwen/Qwen3.5-4B directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint-1278"),
        help="unpacked LoRA adapter directory",
    )
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
    parser.add_argument(
        "--prompt",
        default="Profile update",
        help="text used for the forward smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_map = args.device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device_map == "cpu" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=args.attn_implementation or None,
    )
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    peft_config = next(iter(model.peft_config.values()))
    target_modules = peft_config.target_modules
    if isinstance(target_modules, set):
        target_modules = sorted(target_modules)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        json.dumps(
            {
                "base_model": peft_config.base_model_name_or_path,
                "adapter_rank": peft_config.r,
                "adapter_alpha": peft_config.lora_alpha,
                "adapter_dropout": peft_config.lora_dropout,
                "target_modules": target_modules,
                "trainable_parameters": trainable,
                "logits_shape": list(output.logits.shape),
                "logits_finite": bool(torch.isfinite(output.logits).all().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
