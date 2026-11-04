# PPL-Writer checkpoint

`writer.pt` is the frozen epoch-5 PPL-Writer checkpoint. It is distributed as
a GitHub Release asset, not committed to the repository.

- SHA256: `03842befa219cd0aad73c4ddd5318ca7301c73985a514de90c8901c60549e592`
- Size: 29,562,604 bytes
- Schema: `qwen_lora_latent_writer_checkpoint_v1`
- Epoch: 5
- Base model: `Qwen3-4B-Instruct-2507`
- LoRA: rank 16, alpha 16, dropout 0.05, targets `q_proj` and `v_proj`
- Writer: 64 memory tokens, slot dimension 512, hidden size 2560

## Files

```text
writer.pt        Release asset: LoRA adapter and writer state
config.json      checkpoint architecture and training settings
metrics.json     epoch-5 training metrics
writer_model.py  QwenLoRALatentWriter implementation
load_writer.py   checkpoint loader and forward smoke test
```

Download `writer.pt` from the repository's Releases page and place it in this
directory, or pass its path with `--checkpoint`. The checkpoint is not a full
model. It contains 144 LoRA tensors in `adapter_state`, plus
`memory_embeddings` and the slot projector in `custom_state`. The
`Qwen3-4B-Instruct-2507` base weights are not included.

## Loading

Requires `torch`, `transformers` and `peft`, plus a local copy of the base
model:

```bash
python3 load_writer.py \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --checkpoint writer.pt \
  --device-map cuda:0
```

The loader attaches the LoRA adapters, restores the memory and projector
state, and runs one forward pass. A successful run reports `slots_shape`
`[1, 64, 512]`, `slots_finite: true`, and zero missing or unexpected adapter
and writer tensors.
