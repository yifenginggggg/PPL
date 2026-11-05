# PPL-Writer checkpoint

The release provides the epoch-two PPL-Writer adapter (`checkpoint-1278`),
trained with Progressive Profile Learning on PersonaMem+ profile transitions.

- Base model: `Qwen/Qwen3.5-4B`
- Adapter: LoRA, rank 16, alpha 32, dropout 0.05
- Training: supervised fine tuning on complete profile transitions, seed 42,
  learning rate `1e-4`, effective batch size 8, BF16, 38,400-token sequence
  limit, two evaluated epochs
- Distribution: `writer_checkpoint_1278.zip` is a GitHub Release asset, not
  committed to the repository

## Files

```text
writer_checkpoint_1278.zip  Release asset: `adapter_config.json` and `adapter_model.safetensors`
config.json                 checkpoint architecture and training settings
load_writer.py              adapter loader and forward smoke test
```

Download the release asset and unpack it:

```bash
unzip writer_checkpoint_1278.zip -d checkpoint-1278
```

The adapter is not a full model. The `Qwen3.5-4B` base weights are not
included.

## Loading

Requires `torch`, `transformers` and `peft`, plus a local copy of the base
model:

```bash
python3 load_writer.py \
  --model-path /path/to/Qwen3.5-4B \
  --checkpoint checkpoint-1278 \
  --device-map cuda:0
```

The loader attaches the LoRA adapter and runs one forward pass. A successful
run reports the adapter rank and alpha, the number of trainable parameters,
and finite logits.
