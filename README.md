# Progressive Profile Learning (PPL)

A stage-profile dataset, the profile generation pipeline, the profile
evaluation pipeline, and the PPL-Writer checkpoint.

## Contents

| Path | Description |
| --- | --- |
| `dataset/` | 999 persona records: original chat history, eight chronological history chunks under the 32K budget, eight cumulative stage profiles |
| `profile_generation_pipeline/` | Takes an API endpoint and a history; writes eight cumulative stage profiles |
| `profile_evaluation_pipeline/` | Evaluates one profile against its history on Completeness, Support and Reliability |
| `ppl_writer_checkpoint/` | PPL-Writer config, loader and smoke test; `writer.pt` is a GitHub Release asset |

## Dataset

Each `dataset/persona_<id>.json` contains `persona_id`, the original
`history`, and paired `history_chunk_i` / `profile_stage_i` entries for eight
stages. `profile_stage_i` is the complete cumulative profile after stage 1..i,
not a fragment of stage i.

The dataset covers 999 personas (IDs 0-999 except 558) and 7,992 profiles.
Persona 558 is absent because no authentic raw data or 32K history exists in
any upstream snapshot. All 7,992 stages were written by `gpt-5.6-sol`; each
stage records its own model. See [dataset/README.md](dataset/README.md) for the
schema and chunk reconstruction rule.

## Generation

```bash
export PROFILE_API_KEY=...
python3 profile_generation_pipeline/pipeline.py \
  --history history.json \
  --output-dir outputs/persona_25 \
  --model <model> \
  --api-base-url <openai-compatible-base-url>
```

See [profile_generation_pipeline/README.md](profile_generation_pipeline/README.md).

## Evaluation

```python
from profile_eval.blackbox import evaluate_final

result = evaluate_final(
    history=[{"id": "u1", "text": "I like spicy food."}],
    profile={"id": "profile", "text": "The user likes spicy food."},
    llm=my_model_client,
    model_id="your-model",
)
```

Run from `profile_evaluation_pipeline/`, or add it to `PYTHONPATH`. See
[profile_evaluation_pipeline/README.md](profile_evaluation_pipeline/README.md).

## Status

Pre-release. The PersonaMem-v2 histories are redistributed for research use;
confirm the upstream license and privacy terms before public distribution.
Code licensing is not finalized.
