# Profile evaluation pipeline

This package exposes one provider-neutral black box for evaluating a single
profile against its history.

```text
history + profile + model client
             |
             v
       history -> source_blocks
              |
              v
 source_blocks + profile -> profile_blocks + links
             |
             v
  Python preprocessing and three fixed judgments
             |
             v
 Completeness / Support / Reliability
```

The public entrypoint is `profile_eval.blackbox.evaluate_final`:

```python
from profile_eval.blackbox import evaluate_final

result = evaluate_final(
    history=[{"id": "u1", "text": "I like spicy food."}],
    profile={"id": "profile", "text": "The user likes spicy food."},
    llm=my_model_client,
    model_id="your-model",
)
```

The caller supplies only `history`, `profile`, an LLM client and a model ID.
The profile may have any positive stage number; when omitted, the full-history
default is used. History messages may carry a positive `stage`; messages without
one inherit the current stage, which starts at 1. For dataset stage scopes,
flatten `history_chunk_i` and tag those messages with stage `i`. The
single-profile API depends only on the supplied history and profile. For
multi-profile evaluation, `profile_eval.blackbox.evaluate_batch` extracts the
history source
catalogue once and projects each supplied profile against that immutable
catalogue.

The history extraction prompt and profile extraction prompt are fixed. The
first call receives only history and returns `source_blocks`; the second
receives the immutable `source_blocks` plus the profile and returns
`profile_blocks`, links, coverage and uncertainty. Python then derives
FactEvents, the public materials `source_catalogue`,
`profile_claims` and `uncertain`, and computes:

- **Completeness**: exact recall of active, required user facts from history.
- **Support**: grounded F1 combining source recall and profile-claim precision.
- **Reliability**: profile blocks without missing links, severe relations or
  unresolved assertion status.

The result keeps the fact graph, evidence validation, task items and score
denominators for auditability.

## Model client contract

The client must provide:

```python
generate(*, system_prompt: str, payload: dict, model_id: str) -> object
```

The response may be a JSON object, JSON string or `ModelReply`. Credentials and
provider selection remain outside this package.

## Validation

Exact source and profile evidence is checked against the supplied input.
Malformed graphs and invalid FactBlock/FactEvent fields are rejected. Unknown
links are dropped, missing profile-unit coverage is inferred as uncertain, and
invalid evidence is repaired or downgraded; each change is recorded in
normalization warnings. Provider paraphrases are never silently treated as
exact evidence.

This package contains only this two-stage FactBlock pipeline, its model
backends, and focused tests. The optional batch helper reuses one history
source call across the supplied profiles.

See [profile_eval/BLACKBOX_API.md](profile_eval/BLACKBOX_API.md) for the
machine-readable result envelope and [profile_eval/FACT_BLOCK_SPEC.md](profile_eval/FACT_BLOCK_SPEC.md)
for the FactBlock schema.
