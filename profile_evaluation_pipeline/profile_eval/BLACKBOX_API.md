# Public black-box API

`evaluate_final(history, profile, llm, model_id="model")` evaluates one
profile against one history. It has no filesystem or provider-specific
input requirement. `evaluate_batch(history, profiles, llm, model_id="model")`
uses the same prompts and scoring while sharing the history extraction across
multiple profiles.

## Request

- `history`: a list of user messages, or `{ "messages": [...] }`. A message
  may carry a positive `stage`; messages without one inherit the current stage,
  which starts at 1.
- `profile`: a text string or an object with `id`/`profile_id`, optional
  positive `stage` (defaults to 8 for a full-history profile), and `text` or
  `units`.
- `llm`: an object with `generate(system_prompt, payload, model_id)` or a
  compatible callable.
- `model_id`: provider/model metadata passed to the client.

The fixed history and profile prompts are called once each on a successful
single-profile run:

1. `history -> source_blocks`;
2. `source_blocks + profile -> profile_blocks + links`.

For `evaluate_batch`, the first call is shared and the second call runs once
per profile:

1. `history -> shared source_blocks`;
2. `shared source_blocks + each profile -> profile_blocks + links`.

If an attempt fails schema or evidence validation, the configured retry repeats
the failed attempt. `evaluate_final` reruns the two-stage protocol;
`evaluate_batch` retries only the failing source or profile stage. Python
computes all scores after validation.

## Result

```text
protocol_version
mode
model_id
provider_metadata
normalization_warnings
request
fact_graph
materials
tasks
scores
validation
```

`evaluate_batch` returns `source_catalogue` plus `results`; each item in
`results` has the same envelope as `evaluate_final` and the shared source
catalogue is copied into each item’s fact graph for independent validation.

`materials` contains only:

```text
source_catalogue
profile_claims
uncertain
```

`tasks` contains exactly:

```text
completeness
support
reliability
```

`scores.primary` contains the same three metrics. Each score reports
`rate`, `eligible_n`, `resolved_n`, `uncertain_n` and the applicable counts.

### Completeness

The fraction of active, required, user-owned source facts linked by an exact
equivalent profile claim. Partial links do not receive exact completeness
credit.

### Support

Grounded F1 between common-source recall and profile-claim precision. A profile
block without a valid source link is a precision failure; partial links receive
half relation credit.

### Reliability

The fraction of substantive profile blocks that have a source link, a resolved
assertion status and no severe relation (`unsupported`, `contradictory`,
`overgeneralized`, `stale` or `uncertain`).

The black box retains FactEvents as operation records in the fact graph.
