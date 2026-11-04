# FactBlock protocol

The public pipeline evaluates one profile against one history with two model
calls and deterministic Python scoring.

```text
history -> source_blocks
source_blocks + profile -> profile_blocks + links
FactBlocks + history operations -> FactEvents and materials
materials -> Completeness / Support / Reliability
```

## FactBlock

Each block is one independently assessable proposition. Compound text is split
into atomic blocks. Source blocks preserve the exact history evidence; profile
blocks preserve the exact profile-unit evidence.

```json
{
  "fact_id": "SF0001",
  "origin": "source",
  "stage": 3,
  "profile_id": null,
  "unit_id": null,
  "text": "The user likes spicy food.",
  "kind": "preference",
  "owner": "user",
  "evidence_strength": "explicit",
  "importance": "required",
  "state": "active",
  "assertion_status": "current",
  "condition_type": "none",
  "condition": null,
  "evidence": [
    {"kind": "source", "sid": "u3", "stage": 3, "quote": "I like spicy food."}
  ],
  "links": [],
  "uncertainty": null
}
```

Profile blocks use the same fields, with `origin=profile`, a profile ID, a
unit ID, and profile evidence:

```json
{
  "fact_id": "PF0001",
  "origin": "profile",
  "stage": 8,
  "profile_id": "profile",
  "unit_id": "profile/C1",
  "text": "The user likes spicy food.",
  "kind": "preference",
  "owner": "user",
  "evidence_strength": "explicit",
  "importance": "required",
  "state": "active",
  "assertion_status": "current",
  "condition_type": "none",
  "condition": null,
  "evidence": [
    {"kind": "profile", "profile_id": "profile", "unit_id": "profile/C1", "cid": 1, "span": "The user likes spicy food."}
  ],
  "links": [{"fact_id": "SF0001", "relation": "equivalent"}],
  "uncertainty": null
}
```

Allowed values are defined in `fact_pipeline.py`. Links use:

```text
equivalent | partial | contradictory | unsupported | overgeneralized | stale | uncertain
```

Every source quote and profile span must be an exact contiguous substring of
the supplied input. Unknown IDs and invalid field inventories are rejected;
invalid enum values are normalized to `uncertain`. Unknown links are dropped,
missing profile-unit coverage is inferred as uncertain, and invalid evidence is
repaired or downgraded; each change is recorded in normalization warnings. An
unresolved semantic relation is retained as `uncertain`; it is not silently
converted to an error label.

## FactEvent

FactEvent is a script-generated record of an operation observed in history. The
model does not generate a separate event response. The public compiler
currently materializes explicit withdrawal requests from the history as events.

```json
{
  "event_id": "W_u5",
  "event_type": "withdraw",
  "stage": 5,
  "target_fact_id": "SF0001",
  "before_fact_id": "SF0001",
  "after_fact_id": null,
  "certainty": "clear",
  "scope": "spicy food recommendations",
  "evidence": [
    {"kind": "source", "sid": "u5", "stage": 5, "quote": "Please stop recommending spicy food."}
  ],
  "uncertainty": null
}
```

FactEvents describe observed history changes. They do not certify that a
profile update succeeded and they do not receive an independent score.

## Public materials

Python derives three public material views:

| Material | Contents |
|---|---|
| `source_catalogue` | Source FactBlocks and derived FactEvents. |
| `profile_claims` | Profile FactBlocks grouped by profile unit with links and coverage. |
| `uncertain` | Unresolved blocks, links and extraction coverage. |

The old condition, attribution, withdrawal and fact-fidelity material views
are not part of the public contract.

## Public metrics

- **Completeness**: exact recall of active, required, user-owned source facts
  linked by an equivalent profile claim.
- **Support**: grounded F1 between common-source recall and profile-claim
  precision. Partial links receive half relation credit.
- **Reliability**: the fraction of substantive profile blocks with a valid
  link, a resolved assertion status and no severe relation.

All three metrics return their rate and denominator counts. The profile stage
is input data. The API accepts any positive profile stage and does not require
a multi-profile inventory.
