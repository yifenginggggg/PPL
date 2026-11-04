# Persona records: history + eight stage profiles

999 JSON files, one per PersonaMem-v2 persona. Persona 558 is absent because no
authentic raw data or 32K history exists in any upstream snapshot.

```text
persona_<id>.json
  persona_id
  history              original chat_history list (all messages)
  history_chunk_1      chronological messages used by stage 1
  profile_stage_1      complete cumulative profile after stage 1
  ...
  history_chunk_8
  profile_stage_8
  metadata             source hashes, selection counts, per-stage model
```

`profile_stage_i` is the complete profile after stages 1..i, not a fragment of
stage i. It is never valid to read `profile_stage_i` as an update that applies
only to `history_chunk_i`.

## How chunks map to history

The profile pipeline did not read every raw message. It removed system
scaffolding and empty messages, kept the first `m` messages, and split those
into eight contiguous stage ranges. `metadata.selection` records the counts,
and each stage records `message_start` / `message_end` in that filtered
sequence. For personas with `trimmed_messages > 0`, the tail was outside the
32K budget and is present in `history` but not in any chunk.

`history` is the original file including the synthetic system message; the
chunks start from the filtered sequence. When slicing manually, apply the same
filter first: drop `role == "system"` and empty content, then take
`selected_messages`.

## Provenance

All 7,992 stages were written by `gpt-5.6-sol` using `sol_fast_v1`.

Every stage carries its own `generation_model`, `generation_protocol`,
`profile_sha256`, token counts and validation record in `metadata.stages`.

## Integrity

`manifest.jsonl` has one row per persona with the output SHA256 and stage
models. `manifest.json` reports 999 personas and the missing ID.
