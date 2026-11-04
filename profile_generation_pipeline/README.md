# Profile generation pipeline

One chat history in, eight cumulative stage profiles out. This is the frozen
single-rewrite writer: at each stage the model receives the previous
profile plus the next chronological history chunk and rebuilds the complete
profile from a blank page as continuous English prose.

## Quick start

```bash
export PROFILE_API_KEY=...
python3 pipeline.py \
  --history history.json \
  --output-dir outputs/persona_25 \
  --model gpt-5.6-sol \
  --api-base-url https://api.openai.com/v1
```

Use any OpenAI-compatible endpoint. `--api-base-url` may point at a base URL
(`.../v1`), a full chat-completions URL, or a Responses API URL. The key is
read from the environment variable named by `--api-key-env` (default
`PROFILE_API_KEY`) and is never written to disk.

Check the input without spending API calls:

```bash
python3 pipeline.py --history history.json --output-dir /tmp/profile_dry \
  --model gpt-5.6-sol --dry-run
```

### Codex-backed smoke runs

If the model is available through a signed-in Codex CLI rather than an HTTP
key, use the `codex` backend. It embeds the system instructions in a single
prompt because `codex exec` has no separate system role.

```bash
python3 pipeline.py --history history.json --output-dir outputs/codex_smoke \
  --model <codex-model> --backend codex
```

## Input

Two formats are accepted:

- A raw role-marked message list: `[{"role": "user", "content": "..."}, ...]`,
  or an object with a `messages`, `history`, `chat_history`, `turns`, `chat`,
  or `conversation` list.
- A pre-chunked stage list: eight objects each with a `messages` list. To use
  a released persona record, build this list from its `history_chunk_1` through
  `history_chunk_8` fields. Pre-chunked input defines the stage count and is
  used as-is.

Only `user` messages become evidence. They are rendered as
`USER MESSAGE ID u01` blocks, matching the released input format. System and
developer messages are removed before splitting; assistant messages remain in
the chronological ranges but are not rendered as evidence. Raw histories are
truncated at `--max-history-tokens` (default 32768) and split into contiguous,
message-aligned stages.

## Output

```text
outputs/<run>/
  p1.txt ... p8.txt      complete replacement profile after each stage
  manifest.json          hashes, token counts, attempts, validation state
```

The manifest stores only hashes of the private history, never the history
text. Each stage records the previous-profile hash, so the cumulative chain is
verifiable after the run.

If a stage fails, rerun the same command with `--resume`. Accepted stages are
hash-verified against the manifest and reused; generation continues at the
first missing stage.

## Validation

Every output must pass the frozen deterministic checks before it is accepted:
below the token limit, nonempty English prose, a complete final sentence, no
JSON, headings, bullets, labels, tables, markup, or frozen-prompt leakage. An
invalid answer is retried up to `--max-attempts` and never silently accepted.

Token counts use the Qwen tokenizer when `--tokenizer-model` points at a local
tokenizer and `transformers` is installed. Otherwise the pipeline estimates
tokens as `max(chars / 4.5, words * 1.2)`, which tracks English chat histories
closely; use an exact tokenizer for dataset publication. Stages above
`--max-chunk-tokens` are recorded in `manifest.json` and reported as warnings.

## Scope

The pipeline implements only the single-rewrite writer. Other experimental
variants are not included. The embedded system prompt is frozen and verified
at startup against
`a73eb9b115391664479e84de1531bed6184114c4dabcf7eda8f25e24910efd6b`.
