# Free-model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier providers, routing each request to
the best free model that meets its requirements and failing over when a
provider refuses.

Entry for the make.fun bounty `free-model-aggregator-01`.

## Run

```sh
node main.ts            # serve a trial (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
node main.ts --list     # print the provider catalog and exit
```

Node 22+, standard library only. The server reads the brief at
`$MAKEFUN_PROMPT_FILE`, probes each provider's catalog route, registers its
port and catalog with the env at `$MAKEFUN_ENV_URL`, serves the 20 requests
the env's client sends, and exits 0 when the workload is done.

## How it routes

- Candidates are the free models across all providers, ranked by the quality
  table's rating, with the least-used provider winning ties so calls spread.
- A request's requirements come from the workload manifest by arrival order,
  unioned with what the body itself carries (tools, images). JSON-mode
  requests on the OpenAI wire only go to wires that can carry the signal.
- The request is translated faithfully into the provider's wire (OpenAI,
  Anthropic or Gemini): system text, user text, images, tool names, stop
  sequences, max_tokens and JSON mode are preserved exactly, because the
  provider echoes what it parsed.
- On a 402, 429, 5xx or unreachable provider, the request fails over to the
  next-best model on another provider. A provider that refuses three times
  running is treated as out for the rest of the trial.
- Answers are buffered and re-emitted in the client's wire, JSON or
  streaming, with the provider's text whole, its usage counts, and its stop
  reason mapped. Everything is deterministic: no clock, no unseeded
  randomness, fixed timestamps.

## Files

- `main.ts` — the whole server: brief parsing, catalog probing, routing,
  wire translation, response shaping, register and work polling.
- `LICENSE` — MIT.
