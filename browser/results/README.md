# What is saved, and what it means

One JSON file per run, named `<task_id>__<arm or mode>__<run index>__<UTC timestamp>.json`.
`summary/` holds the tidy tables built from them by `analysis/summarize_results.py`.

Websites change. Every file carries its own timestamp, the commit of the agent under test, and the
model identifiers, so a result can always be read against the day it was taken.

## The fields that matter

| field | meaning |
|---|---|
| `success` | the deterministic verifier's answer, read from the page after the agent stopped. The only measure of success used anywhere |
| `verification` | which clauses passed, and `settle_seconds`, how long the page needed after the clock stopped |
| `status` | how the run ended: `done`, `blocked`, `stale_loop`, `timeout`, `ready` (stopped by an error) |
| `failure_type` | the automatic label. `failure_review.json` overrides it where a human checked |
| `duration_seconds` | first prediction to the accepted stop. The same clock boundary the JEV project reports |
| `setup_seconds` | opening the browser and the start URL, excluded from the clock above |
| `steps` | browser actions actually executed |
| `model_calls` | chooser calls plus text-helper calls |
| `stale_retries` | decisions the executor refused, which never reach `history` |
| `protocol_calls` | browser protocol calls, counted by wrapping the agent's own transport |
| `transient` | whether the run ever stood on a URL satisfying the verifier, and at which step |
| `ambiguity` | elements sharing a role and a name, on the first observation |
| `tabs_before`, `tabs_after` | page tabs open around the run. A click on `target="_blank"` shows up here |
| `env_probe` | iframes, canvas elements and shadow roots on the final page |
| `history` | every executed action: operation, target, label, text typed, latency, whether the page changed, URL |
| `element_tables` | what the chooser was offered at each decision |

## Failure labels

Measured, so a human pass adds nothing:

`iframe` · `canvas` · `dense_action_space` · `popup_tab` · `abandoned_success` · `browser_timeout` ·
`provider_error` · `timeout` · `agent_reported_failure` · `unverifiable`

A judgement call, worth checking by hand with `analysis/review_failures.py --needs-review`:

`false_done` · `unexecutable_target` · `incorrect_target` · `incorrect_text` · `no_progress` ·
`navigation_loop` · `stale_page` · `action_budget` · `element_not_exposed` · `error`

Two are worth explaining.

**`abandoned_success`** means the run reached a page that satisfies the verifier and then left. It is
detected by replaying the run's own URL history against the verifier's URL clauses, identically for
both agents, so it is not available for a verifier with no URL clause.

**`unexecutable_target`** means the chooser kept selecting an element the executor refused. Those
decisions never reach `history`, so the agent's own no-progress rule cannot see them. The runner
counts consecutive refusals and stops at eight.

## Things that are not agent behaviour

`unverifiable` means no open tab matched the agent's own last URL, so the check could not be taken.
It is recorded as a failure to measure, not as a failure by the agent, and those runs are re-run.

`browser_timeout` is the browser-harness daemon failing to answer within its five-second limit,
usually on a slow archive front page. Those runs are re-run rather than counted as agent failures,
and opening the start URL is retried three times before a run is written off. Running two browser
workloads at once causes them, so don't.

## Size

A single VizieR run saves about 2 MB, nearly all of it repeated element tables.
`analysis/compact_results.py` writes `shipped/`, which keeps the first and last table and drops the
rest. Every number quoted in the talk can still be recomputed from the shipped copies.
