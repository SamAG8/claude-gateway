# ADR-0001 — Two engines behind one dispatcher

Status: accepted, 2026-09-21

## Context

Every request was answered the same way: spawn the local `claude` CLI and read its
`stream-json`. That transport rides this machine's Claude login, so a turn costs no
money — but it does cost *capacity*, and the subscription's throughput is the limit
the gateway actually runs into. Nimbus sends a great many turns that do not need
Claude's depth: naming a conversation, compacting a thread, reading an attachment,
answering a short question about the page in front of someone.

The goal is to move that traffic to a cheap, fast third-party model and keep Claude
for work that needs it. Latency is the primary objective — not cost. Cost moves in
the wrong direction here: turns that were free become turns that are billed.

## Decision

**One dispatcher, two transports.** `engine.py` keeps the name and stops running
anything: it selects, drives, and drains. `engines/cli.py` holds the subprocess
transport unchanged; `engines/openrouter.py` streams over HTTPS. Both yield the same
CanonicalEvents, so the Renderer, the Formatters and all three adapters are unaware
which one ran. No registry and no base class: two implementations do not justify a
framework, and the Formatter remains this project's one seam.

**The model map is the routing policy.** A resolved id starting with `openrouter/`
is answered by the HTTP engine. The client picks the model — the Nimbus extension
already has a deterministic per-turn router with context this process cannot see
(page size, attachments, the person's manual choice) — and `models.json` is
hot-reloaded, so re-pointing or disabling a tier is an edit, not a release. A second
prose-inspecting router in the gateway would be two copies of one policy, drifting.

**The gateway enforces only what the client could not know.** `select_engine` is
O(1): a string prefix, two lookups in the model map, and the block kinds of the
final turn. It runs before the lane is taken, touches no payload, and does no I/O.
Four constraints send a request back to Claude, each recorded as a `route_reason`:
an MCP identity is attached (`mcp`), the final turn carries a native document
(`document`), it carries an image the target model cannot see (`image`), or the
engine is switched off (`openrouter-disabled`).

**The MCP rule is hard.** The company-data tools are attached to the `claude` CLI
via `--mcp-config`. An engine with no tool loop cannot serve a turn that was given
an identity to look things up with, and a model that answers such a turn from
nothing produces the worst failure this product has: "I have no access to your
inbox", one turn after listing it. Whether a turn *should* carry that identity is
the client's decision, and it is made in the extension's router.

**One fallback, before the first chunk only.** If the upstream fails before any
content, the turn has not started and Claude can take it; after the first delta an
answer is half-written and no handover is possible. The fallback reloads the
subscription exactly during an upstream outage, which is the pressure this whole
change exists to relieve — so it is counted in the report rather than hidden, and
`OPENROUTER_FALLBACK=0` trades it back for a visible failure.

**An upstream 401 is never forwarded.** It is the gateway's own credential that is
wrong. Passed through, the Nimbus extension maps 401 to `GATEWAY_UNAUTHORIZED` and
tells the person to sign in again over a key they have never seen. It becomes a 502
and an ERROR line for whoever runs the gateway.

## Consequences

- A second lane, `http`, sized separately and never shared with the CLI lanes. A
  subprocess costs CPU and RAM on this host; an HTTP stream costs a socket.
- `usage_log` gains `engine`, `route_reason`, `provider`, `cost_usd` and
  `reasoning_tokens`; the report gains a subscription-versus-cash split and a
  by-route-reason bucket. That bucket is the measurement that matters: it says how
  much traffic aimed at the cheap engine is being caught by a constraint instead.
- Page text and email bodies now leave for a third party on GLM-routed turns. The
  provider set is selectable (`OPENROUTER_PROVIDER_SORT`, and `provider.zdr` or an
  allow-list in the same field if required); the policy is the operator's.
- Reasoning is disabled on the flash tier. These are reasoning models and their
  reasoning tokens come out of `max_tokens`; a titling call asking for 40 tokens
  would spend all of them thinking and return nothing.
- `httpx` becomes a runtime dependency.

## Notes

ConstraAP's MCP server exposes no mailbox tool — invoices, vendors, projects,
contracts, memory and pending items, but nothing that lists or reads an inbox. The
"listing the inbox" example in the extension's `ai-routing.ts` describes the
retired Electron product, not the current server. It does not weaken the rule
above, and nothing here relies on it.
