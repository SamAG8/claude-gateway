# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase. This repo is **single-context**.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the project glossary (Adapter, Canonical Request/Event, Engine, Isolation Mode, Model Map, etc.).
- **`CLAUDE.md`** at the repo root — architecture overview, commands, and critical invariants.
- **`docs/adr/`** — Architecture Decision Records, if present. This repo has none yet; `/grill-with-docs` creates them lazily when a decision actually crystallises. **Proceed silently if absent.**

## File structure (single-context)

```
/
├── CONTEXT.md          ← domain glossary
├── CLAUDE.md           ← architecture + commands + invariants
├── docs/
│   ├── adr/            ← (created lazily by /grill-with-docs)
│   └── agents/         ← this config
├── gateway/            ← the package (engine, canonical, adapters/*, models, errors, content)
└── tests/
```

## Use the glossary's vocabulary

When your output names a domain concept (an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md` — e.g. "Adapter", "Canonical Event", "Engine", "Isolation Mode", "Model Map". Don't drift to synonyms the glossary avoids.

If a concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider), or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding it.
