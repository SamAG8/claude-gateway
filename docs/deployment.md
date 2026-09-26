# Deployment & CI/CD

## Production — how it ships

The gateway runs as the Docker container `claude-gateway` on the production
host — **`155.138.144.164`, hostname `constralabs`** — behind that host's nginx
at `https://ap.constralabs.ai/llm-gateway/`. Its files live in
`/var/www/claude-gateway` (`app/` is the checkout, `docker/` holds the compose
files and the secrets the container mounts at `/run/secrets/app.env`), and the
host's own authority documents are in `/opt/constralabs/docs/`: `OPERATIONS.md`,
`ARCHITECTURE.md`, `AGENTS.md`, and `stacks/claude-gateway.md` for this stack.

**There is one production host and it is 155.138.144.164** (verified
2026-09-22). It is the shared LLM relay for ConstraAP, ConstraBid staging, stbid
and the Nimbus clients, and this repo is its deployment source.

```
git push origin main
        │
        ▼
GitHub Actions ── test ──────────────────► python -m pytest   (GitHub-hosted, Python 3.14)
   (.github/workflows/ci-cd.yml)             │  mocked engine, no claude CLI, no tokens
        │                                    ▼ green
        └── deploy (push to main only) ── self-hosted runner `constralabs-claude-gateway`
                                          ON the production host, user `deploy`
                                            └─ REVISION=<sha7> /opt/constralabs/bin/deploy-app claude-gateway <sha>
                                                 tag rollback image → git reset --hard <sha> in
                                                 /var/www/claude-gateway/app → docker compose build
                                                 (Dockerfile writes /srv/gateway/REVISION from the
                                                 build-arg) → up -d → /health gate over the Docker
                                                 network → auto-rollback on failure → public probe
                                               then: assert /health.revision == <sha7>, and that no
                                               container on the host went unhealthy
```

The `environment: production` on the deploy job is what creates the entries at
<https://github.com/SamAG8/claude-gateway/deployments> and marks them
success/failure. Verify a deploy from anywhere, no SSH needed:

```bash
curl -s https://ap.constralabs.ai/llm-gateway/health
# {"status":"ok","revision":"<short sha>","mcp":true,"pat_auth":true,"openrouter":true}
```

`revision` must equal `git rev-parse --short origin/main`; `unknown` means the
image was built without the `REVISION` build-arg (a manual `deploy-app.sh` run
without `REVISION=…` in its environment does exactly that — honest, but fix it by
re-running with the variable set).

What lives where:

| | |
|---|---|
| Runner | `/home/deploy/actions-runner-claude-gateway`, systemd unit `actions.runner.SamAG8-claude-gateway.constralabs-claude-gateway.service`, `Restart=always` drop-in |
| Labels | `self-hosted, linux, x64, constralabs-claude-gateway` — `runs-on` in the workflow must match |
| Stack | `/var/www/claude-gateway/{app,docker}` — `app/` is a clone of this repo, `docker/` the compose file, Dockerfile and secrets |
| Host runbooks | `/opt/constralabs/docs/OPERATIONS.md` (CI/CD, runners, deploy/rollback), `/opt/constralabs/docs/stacks/claude-gateway.md` (everything stack-specific), `/opt/constralabs/docs/AGENTS.md` (the rules) |

Things the workflow deliberately does **not** do: check the repo out on the
runner (`deploy-app.sh` owns the clone and resets it to the pushed SHA itself),
touch secrets (`docker/secrets/claude-gateway.env`, mounted read-only into the
container), or run on `pull_request` (the repo is public; a self-hosted runner
must never execute a fork's code).

Manual redeploy of `main`: Actions → CI/CD → Run workflow, or on the host
`REVISION=$(git -C /var/www/claude-gateway/app rev-parse --short origin/main) sudo -u deploy -E /var/www/bin/deploy-app.sh claude-gateway`.

Registering the runner again if it is ever removed follows the recipe in the
host's `OPERATIONS.md` ("Rebuilding the runner"), substituting this repo's URL,
the directory above and the labels above. It needs **admin** on this repo for
the registration token.

---

## Rollback and logs

Before it builds, `deploy-app` tags the running image
`claude-gateway:rollback-<ts>`. When the new build fails its health gate it puts
that image back, recreates the container, re-probes it and fails the job, so
production keeps serving the previous build. To roll back a build that passed
the gate but is wrong, run the line `deploy-app` prints at the end of every
deploy, on the host:

```bash
docker tag claude-gateway:rollback-<ts> claude-gateway:current
cd /var/www/claude-gateway/docker && docker compose up -d --force-recreate   # never `down`
curl -s https://ap.constralabs.ai/llm-gateway/health   # reports the older revision
docker compose logs -f claude-gateway                   # logs, from the same directory
```

This repo ships no other deploy path. To run a gateway anywhere else, start it
as README → *Setup* shows and put a TLS-terminating reverse proxy in front: it
binds `0.0.0.0:8000` by default.
