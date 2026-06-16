# Deployment & CI/CD

Push to `main` → GitHub runs the tests → if they pass, it SSHes into your server,
pulls the new commit, updates deps, and restarts the systemd service.

```
git push origin main
        │
        ▼
GitHub Actions ── test job ──────────────► python -m pytest  (Python 3.14)
   (.github/workflows/ci-cd.yml)              │  mocked engine, no claude CLI, no tokens
        │                                     ▼ green
        └── deploy job (push to main only) ── ssh ──► your server
                                                       └─ scripts/deploy.sh:
                                                            git reset --hard <sha>
                                                            pip install -r requirements.txt
                                                            sudo systemctl restart claude-gateway
                                                            curl /health   (fails the job if down)
```

Two moving parts you set up **once**: the server (so it can run the gateway) and
the GitHub ↔ server link (an SSH key + a few secrets). After that it's automatic.

> Paths below use the defaults baked into `claude-gateway.service` and `.env.example`:
> user `gateway`, repo at `/home/gateway/claude-gateway`, service `claude-gateway`.
> If you change them, edit the `.service` file (and `SERVICE_USER` / `INSTALL_DIR`
> in `.env`) to match, and set `DEPLOY_USER` / `DEPLOY_PATH` secrets accordingly.

---

## Part 1 — One-time server setup

Do this on the server. Most steps need a **sudo-capable admin user**; steps 3–6
run *as the unprivileged `gateway` user* (clone/venv/config — no root needed).
The `gateway` account intentionally **can't** `sudo`, so finish those as `gateway`,
`exit` back to your admin user, then continue. (Commands shown for Debian/Ubuntu.)

**1. System packages**

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git curl
```

**2. Create the service user the unit file expects**

```bash
sudo useradd --create-home --shell /bin/bash gateway
```

**3–6. Set up the app as the `gateway` user**

Switch into the `gateway` account **once** and run all of this *without* `sudo` —
you're already that user, and it owns everything here. (If a `sudo …` command
fails with a permission/"can't do that" error, you're still in this shell; `exit`
first.)

```bash
sudo -iu gateway             # become gateway — prompt changes to gateway@...

# 3. Install + authenticate the claude CLI. The gateway shells out to it, so it
#    must be installed and LOGGED IN for this user — CI/CD never does this.
#    Install per https://claude.ai/code, then:
claude                       # complete subscription/OAuth login
# (or, for ISOLATION_MODE=bare, skip the login and set ANTHROPIC_API_KEY in .env)

# 4. Clone the repo at the path the unit file + scripts expect
git clone https://github.com/SamAG8/claude-gateway.git /home/gateway/claude-gateway
cd /home/gateway/claude-gateway

# 5. Virtualenv + dependencies
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt

# 6. Config — create .env and set a strong API_KEY
cp .env.example .env
nano .env                    # set API_KEY (required), HOST, PORT, ISOLATION_MODE, ...

exit                         # <-- back to your admin (sudo) user for the steps below
```

`.env` is gitignored and lives only on the server — CI/CD never touches it.

**7. Install and start the systemd service** (the unit ships in the repo)

```bash
sudo cp /home/gateway/claude-gateway/claude-gateway.service /etc/systemd/system/claude-gateway.service
sudo systemctl daemon-reload
sudo systemctl enable --now claude-gateway
systemctl status claude-gateway --no-pager
curl -fsS http://127.0.0.1:8000/health     # -> {"status":"ok"}
```

**8. Let the deploy user restart the service without a password**

CI runs non-interactively, so `sudo systemctl restart` must not prompt. Allow
*only* that one command for the `gateway` user:

```bash
# Confirm the exact systemctl path first — sudoers matches the full path.
command -v systemctl        # usually /usr/bin/systemctl (or /bin/systemctl)

echo 'gateway ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart claude-gateway' \
  | sudo tee /etc/sudoers.d/claude-gateway-deploy
sudo chmod 0440 /etc/sudoers.d/claude-gateway-deploy
sudo visudo -c              # validate syntax
```

---

## Part 2 — Connect GitHub to the server

**1. Generate a dedicated deploy SSH key** (on your laptop — don't reuse a personal key)

```bash
ssh-keygen -t ed25519 -f ./claude-gateway-deploy -C "github-actions-deploy" -N ""
# -> claude-gateway-deploy       (PRIVATE → GitHub secret DEPLOY_SSH_KEY)
# -> claude-gateway-deploy.pub   (PUBLIC  → server authorized_keys)
```

**2. Authorize the public key for the deploy user**

```bash
ssh-copy-id -i ./claude-gateway-deploy.pub gateway@YOUR_SERVER
# or append claude-gateway-deploy.pub to /home/gateway/.ssh/authorized_keys by hand
```

Test it end to end from your machine:

```bash
ssh -i ./claude-gateway-deploy gateway@YOUR_SERVER \
  'sudo systemctl restart claude-gateway && echo restart-ok'
```

**3. Add the GitHub secrets**

Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Required | Example | Notes |
|---|---|---|---|
| `DEPLOY_HOST` | ✅ | `203.0.113.10` / `gw.example.com` | Server address |
| `DEPLOY_USER` | ✅ | `gateway` | Must match `authorized_keys` + the sudoers rule |
| `DEPLOY_SSH_KEY` | ✅ | *(contents of `claude-gateway-deploy`)* | The **private** key — paste the whole file, incl. the `BEGIN`/`END` lines |
| `DEPLOY_PORT` | — | `22` | Defaults to `22` if unset |
| `DEPLOY_PATH` | — | `/home/gateway/claude-gateway` | Defaults to this if unset |

Paste the private key with `cat claude-gateway-deploy` and copy everything,
including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`.

**4. (Optional) Require approval before deploy**

The workflow's deploy job uses `environment: production`. Create that environment
under **Settings → Environments → New environment → `production`** and add
*Required reviewers* to make each deploy wait for your one-click approval. You can
also store the secrets above on the environment instead of repo-wide.

---

## Part 3 — Use it

**Automatic:** merge/push to `main`. Watch **Actions → CI/CD**; the deploy step
prints the health-check result. A failing `/health` turns the job red.

**Manual deploy** (latest `main`, or a specific commit):

```bash
ssh gateway@YOUR_SERVER
cd /home/gateway/claude-gateway && ./scripts/deploy.sh           # latest origin/main
cd /home/gateway/claude-gateway && ./scripts/deploy.sh <git-sha> # a specific commit
```

**Rollback:** redeploy the last good commit — `./scripts/deploy.sh <previous-sha>`.

---

## Troubleshooting

| Symptom (in the Actions log) | Likely cause / fix |
|---|---|
| `Permission denied (publickey)` | Public key not in the deploy user's `authorized_keys`, or `DEPLOY_USER` / `DEPLOY_SSH_KEY` mismatch. |
| `sudo: a password is required` | sudoers rule missing or its path doesn't match `command -v systemctl`. |
| `Health check failed` | App crashed on boot. On the server: `journalctl -u claude-gateway -e`. Common causes: missing/invalid `API_KEY`, or the `claude` CLI not logged in for `gateway`. |
| `No such file or directory: .../scripts/deploy.sh` | Repo not cloned at `DEPLOY_PATH` (Part 1 step 4), or wrong `DEPLOY_PATH`. |
| `git reset --hard <sha>` fails | The pushed commit isn't on the server's `origin`; ensure the server clones from this GitHub repo. |

Useful on the server:

```bash
journalctl -u claude-gateway -f        # live logs
systemctl status claude-gateway        # current state
```

---

## Security notes

- Use a **dedicated** deploy key, scope sudoers to the single `restart` command,
  and keep `.env` out of git (it already is).
- The gateway binds `0.0.0.0:8000` by default — firewall the port and/or front it
  with a TLS-terminating reverse proxy (nginx/Caddy) for anything internet-facing.
- To harden further, restrict the deploy key in `authorized_keys` with
  `restrict,command="..."` so it can only run the deploy.
