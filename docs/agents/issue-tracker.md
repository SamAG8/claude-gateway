# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues on `SamAG8/claude-gateway`.

## Tooling

- **Local sessions with the `gh` CLI:** use `gh` for all operations (examples below).
- **Claude Code remote/web sessions:** the `gh` CLI is not available — use the GitHub MCP tools (`mcp__github__*`, e.g. `issue_read`, `issue_write`, `add_issue_comment`, `list_issues`) instead. They target the same repo.

## Conventions (gh CLI)

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments`.
- **Comment**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

`gh` infers the repo from `git remote -v` when run inside a clone.

## When a skill says "publish to the issue tracker"

Create a GitHub issue (via `gh` or `mcp__github__issue_write`).

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments` (or `mcp__github__issue_read`).
