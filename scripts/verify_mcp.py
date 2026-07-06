"""Standalone verification for the MCP-connector feature (no pytest needed).

Asserts that build_argv wires the per-user MCP server correctly, keeps built-in
tools disabled, scopes tools to the one server, and stays inert when no token or
no MCP config is present.

    python3 scripts/verify_mcp.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway import config, engine
from gateway.canonical import CanonicalRequest

passed = 0


def check(label, cond):
    global passed
    assert cond, f"FAILED: {label}"
    passed += 1
    print(f"  ok: {label}")


def req(token=None):
    return CanonicalRequest(model="claude-opus-4-8", requested_model="claude-opus-4-8",
                            system="hi", mcp_token=token)


print("MCP connector verification")

# --- MCP disabled (no server URL configured) ------------------------------
config.MCP_SERVER_URL = ""
config.MCP_SERVER_NAME = "constraap"
argv = engine.build_argv(req(token="cap_x"))
check("MCP disabled → no --mcp-config even with a token", "--mcp-config" not in argv)
check("MCP disabled → built-ins still off", argv[argv.index("--tools") + 1] == "")

# --- MCP enabled ----------------------------------------------------------
config.MCP_SERVER_URL = "https://ap.constralabs.ai/mcp"
config.MCP_SERVER_NAME = "constraap"

argv = engine.build_argv(req(token=None))
check("MCP enabled but no token → no --mcp-config", "--mcp-config" not in argv)

argv = engine.build_argv(req(token="cap_secret"))
check("token present → --mcp-config added", "--mcp-config" in argv)
check("--strict-mcp-config added", "--strict-mcp-config" in argv)
check("allowedTools scoped to the one server", "mcp__constraap" in argv)
check("permission mode set for non-interactive run", "bypassPermissions" in argv)
check("built-in tools STILL disabled (isolation preserved)", argv[argv.index("--tools") + 1] == "")

cfg = json.loads(argv[argv.index("--mcp-config") + 1])
srv = cfg["mcpServers"]["constraap"]
check("config type is http", srv["type"] == "http")
check("config url is the server url", srv["url"] == "https://ap.constralabs.ai/mcp")
check("token embedded as Bearer auth", srv["headers"]["Authorization"] == "Bearer cap_secret")

# --- server name honored --------------------------------------------------
config.MCP_SERVER_NAME = "company"
argv = engine.build_argv(req(token="cap_x"))
check("allowedTools tracks MCP_SERVER_NAME", "mcp__company" in argv)

print(f"\nAll {passed} checks passed.")
sys.exit(0)
