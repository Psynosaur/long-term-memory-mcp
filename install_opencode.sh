#!/usr/bin/env bash
# install_opencode.sh
#
# Installs the OpenCode enforcement plugin and AGENTS.md from this repo
# into your OpenCode config directory.
#
# Run this script once to install, and again whenever the files are updated:
#   bash install_opencode.sh
#
# Targets (macOS / Linux):
#   Plugin  : ~/.config/opencode/plugins/long-term-memory.ts
#   AGENTS  : ~/.config/opencode/AGENTS.md
#   SDK dep : @opencode-ai/plugin  (installed via bun in ~/.config/opencode)
#
# Windows: run the equivalent copy commands manually — see README.md.

set -euo pipefail

# ── Resolve repo root (directory containing this script) ──────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PLUGIN_SRC="$REPO_ROOT/opencode/plugin/long-term-memory.ts"
AGENTS_SRC="$REPO_ROOT/opencode/AGENTS.md"

OPENCODE_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"
PLUGINS_DIR="$OPENCODE_DIR/plugins"

# ── Validate sources exist ─────────────────────────────────────────────────────
if [[ ! -f "$PLUGIN_SRC" ]]; then
  echo "ERROR: Plugin source not found: $PLUGIN_SRC" >&2
  exit 1
fi

if [[ ! -f "$AGENTS_SRC" ]]; then
  echo "ERROR: AGENTS.md source not found: $AGENTS_SRC" >&2
  exit 1
fi

# ── Create target directories if they don't exist ─────────────────────────────
mkdir -p "$PLUGINS_DIR"

# ── Copy files ────────────────────────────────────────────────────────────────
echo "Copying plugin  → $PLUGINS_DIR/long-term-memory.ts"
cp "$PLUGIN_SRC" "$PLUGINS_DIR/long-term-memory.ts"

echo "Copying AGENTS  → $OPENCODE_DIR/AGENTS.md"
cp "$AGENTS_SRC" "$OPENCODE_DIR/AGENTS.md"

# ── Install / update the OpenCode plugin SDK (requires bun) ───────────────────
if command -v bun &>/dev/null; then
  echo "Installing @opencode-ai/plugin SDK in $OPENCODE_DIR ..."
  bun add @opencode-ai/plugin --cwd "$OPENCODE_DIR" 2>&1
else
  echo "WARNING: bun not found — skipping SDK install."
  echo "         Run manually: cd $OPENCODE_DIR && bun add @opencode-ai/plugin"
fi

echo ""
echo "Done. OpenCode will pick up the changes on the next session start."
