#!/usr/bin/env bash
# install_kafka.sh
#
# Installs Kafka dependencies, sets up the .env config, and ensures the
# tray app's autostart plist includes --kafka-sharing for next launch.
#
# If the tray app is already running, it is NOT restarted — use the
# "Kafka Sharing" toggle in the tray menu to enable it live.
#
# Run from the project root:
#   bash install_kafka.sh

set -euo pipefail

# ── Resolve repo root ─────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PLIST_PATH="$HOME/Library/LaunchAgents/com.longtermmemorymcp.tray.plist"
LOCK_FILE="$REPO_ROOT/data/logs/tray_app.lock"

echo "╔══════════════════════════════════════════════════╗"
echo "║    LTM Kafka Memory Sharing — Setup              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Detect Python ─────────────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
    PIP="$REPO_ROOT/.venv/bin/pip"
    echo "Using venv Python: $PY"
elif command -v python3 &>/dev/null; then
    PY="python3"
    PIP="python3 -m pip"
    echo "Using system Python: $PY"
else
    echo "ERROR: No Python found. Activate your venv or install Python 3.12+." >&2
    exit 1
fi

echo ""

# ── Step 1: Install confluent-kafka ───────────────────────────────────────────
echo "Step 1: Installing confluent-kafka..."
$PIP install "confluent-kafka>=2.3.0"
echo "  ✓ confluent-kafka installed"
echo ""

# ── Step 2: Set up .env ──────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.env" ]]; then
    echo "Step 2: .env already exists — skipping copy."
    echo "  → Review it and fill in your KAFKA_* credentials."
else
    echo "Step 2: Creating .env from .env.example..."
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo "  ✓ .env created — edit it with your Kafka credentials."
fi
echo ""

# ── Step 3: Show identity ────────────────────────────────────────────────────
IDENTITY_FILE="$REPO_ROOT/data/identity.json"
if [[ -f "$IDENTITY_FILE" ]]; then
    echo "Step 3: Your node identity (from data/identity.json):"
    echo ""
    $PY -c "
import json
with open('$IDENTITY_FILE') as f:
    d = json.load(f)
print(f'  Username:  {d["username"]}')
print(f'  Node UUID: {d["node_uuid"]}')
print()
print(f'  Add this to ALLOWED_KAFKA_USERS in .env:')
print(f'  ALLOWED_KAFKA_USERS="{d["username"]}:{d["node_uuid"]}"')
"
else
    echo "Step 3: No identity.json yet — it will be created on first server start."
    echo "  After the first run, check data/identity.json for your username:node_uuid"
    echo "  and add it to ALLOWED_KAFKA_USERS in .env."
fi

echo ""

# ── Step 4: Patch autostart plist with --kafka-sharing ────────────────────────
_tray_is_running() {
    if [[ -f "$LOCK_FILE" ]]; then
        local pid
        pid="$(cat "$LOCK_FILE" 2>/dev/null)" || return 1
        kill -0 "$pid" 2>/dev/null && return 0
    fi
    return 1
}

if [[ "$(uname)" == "Darwin" && -f "$PLIST_PATH" ]]; then
    if grep -q "kafka-sharing" "$PLIST_PATH"; then
        echo "Step 4: Autostart plist already has --kafka-sharing ✓"
    else
        echo "Step 4: Patching autostart plist to include --kafka-sharing..."
        # Insert --kafka-sharing before the closing </array> of ProgramArguments
        sed -i '' 's|</array>|        <string>--kafka-sharing</string>    </array>|' "$PLIST_PATH"
        echo "  ✓ Plist updated — --kafka-sharing will be active on next launch."
    fi
else
    echo "Step 4: No autostart plist found — skipping."
fi

echo ""
echo "──────────────────────────────────────────────────"
echo ""

# ── Step 5: Tell the user what to do next ─────────────────────────────────────
if _tray_is_running; then
    TRAY_PID="$(cat "$LOCK_FILE" 2>/dev/null)"
    echo "  Tray app is running (pid $TRAY_PID)."
    echo ""
    echo "  → To enable Kafka NOW:  click the tray icon → toggle 'Kafka Sharing'"
    echo "  → On next restart it will start with --kafka-sharing automatically."
    echo ""
    echo "  Remember to edit .env with your KAFKA_* broker credentials first!"
else
    echo "  Tray app is not running."
    echo ""
    echo "  To start with Kafka sharing:"
    echo ""
    # Build the launch command from the plist if available, otherwise use defaults
    if [[ "$(uname)" == "Darwin" && -f "$PLIST_PATH" ]]; then
        echo "    launchctl unload '$PLIST_PATH' 2>/dev/null"
        echo "    launchctl load '$PLIST_PATH'"
        echo ""
        echo "  Or manually:"
    fi
    echo "    $PY tray_app.py --auto-start --kafka-sharing"
    echo ""
    echo "  Remember to edit .env with your KAFKA_* broker credentials first!"
fi

echo ""
echo "Done."
