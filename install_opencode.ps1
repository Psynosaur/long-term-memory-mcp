<#
.SYNOPSIS
    Installs the OpenCode enforcement plugin and AGENTS.md.

.DESCRIPTION
    Copies opencode/plugin/long-term-memory.ts and opencode/AGENTS.md from
    this repo into your OpenCode config directory.

    Re-run anytime the files are updated.

    Targets (Windows):
      Plugin  : %APPDATA%\opencode\plugins\long-term-memory.ts
      AGENTS  : %APPDATA%\opencode\AGENTS.md

.EXAMPLE
    .\install_opencode.ps1

.EXAMPLE
    $env:OPENCODE_CONFIG_DIR = "C:\custom\opencode"; .\install_opencode.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Resolve repo root (directory containing this script) ─────────────────────
$RepoRoot = $PSScriptRoot

$PluginSrc = Join-Path $RepoRoot 'opencode\plugin\long-term-memory.ts'
$AgentsSrc = Join-Path $RepoRoot 'opencode\AGENTS.md'

$OpenCodeDir = if ($env:OPENCODE_CONFIG_DIR) {
    $env:OPENCODE_CONFIG_DIR
} else {
    Join-Path $env:APPDATA 'opencode'
}
$PluginsDir = Join-Path $OpenCodeDir 'plugins'

# ── Validate sources exist ────────────────────────────────────────────────────
if (-not (Test-Path $PluginSrc)) {
    Write-Error "Plugin source not found: $PluginSrc"
    exit 1
}

if (-not (Test-Path $AgentsSrc)) {
    Write-Error "AGENTS.md source not found: $AgentsSrc"
    exit 1
}

# ── Create target directories if they don't exist ────────────────────────────
New-Item -ItemType Directory -Path $PluginsDir -Force | Out-Null

# ── Copy files ────────────────────────────────────────────────────────────────
$PluginDest = Join-Path $PluginsDir 'long-term-memory.ts'
Write-Host "Copying plugin  -> $PluginDest"
Copy-Item -Path $PluginSrc -Destination $PluginDest -Force

$AgentsDest = Join-Path $OpenCodeDir 'AGENTS.md'
Write-Host "Copying AGENTS  -> $AgentsDest"
Copy-Item -Path $AgentsSrc -Destination $AgentsDest -Force

Write-Host ""
Write-Host "Done. OpenCode will pick up the changes on the next session start."
