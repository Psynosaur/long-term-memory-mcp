# install_kafka.ps1
#
# Installs Kafka dependencies, sets up .env config, and patches the
# autostart registry entry to include --kafka-sharing for next launch.
#
# If the tray app is already running, it is NOT restarted — use the
# "Kafka Sharing" toggle in the tray menu to enable it live.
#
# Run from the project root:
#   .\install_kafka.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$RegName = "LongTermMemoryMCP"
$LockFile = Join-Path $RepoRoot "data\logs\tray_app.lock"

Write-Host ""
Write-Host "  LTM Kafka Memory Sharing - Setup" -ForegroundColor Cyan
Write-Host "  =================================" -ForegroundColor Cyan
Write-Host ""

# -- Detect Python --
$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $RepoRoot ".venv\Scripts\pip.exe"

if (Test-Path $VenvPy) {
    $Py = $VenvPy
    $Pip = $VenvPip
    Write-Host "Using venv Python: $Py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Py = "python"
    $Pip = "python -m pip"
    Write-Host "Using system Python: $Py"
} else {
    Write-Host "ERROR: No Python found. Activate your venv or install Python 3.12+." -ForegroundColor Red
    exit 1
}

Write-Host ""

# -- Step 1: Install confluent-kafka --
Write-Host "Step 1: Installing confluent-kafka..." -ForegroundColor Yellow
& $Pip install "confluent-kafka>=2.3.0"
Write-Host "  OK confluent-kafka installed" -ForegroundColor Green
Write-Host ""

# -- Step 2: Set up .env --
$EnvFile = Join-Path $RepoRoot ".env"
$EnvExample = Join-Path $RepoRoot ".env.example"

if (Test-Path $EnvFile) {
    Write-Host "Step 2: .env already exists - skipping copy." -ForegroundColor Yellow
    Write-Host "  -> Review it and fill in your KAFKA_* credentials."
} else {
    Write-Host "Step 2: Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item $EnvExample $EnvFile
    Write-Host "  OK .env created - edit it with your Kafka credentials." -ForegroundColor Green
}
Write-Host ""

# -- Step 3: Show identity --
$IdentityFile = Join-Path $RepoRoot "data\identity.json"

if (Test-Path $IdentityFile) {
    Write-Host "Step 3: Your node identity (from data\identity.json):" -ForegroundColor Yellow
    Write-Host ""
    $identity = Get-Content $IdentityFile | ConvertFrom-Json
    Write-Host "  Username:  $($identity.username)"
    Write-Host "  Node UUID: $($identity.node_uuid)"
    Write-Host ""
    Write-Host "  Add this to ALLOWED_KAFKA_USERS in .env:" -ForegroundColor Cyan
    Write-Host "  ALLOWED_KAFKA_USERS=`"$($identity.username):$($identity.node_uuid)`""
} else {
    Write-Host "Step 3: No identity.json yet - it will be created on first server start." -ForegroundColor Yellow
    Write-Host "  After first run, check data\identity.json for username:node_uuid"
    Write-Host "  and add it to ALLOWED_KAFKA_USERS in .env."
}
Write-Host ""

# -- Step 4: Patch autostart registry with --kafka-sharing --
try {
    $regValue = (Get-ItemProperty -Path $RegPath -Name $RegName -ErrorAction Stop).$RegName
    if ($regValue -match "kafka-sharing") {
        Write-Host "Step 4: Autostart registry already has --kafka-sharing" -ForegroundColor Green
    } else {
        Write-Host "Step 4: Patching autostart registry to include --kafka-sharing..." -ForegroundColor Yellow
        $newValue = "$regValue --kafka-sharing"
        Set-ItemProperty -Path $RegPath -Name $RegName -Value $newValue
        Write-Host "  OK Registry updated - --kafka-sharing will be active on next launch." -ForegroundColor Green
    }
} catch {
    Write-Host "Step 4: No autostart registry entry found - skipping." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  ------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

# -- Step 5: Tell the user what to do next --
$trayRunning = $false
if (Test-Path $LockFile) {
    try {
        $trayPid = [int](Get-Content $LockFile -ErrorAction Stop)
        $proc = Get-Process -Id $trayPid -ErrorAction Stop
        $trayRunning = $true
    } catch {}
}

if ($trayRunning) {
    Write-Host "  Tray app is running (pid $trayPid)." -ForegroundColor Green
    Write-Host ""
    Write-Host "  -> To enable Kafka NOW:  click the tray icon -> toggle 'Kafka Sharing'" -ForegroundColor Cyan
    Write-Host "  -> On next restart it will start with --kafka-sharing automatically."
    Write-Host ""
    Write-Host "  Remember to edit .env with your KAFKA_* broker credentials first!" -ForegroundColor Yellow
} else {
    Write-Host "  Tray app is not running."
    Write-Host ""
    Write-Host "  To start with Kafka sharing:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    $Py tray_app.py --auto-start --kafka-sharing" -ForegroundColor White
    Write-Host ""
    Write-Host "  Remember to edit .env with your KAFKA_* broker credentials first!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
