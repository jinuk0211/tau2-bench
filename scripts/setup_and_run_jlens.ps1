param(
    [ValidateRange(1, 2449)]
    [int]$Start = 165,

    [ValidateRange(1, 2449)]
    [int]$Count = 1,

    [ValidateSet(
        "qwen3-8b",
        "qwen3.5-4b",
        "qwen3.5-9b-base",
        "qwen3.6-27b"
    )]
    [string]$Profile = "qwen3.5-4b",

    [ValidateRange(1, 100)]
    [int]$TopK = 10,

    [ValidateRange(1, 4096)]
    [int]$PositionChunkSize = 64,

    [string]$UserModel = "openai/qwen3:8b",

    [string]$UserApiBase = "http://127.0.0.1:11434/v1",

    [switch]$ListOnly,

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$TotalTasks = 2449
$End = $Start + $Count - 1

if ($End -gt $TotalTasks) {
    throw "Requested range $Start..$End exceeds the $TotalTasks-task catalog."
}

function Invoke-Uv {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$UvArguments
    )

    & uv @UvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv command failed with exit code ${LASTEXITCODE}: uv $UvArguments"
    }
}

Set-Location -LiteralPath $ProjectDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[setup] Installing uv..."
    Invoke-RestMethod "https://astral.sh/uv/install.ps1" | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not available. Reopen PowerShell and run this script again."
}

Write-Host "[setup] Installing Python 3.12 and project dependencies..."
Invoke-Uv python install 3.12
Invoke-Uv sync --python 3.12 --extra jlens --extra dev

$RunLabel = "{0:D4}-{1:D4}" -f $Start, $End
$TraceDir = Join-Path "data\jlens_traces" $RunLabel
$AnalysisDir = Join-Path "data\jlens_analysis" $RunLabel

Write-Host ""
Write-Host "Combined task catalog:"
Write-Host "  1..50     airline"
Write-Host "  51..164   retail"
Write-Host "  165..2449 telecom"
Write-Host "Selection: $Start..$End ($Count task(s))"
Write-Host "Profile:   $Profile"
Write-Host ""

Invoke-Uv run python scripts/run_jlens_range.py `
    --start $Start `
    --count $Count `
    --profile $Profile `
    --list-only

if ($ListOnly) {
    Write-Host "[done] Selection listed; no model was loaded."
    exit 0
}

# Airline/retail are conversational and need an OpenAI-compatible user model.
if ($Start -le 164) {
    $ModelsUrl = "$($UserApiBase.TrimEnd('/'))/models"
    try {
        Invoke-RestMethod -Uri $ModelsUrl -Method Get -TimeoutSec 5 | Out-Null
    }
    catch {
        throw (
            "The selected range includes airline/retail, but the user-model " +
            "endpoint is unavailable at $ModelsUrl. Start the server or pass " +
            "-UserModel and -UserApiBase."
        )
    }
}

Write-Host "[run] Capturing exact tau2 token traces..."
Invoke-Uv run python scripts/run_jlens_range.py `
    --start $Start `
    --count $Count `
    --profile $Profile `
    --user-model $UserModel `
    --user-api-base $UserApiBase `
    --dtype bfloat16 `
    --max-new-tokens 256 `
    --max-concurrency 1 `
    --auto-resume

if (-not (Test-Path -LiteralPath $TraceDir)) {
    throw "Trace directory was not created: $TraceDir"
}

Write-Host "[analyze] Reading every token position and fitted layer..."
Invoke-Uv run tau2 jlens $TraceDir `
    --profile $Profile `
    --output-dir $AnalysisDir `
    --top-k $TopK `
    --layer-stride 1 `
    --position-chunk-size $PositionChunkSize

$ResolvedAnalysisDir = (Resolve-Path -LiteralPath $AnalysisDir).Path
$IndexFile = Join-Path $ResolvedAnalysisDir "index.html"
if (-not (Test-Path -LiteralPath $IndexFile)) {
    throw "Analysis index was not created: $IndexFile"
}

$Port = 8000
while (
    Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue
) {
    $Port++
}

$UvExecutable = (Get-Command uv).Source
Start-Process `
    -FilePath $UvExecutable `
    -ArgumentList @("run", "python", "-m", "http.server", "$Port") `
    -WorkingDirectory $ResolvedAnalysisDir `
    -WindowStyle Hidden

Start-Sleep -Seconds 2
$ResultUrl = "http://localhost:$Port/"

Write-Host ""
Write-Host "J-Lens analysis complete"
Write-Host "  profile: $Profile"
Write-Host "  range:   $Start..$End / $TotalTasks"
Write-Host "  output:  $ResolvedAnalysisDir"
Write-Host "  URL:     $ResultUrl"

if (-not $NoBrowser) {
    Start-Process $ResultUrl
}
