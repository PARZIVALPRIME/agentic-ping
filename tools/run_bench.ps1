<#
.SYNOPSIS
  Run the benchmark in the background so the terminal is never blocked.

.DESCRIPTION
  The benchmark takes ~10-20 minutes for 100 questions (LLM rate limits). Run
  from a blocking terminal it looks "stuck". This launches python detached with
  unbuffered output, streams to a log file, and returns immediately.

.PARAMETER Questions
  Path to a questions jsonl. Defaults to the public eval set.

.PARAMETER Out
  Results json path. Default results/public_results.json.

.PARAMETER Pipelines
  Comma filter, e.g. 'agentic' or 'rag,graphrag'. Default: all three.

.PARAMETER Limit
  0 = all questions.

.PARAMETER Resume
  Skip questions already present in -Out (safe to re-run after an interrupt).

.PARAMETER NoLlm
  Deterministic mode: make no provider calls at all. Always finishes in ~1
  minute regardless of quota; useful as the reproducible baseline.

.PARAMETER Log
  Log file. Default tools/bench_run.txt.

.PARAMETER Follow
  Tail the log until the process exits (blocking, but shows progress live).

.EXAMPLE
  powershell -File tools/run_bench.ps1 -Limit 10
  powershell -File tools/run_bench.ps1 -Resume -Follow
  powershell -File tools/run_bench.ps1 -NoLlm -Out results/deterministic_results.json
  powershell -File tools/run_bench.ps1 -Questions questions-*/questions/eval_hidden.jsonl `
      -Out results/hidden_results.json -Log tools/bench_hidden.txt
#>
[CmdletBinding()]
param(
  [string]$Questions = "",
  [string]$Out = "results/public_results.json",
  [string]$Summary = "",
  [string]$Pipelines = "",
  [int]$Limit = 0,
  [int]$Offset = 0,
  [switch]$Resume,
  [switch]$NoLlm,
  [switch]$NoLlmJudge,
  [string]$Log = "tools/bench_run.txt",
  [switch]$Follow
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not $Questions) {
  $q = Get-ChildItem -Path $root -Directory -Filter "questions-*" |
        Select-Object -First 1
  if (-not $q) { throw "No questions-* directory found; pass -Questions." }
  $Questions = Join-Path $q.FullName "questions/eval_public.jsonl"
}
if (-not $Summary) { $Summary = Join-Path "results" ((Split-Path -Leaf $Out) -replace '\.json$', '_summary.json') }

$logDir = Split-Path -Parent $Log
if ($logDir -and -not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory -Path "results" -Force | Out-Null }

# Build the argument list (python -u => unbuffered, so the log is live).
$argList = @("-u", "run_benchmark.py", $Questions, "--out", $Out, "--summary", $Summary)
if ($Pipelines) { $argList += @("--pipelines", $Pipelines) }
if ($Limit -gt 0) { $argList += @("--limit", "$Limit") }
if ($Offset -gt 0) { $argList += @("--offset", "$Offset") }
if ($Resume) { $argList += "--resume" }
if ($NoLlm) { $argList += "--no-llm" }
if ($NoLlmJudge) { $argList += "--no-llm-judge" }

Write-Host "questions : $Questions"
Write-Host "results   : $Out"
Write-Host "summary   : $Summary"
Write-Host "log       : $Log"
Write-Host "pipelines : $(if ($Pipelines) { $Pipelines } else { 'all' })"
if ($Resume) { Write-Host "resume    : on" }
if ($NoLlm) { Write-Host "mode      : deterministic (no LLM calls)" }
Write-Host ""

$proc = Start-Process -FilePath "python" -ArgumentList $argList `
  -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
  -NoNewWindow -PassThru

Write-Host "started pid $($proc.Id) -- progress is appended to $Log"
Write-Host "check status any time with:  powershell -File tools/bench_status.ps1 -Log $Log"

if ($Follow) {
  Write-Host ""
  Write-Host "--- following (Ctrl+C detaches; the run keeps going) ---" -ForegroundColor DarkGray
  while (-not $proc.HasExited) {
    Start-Sleep -Seconds 5
    if (Test-Path $Log) {
      Get-Content $Log -Tail 3 | ForEach-Object { Write-Host "  $_" }
    }
  }
  Write-Host ""
  Write-Host "process exited with code $($proc.ExitCode)"
  if (Test-Path "$Log.err") {
    $err = Get-Content "$Log.err" -ErrorAction SilentlyContinue | Select-Object -Last 15
    if ($err) { Write-Host "--- stderr (tail) ---" -ForegroundColor DarkYellow; $err }
  }
} else {
  Write-Host ""
  Write-Host "returning immediately (detached). Use -Follow to stream progress."
}