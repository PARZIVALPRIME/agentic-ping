<#
.SYNOPSIS
  Run the public-100 sweep and then the hidden-50 sweep, one after the other.

.DESCRIPTION
  Both sweeps drive the same local GPU-backed model, so running them at the
  same time makes each one slower (and can push a provider call past its
  timeout). tools/run_bench.ps1 starts one run; this chains two, sequentially,
  each into its own log, so the hidden run cannot start while the public one is
  still holding the model.

  The runs are incremental (a finished question is on disk immediately) and
  resumable, so this is safe to launch, poll with tools/bench_status.ps1, and
  re-launch with -PublicResume / -SkipPublic after an interrupt.

  The script blocks for the whole sweep, so launch it detached: 
  Start-Process powershell -ArgumentList '-NoProfile','-File','tools/run_sweep.ps1'

.PARAMETER PublicResume
  Skip public questions already present in -PublicOut.

.PARAMETER SkipPublic / SkipHidden
  Run only one of the two.

.PARAMETER NoLlm
  Deterministic mode for both runs: no provider calls, minutes not hours.

.EXAMPLE
  powershell -File tools/run_sweep.ps1
  powershell -File tools/run_sweep.ps1 -SkipPublic -Limit 5
  powershell -File tools/run_sweep.ps1 -NoLlm -SkipHidden
#>
[CmdletBinding()]
param(
  [string]$PublicQuestions = "",
  [string]$HiddenQuestions = "",
  [string]$PublicOut = "results/public_results.json",
  [string]$PublicSummary = "results/metrics_summary.json",
  [string]$HiddenOut = "results/hidden_llm.json",
  [string]$HiddenSummary = "results/hidden_llm_summary.json",
  [string]$PublicLog = "tools/bench_public_v3.txt",
  [string]$HiddenLog = "tools/bench_hidden_v3.txt",
  [int]$Limit = 0,
  [switch]$SkipPublic,
  [switch]$SkipHidden,
  [switch]$PublicResume,
  [switch]$NoLlm
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$qdir = Get-ChildItem -Path $root -Directory -Filter "questions-*" | Select-Object -First 1
if (-not $qdir) { throw "No questions-* directory found; pass -PublicQuestions/-HiddenQuestions." }
if (-not $PublicQuestions) { $PublicQuestions = Join-Path $qdir.FullName "questions/eval_public.jsonl" }
if (-not $HiddenQuestions) { $HiddenQuestions = Join-Path $qdir.FullName "questions/eval_hidden.jsonl" }

foreach ($pair in @(@($PublicLog, $HiddenLog))) {
  foreach ($log in $pair) {
    $dir = Split-Path -Parent $log
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  }
}

# The public sweep first: it is the one the accuracy table is quoted from, and a
# crash here must not silently take the submission run down with it.
if (-not $SkipPublic) {
  $pubArgs = @("-u", "run_benchmark.py", $PublicQuestions,
               "--out", $PublicOut, "--summary", $PublicSummary)
  if ($Limit -gt 0) { $pubArgs += @("--limit", "$Limit") }
  if ($PublicResume) { $pubArgs += "--resume" }
  if ($NoLlm) { $pubArgs += "--no-llm" }
  Write-Host "public : $PublicQuestions -> $PublicOut (log $PublicLog)"
  & python @pubArgs *> $PublicLog
  Write-Host "public run exited with code $LASTEXITCODE"
}

if (-not $SkipHidden) {
  $hidArgs = @("-u", "run_benchmark.py", $HiddenQuestions,
               "--out", $HiddenOut, "--summary", $HiddenSummary)
  if ($Limit -gt 0) { $hidArgs += @("--limit", "$Limit") }
  if ($NoLlm) { $hidArgs += "--no-llm" }
  Write-Host "hidden : $HiddenQuestions -> $HiddenOut (log $HiddenLog)"
  & python @hidArgs *> $HiddenLog
  Write-Host "hidden run exited with code $LASTEXITCODE"
}

Write-Host "sweep done."
