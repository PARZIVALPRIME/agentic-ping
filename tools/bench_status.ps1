<#
.SYNOPSIS
  Show live progress of a background benchmark run.

.DESCRIPTION
  Reads the log produced by tools/run_bench.ps1, counts completed questions,
  shows the last few progress lines, per-pipeline accuracy so far, recent
  errors, and whether the python process is still alive.

.EXAMPLE
  powershell -File tools/bench_status.ps1
  powershell -File tools/bench_status.ps1 -Log tools/bench_hidden.txt -Out results/hidden_results.json
#>
[CmdletBinding()]
param(
  [string]$Log = "tools/bench_run.txt",
  [string]$Out = "results/public_results.json",
  [int]$Tail = 5
)

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$procs = Get-Process python -ErrorAction SilentlyContinue
if ($procs) {
  $cpu = ($procs | Measure-Object -Property CPU -Sum).Sum
  Write-Host "python running: pid(s) $($procs.Id -join ', ')  cpu=${cpu}s" -ForegroundColor Green
} else {
  Write-Host "python: not running (finished or stopped)" -ForegroundColor Yellow
}

if (-not (Test-Path $Log)) {
  Write-Host "no log at $Log" -ForegroundColor Yellow
  return
}

$lines = Get-Content $Log -ErrorAction SilentlyContinue
# Progress lines look like: "[15:39:58]   1/100 pub-001 aggregation RAG=F Graph=OK Agent=OK"
$prog = $lines | Select-String -Pattern '^\[\d{2}:\d{2}:\d{2}\]\s+\d+/\d+'
$done = ($prog | Measure-Object).Count
$total = 0
if ($prog) {
  $m = [regex]::Match($prog[0].Line, '^\[\d{2}:\d{2}:\d{2}\]\s+\d+/(\d+)')
  if ($m.Success) { $total = [int]$m.Groups[1].Value }
}

Write-Host ""
Write-Host "progress: $done / $total questions" -ForegroundColor Cyan
if ($total -gt 0 -and $done -gt 0) {
  $last = $prog[-1].Line
  if ($last -match 'took=([\d.]+)s') { Write-Host ("last question took: {0}s" -f $Matches[1]) }
  if ($last -match 'elapsed=(\d+)s') { Write-Host ("elapsed: {0}s" -f $Matches[1]) }
  if ($last -match 'eta=([\d.]+)m') { Write-Host ("eta: {0} min" -f $Matches[1]) }
}

if ($prog) {
  Write-Host ""
  Write-Host "last $Tail lines:"
  $prog | Select-Object -Last $Tail | ForEach-Object { Write-Host "  $($_.Line)" }
}

# live tally from the progress lines
if ($done -gt 0) {
  Write-Host ""
  Write-Host "tally so far (OK / answered):"
  $patterns = [ordered]@{
    "RAG"              = '(?:^|\s)RAG=(?<v>\w+)'
    "GraphRAG"         = '(?:^|\s)Graph=(?<v>\w+)'
    "Agentic GraphRAG" = '(?:^|\s)Agent=(?<v>\w+)'
  }
  foreach ($name in $patterns.Keys) {
    $ok = 0
    foreach ($l in $prog) {
      $m = [regex]::Match($l.Line, $patterns[$name])
      if ($m.Success -and $m.Groups['v'].Value -eq "OK") { $ok++ }
    }
    Write-Host ("  {0,-17} {1} / {2}" -f $name, $ok, $done)
  }
}

$errs = Get-Content "$Log.err" -ErrorAction SilentlyContinue | Select-Object -Last 8
if ($errs) {
  Write-Host ""
  Write-Host "stderr tail:" -ForegroundColor DarkYellow
  $errs | ForEach-Object { Write-Host "  $_" }
}

if (Test-Path $Out) {
  $j = Get-Content $Out -Raw | ConvertFrom-Json
  Write-Host ""
  Write-Host "results file: $((($j | Measure-Object).Count)) entries written to $Out"
}