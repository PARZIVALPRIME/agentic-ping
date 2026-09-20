<#
.SYNOPSIS
  Download a large file fast on a connection that throttles single streams.

.DESCRIPTION
  Some routes (ollama.com, GitHub releases, huggingface.co) cap a single TCP
  stream at a few tens of KB/s from this network, while the same host happily
  serves several streams at once - measured here as 43 KB/s on one connection
  versus ~190 KB/s on two. Downloading a 1.46 GiB installer one stream at a
  time would take most of a day; splitting it into ranges and running them
  concurrently turns that into tens of minutes.

  Each range is fetched by its own curl process and retried until its length is
  correct, so an interrupted run can simply be started again: parts that are
  already complete are left alone.

.PARAMETER Url
  File to download. Must support HTTP range requests (206 responses).

.PARAMETER Out
  Final file path. Put it on a drive with room for two copies while it runs.

.PARAMETER Connections
  Number of ranges the file is split into. More parts means finer resume
  granularity (a part that dies is re-fetched whole), so this is about
  recoverability rather than speed.

.PARAMETER Concurrent
  How many ranges to fetch at the same time. Measured on this route: one stream
  bursts to ~200 KB/s and then settles near 25-50 KB/s, and a dozen concurrent
  streams starve each other down to zero, so a small number is both faster and
  more reliable than either extreme. 4 is the default for that reason.

.EXAMPLE
  powershell -File tools/fetch_ollama.ps1
  powershell -File tools/fetch_ollama.ps1 -Connections 24 -Concurrent 4
#>
[CmdletBinding()]
param(
  [string]$Url = "https://ollama.com/download/OllamaSetup.exe",
  [string]$Out = "D:\OllamaSetup.exe",
  [int]$Connections = 16,
  [int]$Concurrent = 4,
  [string]$PartDir = "D:\ollama_parts",
  [int]$MaxRounds = 200
)

$ErrorActionPreference = "Stop"

# ── size first: a one-byte range tells us the total (HEAD is refused here) ──
$headers = & curl.exe -sL -r 0-0 -D - -o NUL --max-time 60 $Url
$range = ($headers | Select-String -Pattern 'Content-Range:\s*bytes\s+\d+-(\d+)/(\d+)').Matches
if (-not $range.Count) { throw "server did not answer a range request; cannot chunk this URL" }
$size = [int64]$range[0].Groups[2].Value
$chunk = [int64][math]::Ceiling($size / $Connections)
Write-Host "url        : $Url"
Write-Host "size       : $size bytes ($([math]::Round($size/1GB,2)) GiB)"
Write-Host "connections: $Connections  (~$([math]::Round($chunk/1MB)) MB each)"
Write-Host "parts      : $PartDir"
Write-Host "output     : $Out"
Write-Host ""

if (-not (Test-Path $PartDir)) { New-Item -ItemType Directory -Path $PartDir -Force | Out-Null }

function Show-Progress([int64]$size, [int64]$elapsedSeconds, $ranges, $PartDir) {
  $bytes = 0; $done = 0
  foreach ($r in $ranges) {
    $path = Join-Path $PartDir ("part_{0:d3}" -f $r.Index)
    if (Test-Path $path) {
      $len = (Get-Item $path).Length
      $bytes += [math]::Min($len, $r.Want)
      if ($len -eq $r.Want) { $done++ }
    }
  }
  $rate = if ($elapsedSeconds -gt 0) { $bytes / $elapsedSeconds } else { 0 }
  Write-Host ("        {0}/{1} parts, {2:N0}/{3:N0} MB, {4:N0} KB/s avg, {5:N1} min elapsed" -f `
    $done, $ranges.Count, ($bytes/1MB), ($size/1MB), ($rate/1KB), ($elapsedSeconds/60))
}

function Get-PartRanges([int64]$total, [int64]$step, [int]$count) {
  $ranges = @()
  for ($i = 0; $i -lt $count; $i++) {
    $start = $i * $step
    if ($start -ge $total) { break }
    $end = [math]::Min($start + $step - 1, $total - 1)
    $ranges += [pscustomobject]@{ Index = $i; Start = $start; End = $end
                                  Want = $end - $start + 1 }
  }
  $ranges
}

$ranges = Get-PartRanges $size $chunk $Connections

$sw = [System.Diagnostics.Stopwatch]::StartNew()
for ($round = 1; $round -le $MaxRounds; $round++) {
  $todo = @()
  foreach ($r in $ranges) {
    $path = Join-Path $PartDir ("part_{0:d3}" -f $r.Index)
    $have = if (Test-Path $path) { (Get-Item $path).Length } else { 0 }
    if ($have -ne $r.Want) { $todo += $r }
  }
  if (-not $todo.Count) { break }

  Write-Host ("round {0}: {1} part(s) to fetch, {2} at a time" -f `
    $round, $todo.Count, $Concurrent)
  for ($i = 0; $i -lt $todo.Count; $i += $Concurrent) {
    $batch = $todo[$i..([math]::Min($i + $Concurrent - 1, $todo.Count - 1))]
    $procs = @()
    foreach ($r in $batch) {
      $path = Join-Path $PartDir ("part_{0:d3}" -f $r.Index)
      # A part that was cut short is re-fetched whole: the range is small, and
      # curl's --retry already covers transient resets within one attempt.
      if (Test-Path $path) { Remove-Item $path -Force }
      $args = @("-sL", "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
                "--max-time", "1800", "-r", "$($r.Start)-$($r.End)",
                "-o", $path, $Url)
      $procs += Start-Process -FilePath "curl.exe" -ArgumentList $args -NoNewWindow -PassThru
    }
    $procs | Wait-Process
    Show-Progress $size $sw.Elapsed.TotalSeconds $ranges $PartDir
  }
}

# ── verify every part, then concatenate in order ───────────────────────────
$bad = @()
foreach ($r in $ranges) {
  $path = Join-Path $PartDir ("part_{0:d3}" -f $r.Index)
  if (-not (Test-Path $path) -or (Get-Item $path).Length -ne $r.Want) { $bad += $r.Index }
}
if ($bad.Count) {
  throw "incomplete parts after $MaxRounds rounds: $($bad -join ', ') - re-run this script to continue"
}

Write-Host ""
Write-Host "all parts complete; concatenating..."
$ordered = $ranges | ForEach-Object { Join-Path $PartDir ("part_{0:d3}" -f $_.Index) }
if (Test-Path $Out) { Remove-Item $Out -Force }
& cmd.exe /c ("copy /b " + ($ordered -join "+") + " `"$Out`"") | Out-Null

$final = (Get-Item $Out).Length
Write-Host ("wrote      : {0} ({1:N0} bytes)" -f $Out, $final)
if ($final -ne $size) { throw "size mismatch: expected $size, got $final" }
Write-Host ("done in    : {0:N1} minutes, {1:N0} B/s average" -f `
  $sw.Elapsed.TotalMinutes, ($size / $sw.Elapsed.TotalSeconds))
Write-Host "you can delete $PartDir now"
