[CmdletBinding()]
param(
  [string]$Python = "python",
  [string]$VolumeRoot = "/runpod-volume/privacy-identity-lora",
  [string]$RunId = "f305faa7-a687-4c16-a5c0-bc3b89ae2c28",
  [int]$ExpectedStep = 400,
  [int]$FallbackStep = 200,
  [string]$ReportPath = "checkpoint-rescue-report.json",
  [string]$CopySelected = "",
  [switch]$FullTensorRead
)
$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "audit_rescue_checkpoint.py"
$argsList = @(
  $scriptPath,
  "--volume-root", $VolumeRoot,
  "--run-id", $RunId,
  "--expected-step", "$ExpectedStep",
  "--fallback-step", "$FallbackStep",
  "--report", $ReportPath
)
if ($CopySelected) { $argsList += @("--copy-selected", $CopySelected) }
if ($FullTensorRead) { $argsList += "--full-tensor-read" }
& $Python @argsList
exit $LASTEXITCODE
