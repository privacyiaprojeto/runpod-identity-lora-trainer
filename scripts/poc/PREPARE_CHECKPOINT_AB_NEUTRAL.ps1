[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$BasePayloadPath,
  [Parameter(Mandatory=$true)][string]$CheckpointReportPath,
  [Parameter(Mandatory=$true)][string]$AdapterBucket,
  [Parameter(Mandatory=$true)][string]$AdapterKey,
  [Parameter(Mandatory=$true)][string]$OutputPayloadPath,
  [string]$AdapterId = "",
  [string]$SourceTrainingRunId = "f305faa7-a687-4c16-a5c0-bc3b89ae2c28"
)
$ErrorActionPreference = "Stop"

$base = Get-Content -LiteralPath $BasePayloadPath -Raw | ConvertFrom-Json
$report = Get-Content -LiteralPath $CheckpointReportPath -Raw | ConvertFrom-Json
if ($report.status -ne "CHECKPOINT_RESCUE_READY" -or -not $report.selected_checkpoint) {
  throw "O relatório não possui checkpoint íntegro selecionado."
}
$selected = $report.selected_checkpoint
if ($selected.integrity -ne "valid") { throw "Checkpoint selecionado não está validado." }

$payload = if ($base.PSObject.Properties["input"]) { $base.input } else { $base }
if (-not $payload.PSObject.Properties["adapter"]) {
  $payload | Add-Member -NotePropertyName adapter -NotePropertyValue ([PSCustomObject]@{})
}
$adapter = $payload.adapter

function Set-Property([object]$Object, [string]$Name, [object]$Value) {
  if ($Object.PSObject.Properties[$Name]) { $Object.$Name = $Value }
  else { $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value }
}

Set-Property $adapter "bucket" $AdapterBucket
Set-Property $adapter "r2_bucket" $AdapterBucket
Set-Property $adapter "key" $AdapterKey
Set-Property $adapter "r2_key" $AdapterKey
Set-Property $adapter "sha256" ([string]$selected.sha256)
Set-Property $adapter "size_bytes" ([long]$selected.size_bytes)
Set-Property $adapter "source_step" ([int]$selected.step)
Set-Property $adapter "source_training_run_id" $SourceTrainingRunId
Set-Property $adapter "private_only" $true
Set-Property $adapter "approval_allowed" $false
if ($AdapterId) { Set-Property $payload "adapter_id" $AdapterId }

# Do not invent a preview schema. The base payload must already be the approved neutral 17-frame contract.
$serialized = $base | ConvertTo-Json -Depth 100
$outputFull = [System.IO.Path]::GetFullPath($OutputPayloadPath)
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($outputFull)) | Out-Null
[System.IO.File]::WriteAllText($outputFull, $serialized, (New-Object System.Text.UTF8Encoding($false)))

[PSCustomObject]@{
  status = "AB_NEUTRAL_RESCUED_CHECKPOINT_PAYLOAD_PREPARED"
  outputPayloadPath = $outputFull
  sourceTrainingRunId = $SourceTrainingRunId
  sourceStep = [int]$selected.step
  adapterSha256 = [string]$selected.sha256
  adapterSizeBytes = [long]$selected.size_bytes
  runPodCalled = $false
  r2WriteExecuted = $false
  frontendChanged = $false
} | ConvertTo-Json -Depth 10
