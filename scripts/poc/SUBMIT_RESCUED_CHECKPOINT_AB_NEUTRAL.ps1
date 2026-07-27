[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$EnvFile,
  [Parameter(Mandatory=$true)][string]$PayloadPath,
  [Parameter(Mandatory=$true)][string]$Confirm,
  [int]$TimeoutSeconds = 30
)
$ErrorActionPreference = "Stop"
$required = "SUBMETER CHECKPOINT RESGATADO AO AB NEUTRO 17 FRAMES"
if ($Confirm -ne $required) { throw "Confirmação inválida. Exigido: $required" }

function Get-DotEnvValue([string]$Name) {
  $line = Get-Content -LiteralPath $EnvFile | Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } | Select-Object -Last 1
  if (-not $line) { return "" }
  (($line -split "=", 2)[1]).Trim().Trim('"').Trim("'")
}

$apiKey = Get-DotEnvValue "RUNPOD_API_KEY"
$endpointId = Get-DotEnvValue "IDENTITY_LORA_TRAINER_ENDPOINT_ID"
if (-not $apiKey -or -not $endpointId) { throw "RUNPOD_API_KEY ou IDENTITY_LORA_TRAINER_ENDPOINT_ID ausente." }
$document = Get-Content -LiteralPath $PayloadPath -Raw | ConvertFrom-Json
$body = if ($document.PSObject.Properties["input"]) { $document } else { [PSCustomObject]@{ input = $document } }

$response = Invoke-RestMethod -Method POST `
  -Uri "https://api.runpod.ai/v2/$endpointId/run" `
  -Headers @{ Authorization = "Bearer $apiKey"; "Content-Type" = "application/json" } `
  -Body ($body | ConvertTo-Json -Depth 100 -Compress) `
  -TimeoutSec $TimeoutSeconds

[PSCustomObject]@{
  status = "AB_NEUTRAL_RESCUED_CHECKPOINT_SUBMITTED"
  endpointId = $endpointId
  providerJobId = $response.id
  providerStatus = $response.status
} | ConvertTo-Json -Depth 10
