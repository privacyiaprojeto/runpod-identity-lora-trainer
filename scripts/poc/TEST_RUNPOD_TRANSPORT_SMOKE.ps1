[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$EnvFile = "C:\Users\Usuario\Desktop\IAPrivacy\backend\.env",

    [Parameter(Mandatory = $false)]
    [ValidateRange(15, 600)]
    [int]$TimeoutSeconds = 120,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 30)]
    [int]$PollSeconds = 2,

    [Parameter(Mandatory = $true)]
    [string]$Confirm,

    [Parameter(Mandatory = $false)]
    [switch]$CancelOnTimeout
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExpectedConfirmation = 'TESTAR TRANSPORTE RUNPOD SEM TREINO D3.6H10'
$ContractVersion = 'privacy-identity-lora-transport-smoke-v1'

if ($Confirm -ne $ExpectedConfirmation) {
    throw "Confirmação inválida. Use exatamente: $ExpectedConfirmation"
}

if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw "Arquivo .env não encontrado: $EnvFile"
}

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $line = Get-Content -LiteralPath $EnvFile |
        Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
        Select-Object -Last 1

    if (-not $line) {
        throw "Variável $Name não encontrada em $EnvFile"
    }

    return (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
}

function Get-OptionalProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $false)]
        [object]$Default = $null
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $Default
    }
    return $property.Value
}

function Invoke-RunPodJson {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('GET', 'POST')]
        [string]$Method,

        [Parameter(Mandatory = $true)]
        [string]$Uri,

        [Parameter(Mandatory = $false)]
        [object]$Body
    )

    $params = @{
        Method  = $Method
        Uri     = $Uri
        Headers = $script:Headers
    }

    if ($null -ne $Body) {
        $params.ContentType = 'application/json'
        $params.Body = ($Body | ConvertTo-Json -Depth 20 -Compress)
    }

    Invoke-RestMethod @params
}

$apiKey = Get-DotEnvValue -Name 'RUNPOD_API_KEY'
$endpointId = Get-DotEnvValue -Name 'IDENTITY_LORA_TRAINER_ENDPOINT_ID'

if ([string]::IsNullOrWhiteSpace($apiKey) -or [string]::IsNullOrWhiteSpace($endpointId)) {
    throw 'RUNPOD_API_KEY e IDENTITY_LORA_TRAINER_ENDPOINT_ID são obrigatórios.'
}

$script:Headers = @{ Authorization = "Bearer $apiKey" }
$requestId = "d3-6h10-transport-$([guid]::NewGuid().ToString('N'))"
$nonce = [guid]::NewGuid().ToString('N')
$submittedAt = [DateTimeOffset]::UtcNow

$payload = @{
    input = @{
        contract_version = $ContractVersion
        execution_mode   = 'queue_transport_only'
        request_id       = $requestId
        nonce            = $nonce
        safety           = @{
            training_allowed = $false
            preview_allowed  = $false
            r2_allowed       = $false
            model_load_allowed = $false
        }
    }
}

$runUri = "https://api.runpod.ai/v2/$endpointId/run"
$provider = Invoke-RunPodJson -Method POST -Uri $runUri -Body $payload
$jobId = [string]$provider.id

if ([string]::IsNullOrWhiteSpace($jobId)) {
    throw 'O RunPod não retornou job ID para o smoke de transporte.'
}

Write-Host (ConvertTo-Json @{
    status = 'STAGE_2_2D3_6H10_TRANSPORT_SMOKE_SUBMITTED'
    endpointIdPrefix = $endpointId.Substring(0, [Math]::Min(12, $endpointId.Length))
    providerJobId = $jobId
    contractVersion = $ContractVersion
    requestId = $requestId
    timeoutSeconds = $TimeoutSeconds
    safety = @{
        trainingStarted = $false
        previewStarted = $false
        r2Called = $false
        modelLoaded = $false
        automaticRetryCreated = $false
    }
} -Depth 20)

$statusUri = "https://api.runpod.ai/v2/$endpointId/status/$jobId"
$deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
$last = $null

while ([DateTimeOffset]::UtcNow -lt $deadline) {
    $last = Invoke-RunPodJson -Method GET -Uri $statusUri
    $state = [string]$last.status

    if ($state -in @('COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELLED')) {
        break
    }

    Start-Sleep -Seconds $PollSeconds
}

if ($null -eq $last) {
    throw 'Não foi possível consultar o job após a submissão.'
}

$finalState = [string]$last.status

if ($finalState -eq 'COMPLETED') {
    $output = Get-OptionalProperty -Object $last -Name 'output'
    if ($null -eq $output) {
        throw 'TRANSPORT_SMOKE_OUTPUT_MISSING: o job concluiu sem output.'
    }
    if ([string]$output.contract_version -ne $ContractVersion) {
        throw "TRANSPORT_SMOKE_CONTRACT_MISMATCH: $($output.contract_version)"
    }
    if ([string]$output.status -ne 'transport_smoke_completed') {
        throw "TRANSPORT_SMOKE_STATUS_INVALID: $($output.status)"
    }
    if ([string]$output.request_id -ne $requestId -or [string]$output.nonce -ne $nonce) {
        throw 'TRANSPORT_SMOKE_CORRELATION_MISMATCH'
    }

    Write-Host (ConvertTo-Json @{
        status = 'STAGE_2_2D3_6H10_RUNPOD_TRANSPORT_SMOKE_READY'
        providerJobId = $jobId
        providerStatus = $finalState
        delayMilliseconds = Get-OptionalProperty -Object $last -Name 'delayTime'
        executionMilliseconds = Get-OptionalProperty -Object $last -Name 'executionTime'
        runpodSdkVersion = $output.runpod_sdk_version
        workerPid = $output.worker_pid
        requestId = $requestId
        safety = $output.safety
    } -Depth 20)
    exit 0
}

if ($finalState -in @('FAILED', 'TIMED_OUT', 'CANCELLED')) {
    Write-Host (ConvertTo-Json @{
        status = 'STAGE_2_2D3_6H10_RUNPOD_TRANSPORT_SMOKE_TERMINAL_FAILURE'
        providerJobId = $jobId
        providerStatus = $finalState
        providerError = Get-OptionalProperty -Object $last -Name 'error'
        output = Get-OptionalProperty -Object $last -Name 'output'
        safety = @{
            automaticRetryCreated = $false
            automaticResubmitCreated = $false
        }
    } -Depth 20)
    exit 2
}

$health = Invoke-RunPodJson -Method GET -Uri "https://api.runpod.ai/v2/$endpointId/health"
$cancelled = $false
if ($CancelOnTimeout) {
    $cancel = Invoke-RunPodJson -Method POST -Uri "https://api.runpod.ai/v2/$endpointId/cancel/$jobId"
    $cancelled = ([string]$cancel.status -eq 'CANCELLED')
}

Write-Host (ConvertTo-Json @{
    status = 'STAGE_2_2D3_6H10_RUNPOD_TRANSPORT_SMOKE_TIMEOUT'
    providerJobId = $jobId
    providerStatus = $finalState
    elapsedSeconds = [Math]::Round(([DateTimeOffset]::UtcNow - $submittedAt).TotalSeconds, 2)
    cancelRequested = [bool]$CancelOnTimeout
    cancelled = $cancelled
    endpointHealth = $health
    nextAction = if ($CancelOnTimeout) { 'Confirme CANCELLED e fila zerada; não envie treino.' } else { 'Não reenvie. Preserve o job e use cancelamento específico após auditoria.' }
    safety = @{
        automaticRetryCreated = $false
        automaticResubmitCreated = $false
        trainingPayloadSubmitted = $false
    }
} -Depth 20)
exit 3
