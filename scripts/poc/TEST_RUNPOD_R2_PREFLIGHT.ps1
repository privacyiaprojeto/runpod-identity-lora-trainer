[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$EnvFile = "C:\Users\Usuario\Desktop\IAPrivacy\backend\.env",

    [Parameter(Mandatory = $true)]
    [string]$TrainingPayloadPath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F-]{36}$')]
    [string]$ActorId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F-]{36}$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [string]$ExpiresAt,

    [Parameter(Mandatory = $false)]
    [ValidateRange(30, 900)]
    [int]$TimeoutSeconds = 240,

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

$ExpectedConfirmation = 'TESTAR R2 PRIVADO SEM TREINO D3.6H11'
$ContractVersion = 'privacy-identity-lora-r2-preflight-v1'

if ($Confirm -ne $ExpectedConfirmation) {
    throw "Confirmação inválida. Use exatamente: $ExpectedConfirmation"
}
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw "Arquivo .env não encontrado: $EnvFile"
}
if (-not (Test-Path -LiteralPath $TrainingPayloadPath -PathType Leaf)) {
    throw "Payload de treinamento não encontrado: $TrainingPayloadPath"
}

function Get-DotEnvValue {
    param([Parameter(Mandatory = $true)][string]$Name)
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
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $false)][object]$Default = $null
    )
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Invoke-RunPodJson {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('GET', 'POST')][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $false)][object]$Body
    )
    $params = @{ Method = $Method; Uri = $Uri; Headers = $script:Headers }
    if ($null -ne $Body) {
        $params.ContentType = 'application/json'
        $params.Body = ($Body | ConvertTo-Json -Depth 40 -Compress)
    }
    Invoke-RestMethod @params
}

$expiry = [DateTimeOffset]::Parse($ExpiresAt).ToUniversalTime()
if ($expiry -le [DateTimeOffset]::UtcNow) {
    throw 'A janela do preflight já expirou.'
}

$document = Get-Content -LiteralPath $TrainingPayloadPath -Raw | ConvertFrom-Json
$training = if ($null -ne $document.PSObject.Properties['input']) { $document.input } else { $document }
$dataset = Get-OptionalProperty -Object $training -Name 'dataset'
$samples = @((Get-OptionalProperty -Object $dataset -Name 'samples' -Default @()))
$manifestSha = [string](Get-OptionalProperty -Object $training -Name 'dataset_manifest_sha256')

if ($samples.Count -lt 1) { throw 'O payload não contém amostras privadas.' }
if ($manifestSha -notmatch '^[0-9a-fA-F]{64}$') { throw 'dataset_manifest_sha256 inválido.' }

$objects = [System.Collections.Generic.List[object]]::new()
$seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
$bucket = $null

foreach ($sample in $samples) {
    $sampleId = [string](Get-OptionalProperty -Object $sample -Name 'sample_id' -Default 'sample')
    foreach ($definition in @(
        @{ Role = 'video_source'; SourceName = 'video_source'; ShaName = 'video_sha256' },
        @{ Role = 'reference_image_source'; SourceName = 'reference_image_source'; ShaName = 'reference_image_sha256' }
    )) {
        $source = Get-OptionalProperty -Object $sample -Name $definition.SourceName
        if ($null -eq $source) { throw "Fonte ausente em $sampleId/$($definition.Role)." }
        $itemBucket = [string](Get-OptionalProperty -Object $source -Name 'bucket')
        $itemKey = [string](Get-OptionalProperty -Object $source -Name 'key')
        $expectedSha = [string](Get-OptionalProperty -Object $sample -Name $definition.ShaName)

        if ([string]::IsNullOrWhiteSpace($itemBucket) -or [string]::IsNullOrWhiteSpace($itemKey)) {
            throw "Bucket/key privado ausente em $sampleId/$($definition.Role)."
        }
        if ($itemKey -match '^https?://' -or $itemKey.StartsWith('/')) {
            throw "Referência pública/absoluta proibida em $sampleId/$($definition.Role)."
        }
        if ($itemKey -notlike "*$ActorId*") {
            throw "Referência fora do escopo do ator em $sampleId/$($definition.Role)."
        }
        if ($expectedSha -notmatch '^[0-9a-fA-F]{64}$') {
            throw "Checksum inválido em $sampleId/$($definition.Role)."
        }
        if ($null -eq $bucket) { $bucket = $itemBucket }
        if ($itemBucket -ne $bucket) { throw 'O dataset usa mais de um bucket; preflight bloqueado.' }

        $identity = "$itemBucket`n$itemKey"
        if ($seen.Add($identity)) {
            $objects.Add(@{
                sample_id = $sampleId
                role = $definition.Role
                source = @{ bucket = $itemBucket; key = $itemKey }
                expected_sha256 = $expectedSha.ToLowerInvariant()
            })
        }
    }
}

if ($objects.Count -lt 2 -or $objects.Count -gt 64) {
    throw "Quantidade de objetos privados fora do limite: $($objects.Count)."
}

$apiKey = Get-DotEnvValue -Name 'RUNPOD_API_KEY'
$endpointId = Get-DotEnvValue -Name 'IDENTITY_LORA_TRAINER_ENDPOINT_ID'
if ([string]::IsNullOrWhiteSpace($apiKey) -or [string]::IsNullOrWhiteSpace($endpointId)) {
    throw 'RUNPOD_API_KEY e IDENTITY_LORA_TRAINER_ENDPOINT_ID são obrigatórios.'
}

$script:Headers = @{ Authorization = "Bearer $apiKey" }
$requestId = "d3-6h11-r2-preflight-$([guid]::NewGuid().ToString('N'))"
$submittedAt = [DateTimeOffset]::UtcNow

$payload = @{
    input = @{
        contract_version = $ContractVersion
        execution_mode = 'private_r2_metadata_preflight'
        request_id = $requestId
        actor_profile_id = $ActorId
        training_run_id = $RunId
        dataset_manifest_sha256 = $manifestSha.ToLowerInvariant()
        bucket = $bucket
        objects = @($objects)
        smoke = @{
            enabled = $true
            one_shot = $true
            max_jobs = 1
            actor_profile_id = $ActorId
            training_run_id = $RunId
            expires_at = $expiry.ToString('o')
        }
        safety = @{
            actor_scoped = $true
            run_scoped = $true
            private_storage_only = $true
            metadata_only = $true
            download_allowed = $false
            write_allowed = $false
            delete_allowed = $false
            training_allowed = $false
            model_load_allowed = $false
            automatic_retry_allowed = $false
            one_shot_smoke = $true
        }
    }
}

$runUri = "https://api.runpod.ai/v2/$endpointId/run"
$provider = Invoke-RunPodJson -Method POST -Uri $runUri -Body $payload
$jobId = [string]$provider.id
if ([string]::IsNullOrWhiteSpace($jobId)) {
    throw 'O RunPod não retornou job ID para o preflight privado do R2.'
}

Write-Host (ConvertTo-Json @{
    status = 'STAGE_2_2D3_6H11_R2_PREFLIGHT_SUBMITTED'
    endpointIdPrefix = $endpointId.Substring(0, [Math]::Min(12, $endpointId.Length))
    providerJobId = $jobId
    contractVersion = $ContractVersion
    requestId = $requestId
    actorProfileId = $ActorId
    trainingRunId = $RunId
    datasetManifestPrefix = $manifestSha.Substring(0, 12)
    objectsSubmitted = $objects.Count
    timeoutSeconds = $TimeoutSeconds
    safety = @{
        trainingStarted = $false
        previewStarted = $false
        modelLoaded = $false
        r2MetadataOnly = $true
        r2ObjectDownloaded = $false
        r2WriteExecuted = $false
        r2DeleteExecuted = $false
        automaticRetryCreated = $false
    }
} -Depth 30)

$statusUri = "https://api.runpod.ai/v2/$endpointId/status/$jobId"
$deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
$last = $null
while ([DateTimeOffset]::UtcNow -lt $deadline) {
    $last = Invoke-RunPodJson -Method GET -Uri $statusUri
    $state = [string]$last.status
    if ($state -in @('COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELLED')) { break }
    Start-Sleep -Seconds $PollSeconds
}
if ($null -eq $last) { throw 'Não foi possível consultar o job após a submissão.' }

$finalState = [string]$last.status
if ($finalState -eq 'COMPLETED') {
    $output = Get-OptionalProperty -Object $last -Name 'output'
    if ($null -eq $output) { throw 'R2_PREFLIGHT_OUTPUT_MISSING' }
    if ([string]$output.contract_version -ne $ContractVersion) { throw 'R2_PREFLIGHT_CONTRACT_MISMATCH' }
    if ([string]$output.status -ne 'r2_private_preflight_completed') { throw 'R2_PREFLIGHT_STATUS_INVALID' }
    if ([string]$output.request_id -ne $requestId) { throw 'R2_PREFLIGHT_CORRELATION_MISMATCH' }
    if ([string]$output.actor_profile_id -ne $ActorId -or [string]$output.training_run_id -ne $RunId) { throw 'R2_PREFLIGHT_SCOPE_MISMATCH' }
    if ([string]$output.dataset_manifest_prefix -ne $manifestSha.Substring(0, 12)) { throw 'R2_PREFLIGHT_MANIFEST_MISMATCH' }

    Write-Host (ConvertTo-Json @{
        status = 'STAGE_2_2D3_6H11_PRIVATE_R2_PREFLIGHT_READY'
        providerJobId = $jobId
        providerStatus = $finalState
        delayMilliseconds = Get-OptionalProperty -Object $last -Name 'delayTime'
        executionMilliseconds = Get-OptionalProperty -Object $last -Name 'executionTime'
        requestId = $requestId
        actorProfileId = $ActorId
        trainingRunId = $RunId
        storage = $output.storage
        safety = $output.safety
        automaticRetryCreated = $false
    } -Depth 30)
    exit 0
}

if ($finalState -in @('FAILED', 'TIMED_OUT', 'CANCELLED')) {
    Write-Host (ConvertTo-Json @{
        status = 'STAGE_2_2D3_6H11_PRIVATE_R2_PREFLIGHT_TERMINAL_FAILURE'
        providerJobId = $jobId
        providerStatus = $finalState
        providerError = Get-OptionalProperty -Object $last -Name 'error'
        output = Get-OptionalProperty -Object $last -Name 'output'
        safety = @{ automaticRetryCreated = $false; automaticResubmitCreated = $false }
    } -Depth 30)
    exit 2
}

$health = Invoke-RunPodJson -Method GET -Uri "https://api.runpod.ai/v2/$endpointId/health"
$cancelled = $false
if ($CancelOnTimeout) {
    $cancel = Invoke-RunPodJson -Method POST -Uri "https://api.runpod.ai/v2/$endpointId/cancel/$jobId"
    $cancelled = ([string]$cancel.status -eq 'CANCELLED')
}

Write-Host (ConvertTo-Json @{
    status = 'STAGE_2_2D3_6H11_PRIVATE_R2_PREFLIGHT_TIMEOUT'
    providerJobId = $jobId
    providerStatus = $finalState
    elapsedSeconds = [Math]::Round(([DateTimeOffset]::UtcNow - $submittedAt).TotalSeconds, 2)
    cancelRequested = [bool]$CancelOnTimeout
    cancelled = $cancelled
    endpointHealth = $health
    safety = @{
        automaticRetryCreated = $false
        automaticResubmitCreated = $false
        trainingPayloadSubmitted = $false
        r2WritePayloadSubmitted = $false
    }
} -Depth 30)
exit 3
