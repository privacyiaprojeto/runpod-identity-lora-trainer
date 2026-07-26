Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-EnvValue {
    param([Parameter(Mandatory=$true)][string]$Name, [string]$EnvFile)
    $processValue = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not [string]::IsNullOrWhiteSpace($processValue)) { return $processValue.Trim() }
    if ($EnvFile -and (Test-Path -LiteralPath $EnvFile)) {
        foreach ($line in Get-Content -LiteralPath $EnvFile -Encoding UTF8) {
            if ($line -match '^\s*#' -or [string]::IsNullOrWhiteSpace($line)) { continue }
            if ($line -match ('^\s*' + [regex]::Escape($Name) + '\s*=\s*(.*)\s*$')) {
                $value = $Matches[1].Trim()
                if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
                return $value
            }
        }
    }
    return $null
}

function Get-RunPodApiKey {
    param([string]$EnvFile)
    $key = Get-EnvValue -Name 'RUNPOD_API_KEY' -EnvFile $EnvFile
    if ([string]::IsNullOrWhiteSpace($key)) { throw 'RUNPOD_API_KEY não encontrada no processo nem no .env informado.' }
    return $key
}

function Assert-NoPlaceholders {
    param([Parameter(Mandatory=$true)]$Payload)
    $text = $Payload | ConvertTo-Json -Depth 100 -Compress
    if ($text -match '__[A-Z0-9_]+__') { throw 'O payload ainda contém placeholders. Preencha todos os campos antes da submissão.' }
    if ($text -match 'https?://') { throw 'O payload controlado não pode conter URL pública/presigned; use referências privadas bucket/key.' }
}

function Get-Sha256Text {
    param([Parameter(Mandatory=$true)][string]$Text)
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Invoke-RunPodOneShot {
    param(
        [Parameter(Mandatory=$true)][string]$EndpointId,
        [Parameter(Mandatory=$true)]$Payload,
        [Parameter(Mandatory=$true)][string]$ApiKey,
        [Parameter(Mandatory=$true)][string]$ResultRoot,
        [int]$PollSeconds = 15,
        [int]$TimeoutSeconds = 7200
    )
    if ($EndpointId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Endpoint ID inválido.' }
    New-Item -ItemType Directory -Path $ResultRoot -Force | Out-Null
    $payloadText = $Payload | ConvertTo-Json -Depth 100 -Compress
    $payloadHash = Get-Sha256Text -Text $payloadText
    $lockDir = Join-Path $ResultRoot '.submission-locks'
    New-Item -ItemType Directory -Path $lockDir -Force | Out-Null
    $lockPath = Join-Path $lockDir ($payloadHash + '.lock.json')
    $stream = $null
    try {
        $stream = [IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $initial = @{ status='reserved_locally'; endpoint_id=$EndpointId; payload_sha256=$payloadHash; reserved_at=(Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json
        $bytes = [Text.Encoding]::UTF8.GetBytes($initial)
        $stream.Write($bytes, 0, $bytes.Length)
    } catch [IO.IOException] {
        throw "Submissão bloqueada: este payload já possui lock local em $lockPath. Não envie novamente sem auditar o job anterior."
    } finally {
        if ($stream) { $stream.Dispose() }
    }

    $headers = @{ Authorization = "Bearer $ApiKey"; 'Content-Type' = 'application/json' }
    $runUri = "https://api.runpod.ai/v2/$EndpointId/run"
    $accepted = $false
    try {
        $response = Invoke-RestMethod -Method Post -Uri $runUri -Headers $headers -Body $payloadText -TimeoutSec 120
        if (-not $response.id) { throw 'RunPod não retornou job id.' }
        $accepted = $true
        $jobId = [string]$response.id
        $jobDir = Join-Path $ResultRoot $jobId
        New-Item -ItemType Directory -Path $jobDir -Force | Out-Null
        $Payload | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $jobDir 'request.json') -Encoding UTF8
        $response | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $jobDir 'accepted.json') -Encoding UTF8
        @{ status='accepted'; endpoint_id=$EndpointId; job_id=$jobId; payload_sha256=$payloadHash; accepted_at=(Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $lockPath -Encoding UTF8
        Write-Host "Job aceito: $jobId"

        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $terminal = @('COMPLETED','FAILED','CANCELLED','TIMED_OUT')
        do {
            if ((Get-Date) -ge $deadline) { throw "Timeout local após $TimeoutSeconds segundos. O job pode continuar no RunPod; não reenvie." }
            Start-Sleep -Seconds $PollSeconds
            $statusUri = "https://api.runpod.ai/v2/$EndpointId/status/$jobId"
            $status = Invoke-RestMethod -Method Get -Uri $statusUri -Headers @{ Authorization = "Bearer $ApiKey" } -TimeoutSec 120
            $status | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $jobDir 'latest-status.json') -Encoding UTF8
            Write-Host ("Status: {0}" -f $status.status)
        } while ($terminal -notcontains [string]$status.status)

        $status | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $jobDir 'final-result.json') -Encoding UTF8
        return @{ JobId=$jobId; JobDir=$jobDir; Result=$status }
    } catch {
        if (-not $accepted -and (Test-Path -LiteralPath $lockPath)) { Remove-Item -LiteralPath $lockPath -Force }
        throw
    }
}

function Get-RunPodOutput {
    param([Parameter(Mandatory=$true)]$Result)
    if ($Result.PSObject.Properties.Name -contains 'output') { return $Result.output }
    return $Result
}
