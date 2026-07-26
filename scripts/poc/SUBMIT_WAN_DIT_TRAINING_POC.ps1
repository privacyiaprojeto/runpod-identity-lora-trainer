param(
    [string]$PayloadPath = (Join-Path $PSScriptRoot '..\poc\training_input.local.json'),
    [string]$EndpointId,
    [string]$EnvFile = 'C:\Users\Usuario\Desktop\IAPrivacy\backend\.env',
    [string]$ResultRoot = (Join-Path $PSScriptRoot '..\poc-results\training'),
    [int]$PollSeconds = 20,
    [int]$TimeoutSeconds = 10800,
    [switch]$ValidateOnly
)
. (Join-Path $PSScriptRoot 'Common-RunPod.ps1')
if (-not (Test-Path -LiteralPath $PayloadPath)) { throw "Payload não encontrado: $PayloadPath" }
$payload = Get-Content -LiteralPath $PayloadPath -Raw -Encoding UTF8 | ConvertFrom-Json
Assert-NoPlaceholders -Payload $payload
$i = $payload.input
if ($i.contract_version -ne 'privacy-identity-lora-training-v2' -or $i.execution_mode -ne 'controlled_training_smoke') { throw 'Contrato/modo do treinamento inválido.' }
$t = $i.training
$expectedTargets = @('cross_attn.q','cross_attn.k','cross_attn.v','cross_attn.o','ffn.0','ffn.2')
if ($t.profile -ne 'wan_dit_identity_video_v1' -or $t.width -ne 832 -or $t.height -ne 480 -or $t.num_frames -ne 17) { throw 'Perfil espacial/temporal divergente.' }
if ($t.optimizer_steps -ne 800 -or $t.learning_rate -ne 0.00005 -or $t.lora_rank -ne 32 -or $t.lora_alpha -ne 32) { throw 'Hiperparâmetros divergentes do POC.' }
if ($t.lora_alpha -ne $t.lora_rank) { throw 'Neste DiffSynth fixado, alpha deve ser igual ao rank; alpha divergente seria apenas declarativo.' }
if ((@($t.checkpoint_steps) -join ',') -ne '400,600,800') { throw 'Checkpoints devem ser 400,600,800.' }
if ((@($t.target_modules) -join ',') -ne ($expectedTargets -join ',')) { throw 'Alvos Wan DiT divergentes.' }
if (-not $t.vace_frozen -or $t.automatic_retry) { throw 'VACE deve ficar congelado e retry automático desligado.' }
if (@($i.dataset.samples).Count -lt 15) { throw 'São necessárias pelo menos 15 amostras.' }
if (@($i.model.artifacts).Count -ne 9) { throw 'O lock do modelo deve conter exatamente 9 artefatos.' }
if ($i.output.public -ne $false -or -not $i.safety.private_storage_only -or -not $i.safety.public_urls_forbidden) { throw 'Saída privada obrigatória.' }
if ($ValidateOnly) { Write-Host 'TRAINING_PAYLOAD_READY — validação local aprovada; nenhum job enviado.'; exit 0 }
if ([string]::IsNullOrWhiteSpace($EndpointId)) {
    foreach ($name in @('IDENTITY_LORA_TRAINER_ENDPOINT_ID','RUNPOD_IDENTITY_LORA_ENDPOINT_ID','RUNPOD_LORA_TRAINER_ENDPOINT_ID','PRIVACY_LORA_ENDPOINT_ID')) {
        $EndpointId = Get-EnvValue -Name $name -EnvFile $EnvFile
        if ($EndpointId) { break }
    }
}
if ([string]::IsNullOrWhiteSpace($EndpointId)) { throw 'Informe -EndpointId do privacy-identity-lora-trainer ou configure RUNPOD_IDENTITY_LORA_ENDPOINT_ID.' }
$key = Get-RunPodApiKey -EnvFile $EnvFile
$run = Invoke-RunPodOneShot -EndpointId $EndpointId -Payload $payload -ApiKey $key -ResultRoot $ResultRoot -PollSeconds $PollSeconds -TimeoutSeconds $TimeoutSeconds
& (Join-Path $PSScriptRoot 'VERIFY_WAN_DIT_TRAINING_RESULT.ps1') -ResultPath (Join-Path $run.JobDir 'final-result.json')
