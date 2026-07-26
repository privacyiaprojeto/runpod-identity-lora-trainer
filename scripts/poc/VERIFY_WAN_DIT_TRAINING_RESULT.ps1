param([Parameter(Mandatory=$true)][string]$ResultPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ResultPath)) { throw "Resultado não encontrado: $ResultPath" }
$r = Get-Content -LiteralPath $ResultPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($r.status -and $r.status -ne 'COMPLETED') { throw "Job não concluído: $($r.status)" }
$o = if ($r.PSObject.Properties.Name -contains 'output') { $r.output } else { $r }
if ($o.status -ne 'training_completed' -or $o.contract_version -ne 'privacy-identity-lora-training-v2') { throw 'Saída do trainer incompatível.' }
$a = $o.adapter
if (-not $a.r2_bucket -or -not $a.r2_key -or -not $a.sha256 -or -not ($a.r2_key -match 'step-800\.safetensors$')) { throw 'Adapter final privado step-800 inválido.' }
if ($a.rank -ne 32 -or $a.alpha -ne 32 -or $a.recommended_strength_model -ne 0.65) { throw 'Metadados LoRA finais divergentes.' }
$m = $a.manifest
if ($m.optimizer_steps -ne 800 -or $m.lora_base_model -ne 'dit' -or $m.remove_prefix_in_ckpt -ne 'pipe.dit.') { throw 'Manifesto do adapter inválido.' }
$steps = @($m.checkpoints | ForEach-Object { $_.step }) -join ','
if ($steps -ne '400,600,800') { throw "Checkpoints incompletos: $steps" }
if ((@($m.target_modules) -join ',') -ne 'cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2') { throw 'Adapter fora do escopo Wan DiT aprovado.' }
Write-Host 'WAN_DIT_TRAINING_COMPLETED — .safetensors step-800 privado e auditado.'
Write-Host ("R2: {0}/{1}" -f $a.r2_bucket, $a.r2_key)
Write-Host ("SHA-256: {0}" -f $a.sha256)
