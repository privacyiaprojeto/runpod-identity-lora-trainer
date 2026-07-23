# Privacy IA — RunPod Identity LoRA Trainer

Worker dedicado ao treinamento controlado da identidade visual. Não substitui o worker de vídeo.

## Segurança padrão

- `PRIVACY_LORA_ALLOW_TRAINING=false`
- `PRIVACY_LORA_DRY_RUN_ONLY=true`
- `PRIVACY_LORA_SMOKE_MODE=false`
- somente bucket/key privados;
- valida checksum de cada material e dos nove artefatos do modelo-base;
- saída privada com `qa_required=true`;
- nenhum produto é liberado pelo worker.

## Primeiro treino real — D3.6B

A primeira execução real usa contrato `privacy-identity-lora-training-v2` e exige escopo one-shot:

- um `actor_profile_id`;
- um `training_run_id`;
- uma janela de expiração curta;
- exatamente um job;
- lock persistente no volume `/runpod-volume/privacy-identity-lora/smoke-locks`.

O worker recusa ator, run ou janela divergentes. Depois que o lock do run é reservado, qualquer nova solicitação para o mesmo run é bloqueada, inclusive após reinício do container.

## Publicação

1. Publique a nova imagem no GHCR usando tag imutável `sha-...`.
2. Atualize o endpoint RunPod dedicado para essa tag.
3. Mantenha o worker fechado até o backend gerar o escopo exato.
4. Para o smoke, replique no endpoint as variáveis retornadas pelo comando de armamento D3.6B.
5. Depois da submissão ou do término, volte o endpoint para:
   - `PRIVACY_LORA_ALLOW_TRAINING=false`
   - `PRIVACY_LORA_DRY_RUN_ONLY=true`
   - `PRIVACY_LORA_SMOKE_MODE=false`

O comando de treinamento segue o contrato do DiffSynth-Studio para Wan2.1-VACE-14B, com dataset interno privado e saída LoRA privada aguardando QA.

## D3.6C — Runtime lock e falha recuperável

- `transformers`, `huggingface_hub`, `accelerate`, `tokenizers` e `peft` ficam fixados em conjunto compatível.
- O build executa `pip check` e um preflight real de importação; uma imagem quebrada não é publicada como válida.
- O handler repete o preflight antes de reservar o lock one-shot, impedindo que incompatibilidade de container consuma a execução controlada.
- Falhas técnicas são classificadas como recuperáveis, porém nunca geram retry automático.
- O adapter continua privado e `qa_pending`; nenhuma liberação de produto ocorre no worker.
