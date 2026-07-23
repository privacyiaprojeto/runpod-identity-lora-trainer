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

## D3.6D — Runtime completo e preflight do entrypoint

- O conjunto homologado inclui `librosa`, `soundfile`, `soxr`, `numba` e `scipy`, além de `libsndfile1` no sistema.
- O build importa o entrypoint oficial `train.py`, valida `WanTrainingModule`/`wan_parser` e executa um probe WAV real de leitura e reamostragem 8 kHz → 16 kHz.
- O mesmo preflight roda no handler antes da reserva do lock one-shot; runtime incompleto não consome nova execução.
- Ausência de dependência de áudio recebe código específico `TRAINING_RUNTIME_AUDIO_DEPENDENCY_MISSING`.
- Nenhum retry é automático, nenhum adapter é aprovado/publicado e todos os produtos permanecem bloqueados até QA.

## D3.6E — Model binding agrupado e preflight do loader

- Os sete shards `diffusion_pytorch_model-00001..00007-of-00007.safetensors` são apresentados ao DiffSynth como **um único componente agrupado** em `--model_paths`.
- O text encoder e o VAE permanecem componentes separados, preservando o contrato oficial do `WanVideoPipeline`.
- O worker aceita somente `Wan-AI/Wan2.1-VACE-14B`, a revisão congelada recebida no run e os nove artefatos já aprovados pelo backend.
- Todos os artefatos devem existir no cache privado; `local_files_only=true` impede fallback silencioso para revisão remota ou download durante o smoke.
- Antes de reservar o lock one-shot, o worker calcula o hash estrutural dos sete shards agrupados com o loader do DiffSynth e exige identificação como `wan_video_vace`.
- O preflight lê apenas cabeçalhos/chaves dos pesos, não instancia GPU nem carrega o checkpoint completo em memória.
- Falhas de binding recebem códigos específicos (`TRAINING_MODEL_DETECTION_FAILED` ou `TRAINING_MODEL_PREFLIGHT_FAILED`) e nunca provocam retry automático.

### Isolamento dos testes de contrato (D3.6E v3)

Os módulos puros de binding, hash e classificação não importam `boto3` ou
`huggingface_hub` no carregamento. Os SDKs de runtime são importados somente
quando uma operação real de R2/Hugging Face é executada. Assim, o job
`contract-tests` continua deliberadamente mínimo (`pytest` apenas) e detecta
acoplamentos indevidos antes do Docker build.
