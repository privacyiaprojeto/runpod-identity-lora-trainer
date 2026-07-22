# Privacy IA — RunPod Identity LoRA Trainer

Worker dedicado ao treinamento controlado da identidade visual. Não substitui o worker de vídeo.

## Segurança padrão

- `PRIVACY_LORA_ALLOW_TRAINING=false`
- `PRIVACY_LORA_DRY_RUN_ONLY=true`
- somente bucket/key privados;
- valida checksum de cada material e dos nove pesos do modelo-base;
- saída privada com `qa_required=true`;
- nenhum produto é liberado pelo worker.

## Publicação

1. Crie um repositório separado com este diretório.
2. Publique a imagem no GHCR.
3. Crie um endpoint RunPod dedicado com volume persistente.
4. Configure os secrets R2 e HF no endpoint.
5. Somente após smoke controlado, configure no backend `IDENTITY_LORA_TRAINER_ENDPOINT_ID`, `IDENTITY_LORA_TRAINING_ENABLED=true` e `IDENTITY_LORA_TRAINER_DRY_RUN_ONLY=false`.

O comando de treinamento segue o contrato oficial do DiffSynth-Studio para Wan2.1-VACE-14B, com dataset interno `video,vace_video,vace_reference_image,prompt` e saída LoRA privada.
