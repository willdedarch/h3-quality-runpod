# MiniMax H3 Quality — ComfyUI para RunPod

Um unico ComfyUI para **RTX PRO 6000 Blackwell 96 GB**, com **200 GB de disco temporario**, sem volume de rede ou volume local persistente. ComfyUI **v0.40.0**; imagem-base fixada por digest; encoder oficial Qwen BF16.

[Abrir o template no RunPod](https://console.runpod.io/deploy?template=952bbvmmrg).

A imagem foi publicada e o template foi registrado. A referencia exata, o build aprovado e a leitura de confirmacao do RunPod ficam em [deployment.json](deployment.json). Os [registros do build](https://github.com/willdedarch/h3-quality-runpod/actions/runs/38828290979) mostram as verificacoes da imagem.

## Cinco workflows

| Arquivo | Modelo e ajustes | Passos |
| --- | --- | --- |
| `01_REF_Oficial_BF16.json` | Ref2VA oficial pruned BF16, sem LoRAs | 20 |
| `02_FirstLast_Oficial_BF16.json` | FL2VA oficial pruned BF16, primeiro e ultimo frame, sem LoRAs | 20 |
| `03_REF_Acao_BF16.json` | Ref2VA BF16 + Combat V2 0.8 + Motion Repair V2 0.6 | 30 |
| `04_FirstLast_Acao_BF16.json` | FL2VA BF16 + as mesmas LoRAs | 30 |
| `05_REF_Singularity_INT8_Comparacao.json` | Singularity v1.3 INT8 completo, sem LoRAs adicionais | 20 |

Todos usam atencao nativa do PyTorch, encoder BF16, video VAE FP16, audio VAE FP32 e 24 fps. Os presets iniciais usam 1 MP horizontal, duracao nominal de 5 segundos e seed fixa 42. Nao ativam Turbo, cache de inferencia nem mecanismos para pular avaliacoes do modelo. As forcas das LoRAs sao pontos iniciais ajustaveis.

Os arquivos 01/02 sao **adaptacoes dos grafos oficiais** para os arquivos selecionados. Os originais exatos e a licenca estao em [upstream](runpod/basic/upstream); os hashes de origem e das adaptacoes estao em [workflows.provenance.json](runpod/basic/workflows.provenance.json).

Singularity e um checkpoint alternativo escolhido pelo workflow. Seu arquivo original publicado e INT8. Selecionar esse workflow nao soma o checkpoint oficial nem carrega todos os modelos ao mesmo tempo na GPU.

## Usar no RunPod

1. Abra o link do template em `deployment.json` e selecione a **RTX PRO 6000 Blackwell completa, com 96 GB**.
2. Mantenha **Container Disk 200 GB**, sem adicionar volumes persistentes ou de rede.
3. Inicie o pod. A inicializacao confere GPU/CUDA e espaco livre, baixa e verifica os oito arquivos de pesos e depois abre o ComfyUI na porta **8188**.
4. No ComfyUI, abra a pasta de workflows `H3_QUALITY`. Envie a imagem de referencia; para First/Last, envie as duas imagens.
5. Baixe os videos de `/workspace/h3-basic/output` antes de parar, reiniciar ou encerrar o pod. Esse diretorio esta no disco temporario.

Para comparar oficial e Singularity, use 01 e 05 com **a mesma imagem, prompt, seed, resolucao, duracao e 20 passos**. Esses dois grafos diferem apenas na escolha do checkpoint e nos identificadores de saida.

## Pesos e inicializacao

[models.lock.json](runpod/basic/models.lock.json) fixa repositorio, revisao, tamanho e SHA-256 de todos os oito arquivos. O total e **172085535086 bytes**, aproximadamente **172.09 GB decimais**. Os pesos ficam fora da imagem e sao baixados em cada pod novo.

O instalador usa duas transferencias simultaneas, verifica o SHA-256 completo e move o arquivo validado para o destino sem manter uma segunda copia completa. Todos os cinco workflows recebem seus arquivos antes da abertura do ComfyUI. Nenhuma quantizacao alternativa e substituida automaticamente.

## Verificacoes e limite da validacao

O build executa oito testes de integridade dos grafos e de tratamento dos downloads, confere os metadados remotos dos pesos, verifica dependencias e registro dos nos nativos e inicia o ComfyUI em CPU. O build nao baixa os pesos nem gera videos.

**Renderizacao e comparacao visual na GPU ainda nao foram executadas.** O resultado das verificacoes de publicacao esta em `deployment.json` e nos registros do GitHub Actions.

## Reconstruir a imagem

```sh
docker build --platform linux/amd64 -f runpod/basic/Dockerfile -t h3-quality:local .
```

O GitHub Actions publica cada build aprovado no GHCR com tag unica. O template deve apontar para o **digest** publicado, para preservar a versao escolhida.
