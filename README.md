# Ethics_in_HOI

Repositório do código utilizado no projeto **"Elaboração assistida por IA de Requisitos Éticos em Interações Humano-Objeto"**, da disciplina Engenharia de Inteligência Artificial.

## Sumário

- [Estrutura do repositório](#estrutura-do-repositório)
- [Prompts e respostas da etapa final](#prompts-e-respostas-da-etapa-final)
- [Execução do HOTR em CPU](#execução-do-hotr-em-cpu)
  - [Ambiente utilizado](#ambiente-utilizado)
  - [1. Clonar o projeto](#1-clonar-o-projeto)
  - [2. Instalar o Miniconda](#2-instalar-o-miniconda)
  - [3. Criar o ambiente do HOTR](#3-criar-o-ambiente-do-hotr)
  - [4. Baixar o dataset HICO-DET](#4-baixar-o-dataset-hico-det)
  - [5. Baixar o checkpoint](#5-baixar-o-checkpoint)
  - [6. Configuração opcional: limitar a quantidade de anotações](#6-configuração-opcional-limitar-a-quantidade-de-anotações)
  - [7. Configuração para salvar o arquivo de saída](#7-configuração-para-salvar-o-arquivo-de-saída)
  - [8. Executar o projeto](#8-executar-o-projeto)
  - [9. Gerar as triplas](#9-gerar-as-triplas)
  - [10. Gerar o grafo](#10-gerar-o-grafo)
  - [11. Gerar a imagem (opcional)](#11-gerar-a-imagem-opcional)

---

## Estrutura do repositório

A pasta `hotr` contém três subpastas, além dos arquivos criados para converter as predições em triplas, construir os grafos e gerar as imagens com caixas delimitadoras, rótulos, ações e scores.

| Subpasta | Conteúdo |
|---|---|
| `hico_20160224_det` | Arquivos associados ao HICO-DET que **não estão disponíveis** no [arquivo oficial do benchmark](https://drive.google.com/open?id=1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS). |
| `hotr` | Arquivos de código do HOTR **adaptados para este trabalho**. O repositório oficial, para clonagem do projeto e download do checkpoint do HICO-DET, está disponível em [umich-ywchao-hico.github.io](https://umich-ywchao-hico.github.io/). |
| `results` | Arquivos **gerados pela execução dos códigos**: triplas, grafos, contabilização dos resultados e um exemplo de imagem analisada. |

## Prompts e respostas da etapa final

O prompt e as respostas utilizados na etapa final do projeto podem ser acessados em [prompt_answers.md](https://github.com/L4R1SS4/Ethics_in_HOI/blob/main/prompt_answers.md).

---

## Execução do HOTR em CPU

Esta sequência de comandos executa o HOTR em **CPU**, mesmo na presença de GPU Nvidia. Para execução com `conda` e GPU, consulte o repositório oficial.

### Ambiente utilizado

| Item | Especificação |
|---|---|
| Processador | Intel Core i5-1334U |
| CPUs lógicas | 12 |
| Memória RAM | 16 GB (~12 GB disponíveis) |
| Swap | 4 GB |
| GPU | Sem NVIDIA/CUDA |

### 1. Clonar o projeto

```bash
git clone https://github.com/kakaobrain/hotr.git
cd hotr
```

### 2. Instalar o Miniconda

```bash
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Durante a instalação:

- pressione `Enter` para avançar pelo contrato;
- digite `yes` quando perguntar se aceita os termos;
- quando perguntar o diretório de instalação, pressione `Enter` para usar o padrão (`~/miniconda3`);
- quando perguntar se deseja inicializar o Conda, responda `yes`.

Em seguida:

```bash
~/miniconda3/bin/conda init bash   # pode não ser necessário
source ~/.bashrc
```

### 3. Criar o ambiente do HOTR

```bash
conda create -n kakaobrain python=3.7 -y
conda activate kakaobrain
conda install -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -y
conda install cython scipy -y
pip install pycocotools opencv-python wandb networkx
```

### 4. Baixar o dataset HICO-DET

1. Acesse o [link do dataset](https://drive.google.com/open?id=1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS) no navegador.
2. Baixe o arquivo `hico_20160224_det.tar.gz`.
3. Mova o arquivo compactado para `hotr/`.
4. Descompacte `hico_20160224_det`.

Em seguida, baixe a pasta `annotations` e o arquivo `list_action.txt` disponíveis neste repositório e coloque-os em `hotr/hico_20160224_det/`.

### 5. Baixar o checkpoint

1. Acesse o [link do checkpoint](https://arena.kakaocdn.net/brainrepo/hotr/hico_ft_q16.pth) no navegador.
2. Aguarde o download do arquivo `hico_ft_q16.pth`.
3. Entre na pasta `hotr/` e crie a pasta `hico_det`.
4. Descompacte o arquivo baixado no caminho `hotr/hico_det/`.

### 6. Configuração opcional: limitar a quantidade de anotações

Como o conjunto possui cerca de 9,5 mil anotações, o processo pode ser lento. Uma sugestão é limitar essa quantidade.

Uma versão com esse limite está no arquivo `hico.py`, disponível neste repositório. Baixe-o e substitua:

```
hotr/hotr/data/datasets/hico.py
```

### 7. Configuração para salvar o arquivo de saída

Baixe o arquivo `evaluator_hico.py`, disponível neste repositório, e substitua:

```
hotr/hotr/engine/evaluator_hico.py
```

### 8. Executar o projeto

```bash
python main.py \
    --device cpu \
    --batch_size 1 \
    --num_workers 2 \
    --HOIDet \
    --share_enc \
    --pretrained_dec \
    --num_hoi_queries 16 \
    --object_threshold 0 \
    --temperature 0.2 \
    --no_aux_loss \
    --eval \
    --dataset_file hico-det \
    --data_path hico_20160224_det \
    --resume hico_det/hico_ft_q16.pth
```

### 9. Gerar as triplas

```bash
python predictions_to_triplets.py
```

### 10. Gerar o grafo

```bash
python build_graph.py
```

### 11. Gerar a imagem (opcional)

```bash
python visualize_hoi.py ID
```

Substitua `ID` pelo índice da imagem analisada no processo anterior. Com os parâmetros utilizados neste projeto, `ID` pode ser um número entre 1 e 955.
