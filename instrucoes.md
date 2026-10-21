## Sequência de códigos para executar o HOTR em CPU, na presença de GPU Nvidia, acessar o repositório oficial para execução com "conda"

- Informações do notebook
- Intel Core i5-1334U
- 12 CPUs lógicas
- 16 GB de RAM
- ~12 GB disponíveis
- 4 GB de swap
- sem NVIDIA/CUDA

## Clonar o projeto do GitHub

git clone https://github.com/kakaobrain/hotr.git
cd hotr

## Instalar Miniconda

cd ~/Downloads

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

## Durante a instalação:
- pressione Enter para avançar pelo contrato;
- digite yes quando perguntar se aceita os termos;
- quando perguntar o diretório de instalação, pode simplesmente pressionar Enter para usar o padrão (~/miniconda3);
- quando perguntar se deseja inicializar o Conda, responda yes.

~/miniconda3/bin/conda init bash #### pode não ser necessário ####

source ~/.bashrc

## Criar ambiente do HOTR

conda create -n kakaobrain python=3.7 -y

conda activate kakaobrain

conda install -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -y

conda install cython scipy -y

pip install pycocotools opencv-python wandb networkx

## Baixar Dataset HICO-DET

Colar o seguinte link no navegador: https://drive.google.com/open?id=1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS

Fazer o download do arquivo: hico_20160224_det.tar.gz

Mover o arquivo compactado para hotr/

Descompactar "hico_20160224_det"

####

Fazer o download da pasta "annotations" e do arquivo "list_action.txt" dispoíveis nesse repositório

Colocá-las no caminho hotr/hico_20160224_det/

## Baixar checkpoint

Colar o seguinte link no navegador: https://arena.kakaocdn.net/brainrepo/hotr/hico_ft_q16.pth

Aguardar download dos arquivos "hico_ft_q16.pth" e "hico_ft_q16"

Entrar na pasta hotr/

Criar a pasta "hico_det"

Mover os arquivos baixados para hotr/hico_det

## Configuração opcional - para controlar a quantidade de saída(?)

Caso o processo se demonstre lento, por se tratarem de 9,5k anotações, uma sugestão é limitar essa quantidade

Uma sugestão está no arquivo "hico.py" disponível nesse repositório

Baixá-lo e substituir: hotr/hotr/data/datasets/hico.py

## Configuração para salvar o arquivo de saída

Baixar o arquivo "evaluator_hico.py" disponível nesse repositório

Substituir: hort/hotr/engine/evaluator_hico.py"

## Executar o projeto

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

## Geração dos triplets

python predictions_to_triplets.py

## Geração do grafo

python build_graph.py

## Geração da imagem (opcional)

python visualize_hoi.py ID

Onde "ID" deve ser substituído pelo índice da imagem analisada no processo anterior
No caso dos parâmetros utilizados neste projeto, "ID" pode ser um número entre 1 e 955
