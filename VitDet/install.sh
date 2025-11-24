#!/usr/bin/env bash

# === CONFIGURAÇÕES ===
MODEL_NAME="vitdet"
PYTHON_VERSION="3.10"

echo ">>> Criando ambiente conda..."
conda create -n $MODEL_NAME python=$PYTHON_VERSION -y

echo ">>> Instalando PyTorch..."
conda run -n $MODEL_NAME pip install \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo ">>> Instalando requirements.txt..."
conda run -n $MODEL_NAME pip install -r requirements.txt

echo ">>> Instalando Detectron2..."
conda run -n $MODEL_NAME pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

echo ">>> Clonando repositório Detectron2..."
git clone https://github.com/facebookresearch/detectron2.git || true

echo ">>> Registrando kernel Jupyter..."
conda run -n $MODEL_NAME python -m ipykernel install --user \
    --name=$MODEL_NAME \
    --display-name="VitDet ($MODEL_NAME)"

echo ">>> Ambiente $MODEL_NAME instalado com sucesso!"
