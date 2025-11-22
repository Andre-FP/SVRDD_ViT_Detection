#!/usr/bin/env bash
set -e

# === CONFIGURAÇÕES ===
MODEL_NAME="fasterrcnn"
PYTHON_VERSION="3.11"
CUDA_VERSION="cu126"

if conda env list | grep -q "^$MODEL_NAME "; then
    echo ">>> Ambiente '$MODEL_NAME' já existe. Pulando criação."
else
    echo ">>> Criando ambiente conda..."
    conda create -n "$MODEL_NAME" python="$PYTHON_VERSION" -y
fi

echo ">>> Instalando PyTorch..."
conda run -n $MODEL_NAME pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/$CUDA_VERSION

echo ">>> Instalando requirements.txt..."
conda run -n $MODEL_NAME pip install -r requirements_$MODEL_NAME.txt

echo ">>> Registrando kernel Jupyter..."
conda run -n $MODEL_NAME python -m ipykernel install --user \
    --name=$MODEL_NAME \
    --display-name="$MODEL_NAME (conda)"

echo ">>> Ambiente $MODEL_NAME instalado com sucesso!"
