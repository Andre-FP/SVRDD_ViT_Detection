import os
import torch
import warnings
import os
from PIL import Image
from pycocotools.coco import COCO

# Importações do Hugging Face e PyTorch
from transformers import (
    DeformableDetrImageProcessor,
    DeformableDetrForObjectDetection,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection # <--- Importação chave!
from pytorch_lightning.loggers import TensorBoardLogger

# ⚠️ Ignorar warnings (como os de precisão mista ou de compatibilidade)
warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*",
    category=UserWarning,
    module="torch.nn.modules.module"
)
warnings.filterwarnings(
    "ignore",
    message=".*The given NumPy array is not writeable.*",
    category=UserWarning,
    module="pycocotools.coco"
)


# ============================================================
# 1️⃣ Variáveis e Configuração do Ambiente
# ============================================================
# Definição do Device para a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device detectado: {device}")

# Caminhos do dataset
ROOT_DIR = "/workspace/SVRDD_COCO"
TRAIN_DIRECTORY = os.path.join(ROOT_DIR, "train")
VAL_DIRECTORY = os.path.join(ROOT_DIR, "valid")
ANNOTATION_FILE_NAME = "_annotations.coco.json" # Nome padrão do seu arquivo

# Labels
id2label = {
    0: "longitudinal crack", 1: "transverse crack", 2: "alligator crack",
    3: "pothole", 4: "manhole cover", 5: "longitudinal patch",
    6: "transverse patch",
}
label2id = {v: k for k, v in id2label.items()}

# Modelo base
model_name = "SenseTime/deformable-detr"


# ============================================================
# 2️⃣ Classe Customizada para Dataloading Robusto
# ============================================================
class CocoDetectionCustom(CocoDetection):
    # Herda a lógica de leitura de arquivos COCO do torchvision
    def __init__(self, image_directory_path: str, image_processor):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        
        # O __init__ do CocoDetection carrega as anotações e os caminhos
        super(CocoDetectionCustom, self).__init__(image_directory_path, annotation_file_path)
        
        self.image_processor = image_processor

    def __getitem__(self, idx):
        # 1. Obter Imagem (PIL Image) e Anotações (lista COCO)
        image, annotations = super(CocoDetectionCustom, self).__getitem__(idx)
        image_id = self.ids[idx] # Obtém o ID da imagem do índice
        
        
        # 2. Formatar anotações para o processor do Hugging Face
        annotations = {'image_id': image_id, 'annotations': annotations}
        
        # 3. Processar a amostra (redimensionamento e conversão COCO -> DETR)
        # return_tensors="pt" é aplicado aqui, mas sem padding
        encoding = self.image_processor(images=image, annotations=annotations, return_tensors="pt")
        
        # 4. Retornar amostra sem a dimensão de batch (squeeze)
        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "pixel_mask": encoding["pixel_mask"].squeeze(),
            "labels": encoding["labels"][0], # O labels é uma lista, pegamos o primeiro elemento
        }

# ============================================================
# 3️⃣ Inicialização do Modelo e Datasets
# ============================================================
print("Carregando modelo e processor...")
processor = DeformableDetrImageProcessor.from_pretrained(model_name)

print("Instanciando Datasets...")
TRAIN_DATASET = CocoDetectionCustom(
    image_directory_path=TRAIN_DIRECTORY, 
    image_processor=processor
)
VAL_DATASET = CocoDetectionCustom(
    image_directory_path=VAL_DIRECTORY, 
    image_processor=processor
)
TEST_DATASET = CocoDetectionCustom(
    image_directory_path=VAL_DIRECTORY, 
    image_processor=processor
)

# ============================================================
# 4️⃣ Collate Function para BATCHING e PADDING
# ============================================================
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

num_workers = min(8, max(1, (os.cpu_count() or 4)//2))
print("num_workers =", num_workers)

params_data_loader = {
    "collate_fn": collate_fn, 
    "batch_size": 8, 
    "num_workers": num_workers,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2
}

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, shuffle=True, **params_data_loader)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, shuffle=False, **params_data_loader)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, **params_data_loader)

from transformers import DeformableDetrForObjectDetection
import pytorch_lightning as pl
import torch.nn as nn

class DeformableDetr(pl.LightningModule): # Mude o nome da classe para clareza
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        
        self.model = DeformableDetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        self.reinit_detection_heads()
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        
        # Freeze backbone
        for p in self.model.model.backbone.parameters():
            p.requires_grad = False
        
    def reinit_detection_heads(self):
        for name, module in self.model.named_modules():
            if "class_embed" in name or "bbox_embed" in name:
                for pn, p in module.named_parameters(recurse=False):
                    with torch.no_grad():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)
                        else:
                            p.zero_()
        print("Heads (class_embed / bbox_embed) reinicializadas.")
    
    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        # Move explicitamente para o device
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
    
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# 6️⃣ Trainer e Início
# ============================================================

model = DeformableDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

checkpoint = ModelCheckpoint(
    monitor="validation_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="best-{epoch}-{val_loss:.2f}"
)

early_stop = EarlyStopping(
    monitor="validation_loss",
    patience=12,
    mode="min"
)

tb_logger = TensorBoardLogger("tb_logs", name="vitdet")

trainer = Trainer(
    devices=1, 
    accelerator="gpu", 
    max_epochs=100,  
    accumulate_grad_batches=4, 
    log_every_n_steps=100, 
    precision=16,
    callbacks=[checkpoint, early_stop],
    logger=tb_logger,
)


print("🚀 Iniciando treinamento com Pytorch Trainer...")
trainer.fit(model)


