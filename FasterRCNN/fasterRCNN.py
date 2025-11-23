import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import to_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

# PARAMETERS
NUM_CLASSES = 7 # 7 classes
BATCH_SIZE = 8
EPOCHS = 20

# Caminhos do dataset
ROOT_DIR = "../../SVRDD_COCO"
TRAIN_DIRECTORY = os.path.join(ROOT_DIR, "train")
VAL_DIRECTORY = os.path.join(ROOT_DIR, "valid")
TEST_DIRECTORY = os.path.join(ROOT_DIR, "test")
ANNOTATION_FILE_NAME = "_annotations.coco.json" # Nome padrão do seu arquivo
CLASSES = {
    0: "longitudinal crack",
    1: "transverse crack",
    2: "alligator crack",
    3: "pothole",
    4: "manhole cover",
    5: "longitudinal patch",
    6: "transverse patch",
}

class CocoDetectionCustom(CocoDetection):
    def __init__(self, image_directory_path: str):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super().__init__(image_directory_path, annotation_file_path)

    def __getitem__(self, idx):
        # super retorna (PIL Image, annotations list)
        img, anns = super().__getitem__(idx)

        # converte para Tensor float32 em [0,1]
        img = to_tensor(img)  # shape [C,H,W], dtype=float32

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for a in anns:
            xmin, ymin, w, h = a["bbox"]
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(a["category_id"])
            areas.append(w * h)
            iscrowd.append(a.get("iscrowd", 0))

        if len(boxes) == 0:
            # lidar com imagens sem anotações (opcional: pular na collate)
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([int(self.ids[idx])])  # id do COCO

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = CocoDetectionCustom(TRAIN_DIRECTORY)
val_dataset = CocoDetectionCustom(VAL_DIRECTORY)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,        # ajusta conforme CPU (p.ex. os.cpu_count())
    pin_memory=True
)

valid_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

import torch
import pytorch_lightning as pl
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json


def _set_bn_eval(module):
    # mantém BatchNorm em eval para não alterar running stats
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
    elif isinstance(module, torch.nn.Dropout):
        module.eval()

class FasterRCNNLightning(pl.LightningModule):
    def __init__(self, num_classes, weights="DEFAULT", lr=0.005):
        super().__init__()
        self.save_hyperparameters()

        # carregar modelo pré-treinado
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        # substituir a head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        self.model.train()
        # Força BatchNorm para eval para evitar updates de running stats
        self.model.apply(_set_bn_eval)
        
        with torch.no_grad():
            loss_dict = self.model(images, targets)
            
        loss = sum(loss_dict.values())

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=0.0005
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Defining Model
model = FasterRCNNLightning(num_classes=NUM_CLASSES)
model.to(device)

# Defining Callbacks and Logs
#earlystop_cb = EarlyStopping(
#    monitor="val_loss",   # era "validation_loss"
#    patience=15,
#    mode="min"
#)

checkpoint_cb = ModelCheckpoint(
    monitor="val_loss",        # OK se você logar mAP em on_validation_epoch_end
    mode="min",
    save_top_k=2,
    filename="best-val-loss"
)

tb_logger = TensorBoardLogger("tb_logs", name="vitdet")

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu",
    devices=1,
    precision="16",  # AMP habilitado
    callbacks=[checkpoint_cb],    #, earlystop_cb],
    log_every_n_steps=10,
    enable_progress_bar=True,
    enable_checkpointing=True,
    logger=tb_logger
)

# Training
trainer.fit(model, train_data_loader, valid_data_loader)

