import os
import torch
import pytorch_lightning as pl
from detectron2.config import LazyConfig, instantiate
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_optimizer
from detectron2.utils.events import EventStorage
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn

# ---------- Paths e LazyConfig ----------
CONFIG_FILE = "./detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
PRETRAINED = './output_vitdet/model_final_b.pkl'
#CONFIG_FILE = "./detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_l_100ep.py"
#PRETRAINED = './output_vitdet/model_final_6146ed.pkl'
ROOT = "/workspace/SVRDD_COCO"
EPOCHS = 50
OUTPUT_DIR = "./outputs/vitdet"

register_coco_instances("asphalt_train",  {}, f"{ROOT}/train/_annotations.coco.json", f"{ROOT}/train")
register_coco_instances("asphalt_val",    {}, f"{ROOT}/valid/_annotations.coco.json", f"{ROOT}/valid")
register_coco_instances("asphalt_test",   {}, f"{ROOT}/test/_annotations.coco.json", f"{ROOT}/test")

cfg = LazyConfig.load(CONFIG_FILE)
cfg.dataloader.train.dataset.names = ("asphalt_train",)
cfg.dataloader.test.dataset.names  = ("asphalt_val",)
cfg.dataloader.evaluator.dataset_names = ("asphalt_test",)

cfg.dataloader.train.mapper = {
    "_target_": "detectron2.data.DatasetMapper",
    "is_train": True,
    "augmentations": [],   # <-- sem data augmentation
    "image_format": "BGR",
    "use_instance_mask": False,
}

cfg.dataloader.test.mapper = {
    "_target_": "detectron2.data.DatasetMapper",
    "is_train": True,
    "augmentations": [],   # <-- sem data augmentation
    "image_format": "BGR",
    "use_instance_mask": False,
}
cfg.dataloader.evaluator.mapper = {
    "_target_": "detectron2.data.DatasetMapper",
    "is_train": False,
    "augmentations": [],   # <-- sem data augmentation
    "image_format": "BGR",
    "use_instance_mask": False,
}

cfg.train.init_checkpoint = PRETRAINED
cfg.train.output_dir = OUTPUT_DIR
cfg.dataloader.train.total_batch_size = 1
cfg.dataloader.train.num_workers = 4
cfg.model.roi_heads.num_classes = 7
cfg.optimizer.lr = 1e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Instantiating data loader...")
train_loader = instantiate(cfg.dataloader.train)
val_loader = instantiate(cfg.dataloader.test)

# ---------- Lightning module wrapper ----------

def _set_bn_eval(module):
    # utility: seta todos BatchNorm para eval() (não atualiza running stats)
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()

class LitDetect(pl.LightningModule):
    def __init__(self, model, optimizer_cfg, train_loader, val_loader):
        super().__init__()
        
        self.model = model

        # we won't save optimizer in state dict here; lightning will re-create it in configure_optimizers
        self._optimizer_cfg = optimizer_cfg

        self._train_loader = train_loader
        self._val_loader = val_loader

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader
    
    def forward(self, batch):
        # not used for training, but keep it
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # detectron2 model expects a list[dict] batch with tensor fields or Instances on device
        with EventStorage(): 
            loss_dict = self.model(batch)  # returns dict of losses in train mode
        losses = sum(loss_dict.values())
        # log each loss
        for k, v in loss_dict.items():
            # v may be scalar tensor
            self.log(
                f"train_{k}", 
                v.item(),
                on_step=True, 
                on_epoch=True, 
                prog_bar=True, 
                sync_dist=True, 
                batch_size=len(batch)
            )
        self.log(
            "train_loss", 
            losses.item(), 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True, 
            batch_size=len(batch)
        )
        return losses

    def validation_step(self, batch, batch_idx):
        # queremos o loss_dict, então forçamos o caminho de treino
        was_training = self.model.training  # salvar estado anterior

        # 1) colocar em train() para que forward retorne losses
        self.model.train()

        # 2) evitar atualização de batchnorm statistics
        _set_bn_eval(self.model)

        # 3) sem grad para não consumir memória nem calcular grads
        with EventStorage(): 
            with torch.no_grad():
                out = self.model(batch)

        # restaurar estado original (opcional)
        if not was_training:
            self.model.eval()

        # checar tipo/forma do retorno
        if isinstance(out, dict):
            loss_dict = out
        elif isinstance(out, (list, tuple)):
            # se veio lista = predições -> algo deu errado (provavelmente model ainda em eval())
            raise RuntimeError(
                "Forward retornou predições (list). Assegure que o batch contém 'gt_instances' "
                "e que forçamos model.train() antes do forward."
            )
        else:
            raise RuntimeError(f"Retorno inesperado do modelo: {type(out)}")

        losses = sum(v for v in loss_dict.values())
        # log individual losses e agregado
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch), prog_bar=True)
        self.log("val_loss", losses.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch), prog_bar=True)

        return {"val_loss": losses}

    def configure_optimizers(self):
        # rebuild optimizer similar ao que fizemos
        optim_cfg = cfg.optimizer
        
        base_lr = float(optim_cfg.lr)
        weight_decay = float(getattr(optim_cfg, "weight_decay", 0.0))
        
        # ViTDet costuma usar LR menor no backbone (camadas mais profundas)
        backbone_lr_mult = getattr(optim_cfg, "backbone_multiplier", 0.1)
        
        backbone_params = []
        other_params = []
        
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            else:
                other_params.append(p)
        
        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
        )
        return optimizer

        
    
# ---------- Instancia Lightning e Trainer ----------
print("Instantiating model...")
model = instantiate(cfg.model)
model.mask_on = False
model.roi_heads.mask_on = False
model.roi_heads.mask_head = None

# for n,p in model.named_parameters():
#     if "backbone" in n:
#         p.requires_grad = False

model.to(device)

print("Checkpointer...")
checkpointer = DetectionCheckpointer(model, save_dir=cfg.train.output_dir)
if cfg.train.init_checkpoint:
    checkpointer.load(cfg.train.init_checkpoint)

print("Loading Model Lightning...")
lit_model = LitDetect(model=model, optimizer_cfg=cfg.optimizer, train_loader=train_loader, val_loader=val_loader)

# load pretrained weights via detectron2 checkpointer
ckpt = ModelCheckpoint(
    dirpath=cfg.train.output_dir, 
    monitor="val_loss", 
    mode="min", 
    save_top_k=1,
    save_last=True,
    filename="best-{epoch:02d}-{val_loss:.2f}"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=15,          # para após 5 épocas sem melhora
    min_delta=0.0,
)

lr_mon = LearningRateMonitor(logging_interval="step")

tb_logger = TensorBoardLogger("tb_logs", name="vitdet")

print("Training...")
pl_trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=EPOCHS,
    precision=16,
    callbacks=[ckpt, lr_mon, early_stop],
    accumulate_grad_batches=1,
    log_every_n_steps=200,
    logger=tb_logger,
    limit_train_batches=6000//cfg.dataloader.train.total_batch_size, 
    limit_val_batches=1000//cfg.dataloader.train.total_batch_size,  
)

pl_trainer.fit(lit_model)
checkpointer.save("model_vitdet_full_2")

def save_detectron2_pickle(model, path="output_vitdet/vitdet_finetuned.pkl"):
    data = {
        "model": model.state_dict(),
        "__author__": "detectron2",
        "matching_heuristics": True,
    }
    torch.save(data, path)
    print(f"Saved to {path}")

save_detectron2_pickle(model, path="model_vitdet_full_2.pkl")