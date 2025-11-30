from PIL import Image
from io import BytesIO
import requests
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device detectado: {device}")

# Caminhos do dataset
ROOT_DIR = "../../SVRDD_COCO"
OUTPUT_PATH = "./output_rfdetr_synth"
EPOCHS = 50

model = RFDETRBase()

model.train(
    dataset_dir=ROOT_DIR,
    epochs=EPOCHS,
    batch_size=8,
    grad_accum_steps=2,
    lr=1e-4,
    output_dir=OUTPUT_PATH,
    device=device,
    tensorboard=True,
    resolution=1064,
    checkpoint_interval=5
)