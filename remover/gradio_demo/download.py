import os
import cv2
import numpy as np
from PIL import Image
import torch
import time
import random
from huggingface_hub import snapshot_download
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


os.makedirs("./SAM2-Video-Predictor/checkpoints/", exist_ok=True)
snapshot_download(repo_id="facebook/sam2-hiera-large", local_dir="./SAM2-Video-Predictor/checkpoints/")
print("Download sam2 completed")