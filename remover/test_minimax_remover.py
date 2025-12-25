import torch
from diffusers.utils import export_to_video
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from remover.transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from remover.pipeline_minimax_remover import Minimax_Remover_Pipeline
import argparse

# the iterations is the hyperparameter for mask dilation
def inference(pixel_values, masks, iterations=6):
    video = pipe(
        images=pixel_values,
        masks=masks,
        num_frames=video_length,
        height=512,
        width=512,
        num_inference_steps=12,
        generator=torch.Generator(device=device).manual_seed(random_seed),
        iterations=iterations
    ).frames[0]
    export_to_video(video, "./outputs/output.mp4")

def load_video(video_path):
    vr = VideoReader(video_path)
    images = vr.get_batch(list(range(video_length))).asnumpy()
    images = torch.from_numpy(images)/127.5 - 1.0
    return images

def load_mask(mask_path):
    vr = VideoReader(mask_path)
    masks = vr.get_batch(list(range(video_length))).asnumpy()
    masks = torch.from_numpy(masks)
    masks = masks[:, :, :, :1]
    masks[masks > 20] = 255
    masks[masks < 255] = 0
    masks = masks / 255.0
    return masks

random_seed = 42
video_length = 81
device = torch.device("cuda:0")

vae = AutoencoderKLWan.from_pretrained("./vae", torch_dtype=torch.float16)
transformer = Transformer3DModel.from_pretrained("./transformer", torch_dtype=torch.float16)
scheduler = UniPCMultistepScheduler.from_pretrained("./scheduler")

pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
pipe.to(device)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="world", help="名字")
parser.add_argument("--num_inference_steps", type=int, default=12)
parser.add_argument("--video_length", type=int, default=24)
args = parser.parse_args()

video_path = "./outputs/origin.mp4"
mask_path = "./outputs/mask.mp4"

images = load_video(video_path)
masks = load_mask(mask_path)

inference(images, masks)