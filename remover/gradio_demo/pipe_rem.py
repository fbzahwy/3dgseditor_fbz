import os
import cv2
import json
import argparse
import numpy as np

import torch
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip

from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


# -------------------------
# Video / IO
# -------------------------
def load_video_frames_512(video_path: str, max_frames: int, frame_size: int = 512) -> np.ndarray:
    """Load frames -> resize to (frame_size, frame_size). Return [T,512,512,3] uint8."""
    vr = VideoReader(video_path, ctx=cpu(0))
    n = min(len(vr), max_frames)
    frames = []
    for i in range(n):
        f = vr[i].asnumpy()  # HWC, uint8
        f = cv2.resize(f, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frames.append(f)
    del vr
    return np.stack(frames, axis=0)


def save_video(frames_uint8, out_path, fps: int):
    """frames_uint8: list of 512x512x3 uint8"""
    clip = ImageSequenceClip(frames_uint8, fps=fps)
    clip.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)


# -------------------------
# Model loaders
# -------------------------
def load_remover_pipe(vae_path, transformer_path, scheduler_path, device):
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained(transformer_path, torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained(scheduler_path)

    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(device)
    return pipe


def load_sam2_predictors(config, checkpoint, device, image_size: int = 512):
    video_predictor = build_sam2_video_predictor(config, checkpoint, device=device)
    model = build_sam2(config, checkpoint, device=device)
    model.image_size = image_size  # FIX: 512
    image_predictor = SAM2ImagePredictor(sam_model=model)
    return image_predictor, video_predictor


# -------------------------
# Preprocess for remover (FIX 512)
# -------------------------
def preprocess_for_removal_512(images_512, masks_512, device, frame_size: int = 512):
    """
    images_512: [T,512,512,3] uint8
    masks_512:  [T,512,512] float32/uint8
    return:
      img_tensor:  [T,512,512,3] half, in [-1,1]
      mask_tensor: [T,512,512] half, 0/1
    """
    out_images = []
    out_masks = []
    for img, msk in zip(images_512, masks_512):
        # 确保就是 512
        if img.shape[0] != frame_size or img.shape[1] != frame_size:
            img = cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        if msk.shape[0] != frame_size or msk.shape[1] != frame_size:
            msk = cv2.resize(msk, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 127.5 - 1.0  # [-1,1]
        msk = (msk > 0.5).astype(np.float32)

        out_images.append(img)
        out_masks.append(msk)

    arr_images = np.stack(out_images, axis=0)
    arr_masks = np.stack(out_masks, axis=0)

    return (
        torch.from_numpy(arr_images).half().to(device),
        torch.from_numpy(arr_masks).half().to(device),
    )


# -------------------------
# SAM2: first-frame mask from normalized points ONLY
# -------------------------
# def segment_first_frame_with_normalized_points(image_predictor, first_frame_512, points_xy_norm):
#     """
#     first_frame_512: 512x512x3 uint8
#     points_xy_norm: Nx2 float32 in [0,1]
#     return: 512x512 float32 mask (0/1)
#     """
#     point_coords = points_xy_norm.astype(np.float32)
#     point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)  # positive only

#     image_predictor.set_image(first_frame_512)
#     mask, _, _ = image_predictor.predict(
#         point_coords=point_coords,
#         point_labels=point_labels,
#         multimask_output=False,
#         normalize_coords=False,  # 你的点已经 normalized，保持与你原代码一致
#     )

#     mask = np.squeeze(mask).astype(np.float32)
#     mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
#     mask = (mask > 0.5).astype(np.float32)
#     return mask
def segment_first_frame_with_normalized_points(
    image_predictor,
    first_frame_512,
    points_xy_norm,
    debug_dir: str = None,
    prefix: str = "first_frame"
):
    point_coords = points_xy_norm.astype(np.float32)
    point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)  # positive only

    image_predictor.set_image(first_frame_512)
    mask, _, _ = image_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
        normalize_coords=False,
    )

    mask = np.squeeze(mask).astype(np.float32)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask > 0.5).astype(np.uint8)  # 0/1 for debug

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        # 1) save binary mask png (0/255)
        mask_png = (mask_bin * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_mask.png"), mask_png)

        # 2) save overlay on first frame
        overlay = first_frame_512.copy()
        # red overlay where mask==1
        overlay[mask_bin == 1] = (0.5 * overlay[mask_bin == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 3) save points on image (for checking coords)
        pts_img = first_frame_512.copy()
        for (x, y) in point_coords:
            px = int(round(x * 511))
            py = int(round(y * 511))
            cv2.circle(pts_img, (px, py), radius=4, color=(255, 0, 0), thickness=-1)  # red dot in RGB
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_points.png"), cv2.cvtColor(pts_img, cv2.COLOR_RGB2BGR))

        # 4) optional: save raw mask as npy for later inspection
        np.save(os.path.join(debug_dir, f"{prefix}_mask.npy"), mask.astype(np.float32))

    return mask_bin.astype(np.float32)


# -------------------------
# SAM2: tracking masks across frames (FIX 512)
# -------------------------
def track_masks_512(video_predictor, frames_512, first_mask_512, obj_id: int, device: str):
    """
    frames_512: [T,512,512,3] uint8
    first_mask_512: [512,512] float32
    return masks: [T,512,512] float32
    """
    frames_f = frames_512.astype(np.float32) / 255.0
    inference_state = video_predictor.init_state(images=frames_f, device=device)

    mask_t = torch.from_numpy(first_mask_512)  # 512x512
    video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=mask_t
    )

    masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        # 单目标：取第一个 logit
        logit = out_mask_logits[0]
        out_mask = logit.cpu().squeeze().detach().numpy()
        out_mask = (out_mask > 0).astype(np.float32)  # 512x512
        if out_mask.shape[0] != 512 or out_mask.shape[1] != 512:
            out_mask = cv2.resize(out_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        masks.append(out_mask)

    return np.stack(masks, axis=0)  # [T,512,512]


# -------------------------
# Remover inference (FIX 512)
# -------------------------
def run_remover_512(pipe, frames_512, masks_512, num_inference_steps, iterations, seed, device):
    img_tensor, mask_tensor = preprocess_for_removal_512(frames_512, masks_512, device, frame_size=512)
    mask_tensor = mask_tensor[:, :, :, None]  # [T,512,512,1]

    g = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        out = pipe(
            images=img_tensor,
            masks=mask_tensor,
            num_frames=mask_tensor.shape[0],
            height=512,   # FIX
            width=512,    # FIX
            num_inference_steps=int(num_inference_steps),
            generator=g,
            iterations=int(iterations),
        ).frames[0]

    out = np.uint8(np.clip(out * 255.0, 0, 255))
    out_frames = [out[i] for i in range(out.shape[0])]

    # 双保险：强制输出 512x512
    out_frames = [cv2.resize(f, (512, 512), interpolation=cv2.INTER_LINEAR) for f in out_frames]
    return out_frames


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)

    # SAM2
    parser.add_argument("--sam2_checkpoint", type=str, default="./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--sam2_config", type=str, default="sam2_hiera_l.yaml")

    # Remover
    parser.add_argument("--vae", type=str, default="./model/vae")
    parser.add_argument("--transformer", type=str, default="./model/transformer")
    parser.add_argument("--scheduler", type=str, default="./model/scheduler")

    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    # points: normalized only
    parser.add_argument("--points_json", type=str, required=True, help='JSON list normalized, e.g. "[[0.2,0.3],[0.4,0.5]]"')

    # optional
    parser.add_argument("--no_tracking", action="store_true", help="If set, use first-frame mask for all frames")
    parser.add_argument("--obj_id", type=int, default=1)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load frames -> 512
    frames = load_video_frames_512(args.video_path, max_frames=args.video_length, frame_size=512)
    print(f"[INFO] loaded frames: {frames.shape} (must be T,512,512,3)")

    # 2) parse points (normalized only)
    points_xy = np.array(json.loads(args.points_json), dtype=np.float32)
    print(f"[INFO] parsed points: {points_xy.shape[0]} (normalized)")

    # 3) load models
    print("[INFO] loading SAM2...")
    image_predictor, video_predictor = load_sam2_predictors(
        args.sam2_config, args.sam2_checkpoint, device, image_size=512
    )

    print("[INFO] loading remover...")
    pipe = load_remover_pipe(args.vae, args.transformer, args.scheduler, device)

    # 4) first frame mask
    print("[INFO] segmenting first frame...")
    first_mask = segment_first_frame_with_normalized_points(image_predictor, frames[0], points_xy, debug_dir=args.out_dir, prefix="frame0")
    print(f"[INFO] first mask sum={first_mask.sum():.1f}")

    # 5) tracking
    if args.no_tracking:
        print("[INFO] no_tracking enabled, repeating first mask")
        masks = np.repeat(first_mask[None, ...], frames.shape[0], axis=0)
    else:
        print("[INFO] tracking masks with SAM2 video predictor...")
        masks = track_masks_512(video_predictor, frames, first_mask, obj_id=args.obj_id, device=device)
        print(f"[INFO] masks: {masks.shape}")

    # 6) remover
    print("[INFO] running minimax remover...")
    out_frames = run_remover_512(
        pipe=pipe,
        frames_512=frames,
        masks_512=masks,
        num_inference_steps=args.num_inference_steps,
        iterations=args.iterations,
        seed=args.seed,
        device=device
    )

    # 7) save
    base = os.path.splitext(os.path.basename(args.video_path))[0]
    out_path = os.path.join(args.out_dir, f"{base}_removed_512.mp4")
    print(f"[INFO] saving to: {out_path}")
    save_video(out_frames, out_path, fps=args.fps)
    print("[DONE]")


if __name__ == "__main__":
    main()
