
import torch

from threestudio.utils.misc import get_device, step_check, dilate_mask, erode_mask, fill_closed_areas
from threestudio.utils.perceptual import PerceptualLoss
import ui_utils
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from torchvision.transforms.functional import to_pil_image
import ImageReward as RM
# Diffusion model (cached) + prompts + edited_frames + training config

def tensor_to_pil(img: torch.Tensor):
    """
    img: (1,H,W,3) or (H,W,3), float tensor in [0,1]
    """
    if img.dim() == 4:
        img = img[0]
    img = img.detach().clamp(0, 1).cpu()
    img = img.permute(2, 0, 1)  # HWC -> CHW
    return to_pil_image(img)

class EditGuidance2:
    def __init__(self, guidance, gaussian, origin_frames, text_prompt, reward_text, per_editing_step, edit_begin_step,
                 edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
                 lambda_anchor_opacity, train_frames, train_frustums, cams, server
                 ):
        self.guidance = guidance
        self.gaussian = gaussian
        self.per_editing_step = per_editing_step
        self.edit_begin_step = edit_begin_step
        self.edit_until_step = edit_until_step
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.origin_frames = origin_frames
        self.cams = cams
        self.server = server
        self.train_frames = train_frames
        self.train_frustums = train_frustums
        self.edit_frames = {}
        self.visible = True
        self.reward_model = RM.load("ImageReward-v1.0")
        self.reward_text=reward_text
        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )()
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.reward_ema_mean = 0.0
        self.reward_ema_var = 1.0
        self.reward_beta = 0.99

    def __call__(self, rendering, view_index, step):
        self.gaussian.update_learning_rate(step)

        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.per_editing_step > 0
                and self.edit_begin_step
                < step
                < self.edit_until_step
                and step % self.per_editing_step == 0
        ):
            result = self.guidance(
                rendering,
                self.origin_frames[view_index],
                self.prompt_utils,
            )
            self.edit_frames[view_index] = result["edit_images"].detach().clone() # 1 H W C
            self.train_frustums[view_index].remove()
            self.train_frustums[view_index] = ui_utils.new_frustums(view_index, self.train_frames[view_index],
                                                                    self.cams[view_index], self.edit_frames[view_index], self.visible, self.server)
            # print("edited image index", cur_index)

        gt_image = self.edit_frames[view_index]

        with torch.no_grad():
            pil_img = tensor_to_pil(gt_image)
            _, reward = self.reward_model.inference_rank(self.reward_text, [pil_img])

        with torch.no_grad():
            r = float(reward)  # ImageReward 输出是 list
            print(f"cur eidted image reward:{reward}")

        # EMA 归一化
        delta = r - self.reward_ema_mean
        self.reward_ema_mean = self.reward_beta * self.reward_ema_mean + (1 - self.reward_beta) * r
        self.reward_ema_var  = self.reward_beta * self.reward_ema_var  + (1 - self.reward_beta) * (delta ** 2)

        std = (self.reward_ema_var + 1e-6) ** 0.5
        z = (r - self.reward_ema_mean) / std

        w_min, w_max = 0.6, 1.4
        alpha = 0.4
        temperature = 1.0

        reward_weight = 1.0 + alpha * torch.tanh(torch.tensor(z / temperature, device=rendering.device))
        reward_weight = reward_weight.clamp(w_min, w_max)

        loss_edit = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
                    self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                    gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()
        
        # anchor loss
        loss_anchor = 0.0
        if (
                self.lambda_anchor_color > 0
                or self.lambda_anchor_geo > 0
                or self.lambda_anchor_scale > 0
                or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss_anchor = self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                            self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                            self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                            self.lambda_anchor_scale * anchor_out['loss_anchor_scale']
            
        loss = reward_weight * loss_edit + loss_anchor
        return loss
