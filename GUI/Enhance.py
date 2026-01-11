import torch

from threestudio.utils.misc import get_device, step_check, dilate_mask, erode_mask, fill_closed_areas
from threestudio.utils.perceptual import PerceptualLoss
import ui_utils
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from PIL import Image
from torchvision.transforms import ToTensor
# Diffusion model (cached) + prompts + edited_frames + training config

class EnhanceGuidance:
    def __init__(self, guidance, gaussian, origin_frames, text_prompt, per_editing_step, edit_begin_step,
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
        self.to_tensor = ToTensor()
        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )()
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())


    def __call__(self, rendering, view_index, step):
        self.gaussian.update_learning_rate(step)
        img = rendering
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.clamp(0.0, 1.0)
        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.per_editing_step > 0
                and self.edit_begin_step
                < step
                < self.edit_until_step
                and step % self.per_editing_step == 0
        ):
            img_01 = img * 2.0 - 1.0

            result = self.guidance(
                prompt=("Fix the purple cloth: remove the black stain/dark patch and restore clean purple fabric with matching weave texture, consistent color and lighting, seamless."),
                negative_prompt=(
                    "black stain, dark patch, shadow blob, burn mark, hole, wood grain, brown, seams, halo"
                ),
                image=img_01,
                controlnet_conditioning_image=img,
                width=512,
                height=512,
                strength=0.5,
                guidance_scale=4.5,
                controlnet_conditioning_scale=0.3,
                generator=torch.manual_seed(0),
                num_inference_steps=24,
                eta=0.0,
            ).images[0]
            result.save('output.png')

            self.edit_frames[view_index] = self.to_tensor(result).to("cuda")[None].permute(0,2,3,1) # 1 H W C
            self.train_frustums[view_index].remove()
            self.train_frustums[view_index] = ui_utils.new_frustums(view_index, self.train_frames[view_index],
                                                                    self.cams[view_index], self.edit_frames[view_index], self.visible, self.server)
            # print("edited image index", cur_index)

        gt_image = self.edit_frames[view_index].permute(0,3,1,2)

        loss = self.lambda_l1 * torch.nn.functional.l1_loss(img, gt_image) + \
               self.lambda_p * self.perceptual_loss(img,
                                                    gt_image, ).sum()

        # anchor loss
        if (
                self.lambda_anchor_color > 0
                or self.lambda_anchor_geo > 0
                or self.lambda_anchor_scale > 0
                or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss += self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                    self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                    self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                    self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        return loss

