#python webui.py --gs_source ./dataset/bicycle/point_cloud/iteration_30000/point_cloud.ply --colmap_dir ./dataset/bicycle/
import time
import numpy as np
import torch
import torchvision
import copy
import rembg
from gaussiansplatting.scene.colmap_loader import qvec2rotmat
from gaussiansplatting.scene.cameras import Simple_Camera
from threestudio.utils.dpt import DPT
from torchvision.ops import masks_to_boxes
from gaussiansplatting.utils.graphics_utils import fov2focal
import torch.nn.functional as F
import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from PIL import Image
import ImageReward as RM
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from kornia.geometry.quaternion import Quaternion

# import threestudio
# import os
# from threestudio.systems.base import BaseLift3DSystem
# from pathlib import Path
# import subprocess
# import rembg
# from threestudio.utils.clip_metrics import ClipSimilarity

# from threestudio.utils.lama import InpaintModel
# from threestudio.utils.ops import binary_cross_entropy
from threestudio.utils.typing import *
from threestudio.utils.transform import rotate_gaussians
from gaussiansplatting.gaussian_renderer import render, point_cloud_render
from gaussiansplatting.scene import GaussianModel
from threestudio.utils.perceptual import PerceptualLoss

from gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)

# from gaussiansplatting.utils.graphics_utils import fov2focal
from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

# from gaussiansplatting.utils.general_utils import inverse_sigmoid
# from gaussiansplatting.gaussian_renderer import camera2rasterizer
from argparse import ArgumentParser
from threestudio.utils.misc import (
    get_device,
    step_check,
    dilate_mask,
    erode_mask,
    fill_closed_areas,
)
from threestudio.utils.sam import LangSAMTextSegmentor
from threestudio.utils.camera import camera_ray_sample_points, project, unproject, unproject2
from utils.camera_proximity_utils import find_nearby_camera

# from threestudio.utils.dpt import DPT
# from threestudio.utils.config import parse_structured
from gaussiansplatting.scene.camera_scene import CamScene
import math
from GUI.EditGuidance import EditGuidance
from GUI.EditGuidance2 import EditGuidance2
from GUI.DelGuidance import DelGuidance
from GUI.Enhance import EnhanceGuidance

# from GUI.AddGuidance import AddGuidance
import os
import random
import ui_utils

import datetime
import subprocess
from pathlib import Path
from threestudio.utils.transform import (
    rotate_gaussians,
    rotate_gaussians_obj,
    translate_gaussians,
    translate_gaussians_obj,
    scale_gaussians,
    scale_gaussians_obj,
    default_model_mtx,
)


class WebUI:
    def __init__(self, cfg) -> None:
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.port = cfg.port
        # training cfg

        self.use_sam = False
        self.guidance = None
        self.stop_training = False
        self.inpaint_end_flag = False
        self.scale_depth = True
        self.depth_end_flag = False
        self.seg_scale = True
        self.seg_scale_end = False
        # from original system初始化高斯模型
        self.points3d = []
        self.cam_sam_2dpoint=[] #获取当前相机位置下用户点击的2D分割点
        self.cam_sam_2dpoint_fcam=[]

        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        # front end related 相机 & 背景 & 语义分组
        self.colmap_cameras = None
        self.render_cameras = None

        # diffusion model
        self.ip2p = None
        self.ctn_ip2p = None
        self.ctn_inpaint = None
        self.ctn_ip2p = None
        self.training = False

        #如果有 COLMAP，就加载相机 & 场景尺度
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {} #存编辑中的帧图像
        self.origin_frames = {} #原始渲染帧（修改前）
        self.masks_2D = {} #2D mask 图（比如 SAM segment 出来的对象 mask）
        self.text_segmentor = LangSAMTextSegmentor().to(get_device()) #一种“文字 + SAM” 的语义分割器
        self.sam_predictor = self.text_segmentor.model.sam #具体负责 SAM 推理的模型
        self.sam_predictor.is_image_set = True
        self.sam_features = {} #缓存各帧的特征（比如图像编码结果），复用加速
        self.semantic_gauassian_masks = {} #高斯级别的语义 mask
        self.semantic_gauassian_masks["ALL"] = torch.ones_like(self.gaussian._opacity)

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        # status
        self.display_semantic_mask = False #在 viewer 中是否显示语义 mask
        self.display_point_prompt = False #是否显示 SAM 的点提示

        self.viewer_need_update = False #参数改变后需要触发一次重渲染/重训练
        self.system_need_update = False
        self.inpaint_again = True #补洞操作是否需要重新运行
        self.scale_depth = True #深度缩放操作是否需要重新运行

        self.server = viser.ViserServer(port=self.port)
        self.add_theme()
        self.draw_flag = True
        #这些都是暂存值
        self.user_add_obj=None #存放用户加载的物体
        self.user_add_startidx=None #存放用户加入物体的索引
        self.user_add_objlen=None #存放用户加入物体的点数量
        self.user_add_initpos=None #存放初始物体位置
        self.user_add_initR=None #存放初始物体旋转
        self.user_add_initT=None #存放初始物体平移

        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=2048
            )

            self.FoV_slider = self.server.add_gui_slider(
                "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            )

            self.fps = self.server.add_gui_text(
                "FPS", initial_value="-1", disabled=True
            )
            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "comp_rgb",
                ],
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False #勾选后可能会在 2D 面板显示当前渲染帧
            )

        with self.server.add_gui_folder("Semantic Tracing"):
            self.sam_enabled = self.server.add_gui_checkbox(
                "Enable SAM", #开关文本/点提示分割
                initial_value=False,
            )
            self.add_sam_points = self.server.add_gui_checkbox(
                "Add SAM Points", initial_value=False #如果勾上，可以在 2D 画面上点击添加 point prompt 给 SAM
            )
            self.save_firstframe_points = self.server.add_gui_checkbox(
                "Save First Frame SAM Points", initial_value=False #
            )
            self.sam_group_name = self.server.add_gui_text(
                "SAM Group Name", initial_value="table" #为当前分割出的物体起一个语义组名字，比如 "table"
            )
            self.clear_sam_pins = self.server.add_gui_button(
                "Clear SAM Pins", #清空所有点提示
            )
            self.text_seg_prompt = self.server.add_gui_text(
                "Text Seg Prompt", initial_value="a bike" #文本提示，比如 “a bike”，喂给 LangSAM 做语义分割
            )
            self.semantic_groups = self.server.add_gui_dropdown(
                "Semantic Group", #当前选择查看/操作的语义组，初始只有 "ALL"，后面会添加新语义组名
                options=["ALL"],
            )

            self.seg_cam_num = self.server.add_gui_slider(
                "Seg Camera Nums", min=1, max=200, step=1, initial_value=24 #语义追踪时要用几台相机。从 24 个视角做 2D 分割，再融合成 3D 语义 mask
            )

            self.mask_thres = self.server.add_gui_slider( #语义 mask 阈值（置信度 > threshold 才算该类）
                "Seg Threshold", min=0.2, max=0.99999, step=0.00001, initial_value=0.7, visible=False
            )

            self.show_semantic_mask = self.server.add_gui_checkbox(#是否在渲染里叠加显示某个语义组的 mask（比如高亮显示被选中的物体）
                "Show Semantic Mask", initial_value=False
            )
            self.seg_scale_end_button = self.server.add_gui_button(
                "End Seg Scale!",
                visible=False,
            )
            self.submit_seg_prompt = self.server.add_gui_button("Tracing Begin!") #开始语义追踪的按钮

        with self.server.add_gui_folder("Edit Setting"):
            self.edit_type = self.server.add_gui_dropdown(
                "Edit Type", ("Edit", "Delete_base_image", "Delete_base_video", "Add")
            )
            self.guidance_type = self.server.add_gui_dropdown(
                "Guidance Type", ("InstructPix2Pix", "ControlNet-Pix2Pix")
            )
            self.edit_frame_show = self.server.add_gui_checkbox( #是否显示编辑前/后的 2D 帧
                "Show Edit Frame", initial_value=True, visible=False
            )
            self.edit_text = self.server.add_gui_text( #编辑文本，比如 “turn the car red”，传给编辑模型
                "Text",
                initial_value="",
                visible=True,
            )
            self.reward_text = self.server.add_gui_text( #得分文本
                "Reward Text",
                initial_value="",
                visible=True,
            )
            self.draw_bbox = self.server.add_gui_checkbox( #是否画框
                "Draw Bounding Box", initial_value=False, visible=False
            )
            #框的两角坐标
            self.left_up = self.server.add_gui_vector2(
                "Left UP",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )
            self.right_down = self.server.add_gui_vector2(
                "Right Down",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )

            self.inpaint_seed = self.server.add_gui_slider( #随机种子，控制生成结果变化
                "Inpaint Seed", min=0, max=1000, step=1, initial_value=0, visible=False
            )

            self.refine_text = self.server.add_gui_text( #进一步 refine inpaint 结果的文本提示
                "Refine Text",
                initial_value="",
                visible=False,
            )
            self.inpaint_end = self.server.add_gui_button( #用户点这个说明 2D inpaint 阶段结束，可以把修改反投影到 3D 高斯上
                "End 2D Inpainting!",
                visible=False,
            )

            self.depth_scaler = self.server.add_gui_slider( #对某个 region 的深度进行缩放（例如让物体离相机更近/更远）
                "Depth Scale", min=0.0, max=5.0, step=0.01, initial_value=1.0, visible=False
            )
            self.depth_end = self.server.add_gui_button(
                "End Depth Scale!",
                visible=False,
            )
            self.edit_begin_button = self.server.add_gui_button("Edit Begin!")
            self.edit_end_button = self.server.add_gui_button(
                "End Editing!", visible=False
            )

            #加入坦克物体
            self.addpreobj = self.server.add_gui_button("Add tankM60",visible=True)
            self.draw_points = self.server.add_gui_checkbox( #是否画框
                "Draw Points", initial_value=False, visible=True
            )

            self.user_add = self.server.add_gui_checkbox(
                "UserAdd", initial_value=True #勾选后可能会在 2D 面板显示当前渲染帧
            )

            self.center= self.server.add_gui_vector2(
                "Center",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )
            self.user_scale = self.server.add_gui_slider(
                "User_Scale", min=0, max=100, step=0.1, initial_value=1
            )

            self.user_input_rotation_x=self.server.add_gui_slider(
                "User_Rotation_x", min=-180, max=180, step=1, initial_value=0
            )

            self.user_input_rotation_y=self.server.add_gui_slider(
                "User_Rotation_y", min=-180, max=180, step=1, initial_value=0
            )

            self.user_input_rotation_z=self.server.add_gui_slider(
                "User_Rotation_z", min=-180, max=180, step=1, initial_value=0
            )

            self.translate_x=self.server.add_gui_slider(
                "Translate_x", min=-10, max=10, step=0.01, initial_value=0
            )

            self.translate_y=self.server.add_gui_slider(
                "Translate_y", min=-10, max=10, step=0.01, initial_value=0
            )

            self.translate_z=self.server.add_gui_slider(
                "Translate_z", min=-10, max=10, step=0.01, initial_value=0
            )

            self.deletepreobj = self.server.add_gui_button("Delete tankM60",visible=True)

            with self.server.add_gui_folder("Advanced Options"):
                self.edit_cam_num = self.server.add_gui_slider( #编辑训练时，每轮用多少视角
                    "Camera Num", min=1, max=200, step=1, initial_value=48
                )
                self.edit_train_steps = self.server.add_gui_slider( #本轮编辑的总步数
                    "Total Step", min=0, max=5000, step=100, initial_value=1500
                )
                self.densify_until_step = self.server.add_gui_slider( #densification 截止步（超过这个 step 就不再加点）
                    "Densify Until Step",
                    min=0,
                    max=5000,
                    step=50,
                    initial_value=1300,
                )

                self.densification_interval = self.server.add_gui_slider( #每隔多少步执行一次 densify_and_prune
                    "Densify Interval",
                    min=25,
                    max=1000,
                    step=25,
                    initial_value=100,
                )
                self.max_densify_percent = self.server.add_gui_slider( #densify 的最多点数比例（防止点数爆炸）
                    "Max Densify Percent",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    initial_value=0.01,
                )
                self.min_opacity = self.server.add_gui_slider( #prune 时的最小不透明度阈值（太透明的点删掉）
                    "Min Opacity",
                    min=0.0,
                    max=0.1,
                    step=0.0001,
                    initial_value=0.005,
                )

                self.per_editing_step = self.server.add_gui_slider( #每隔多少步应用一次编辑引导（比如 diffusion guidance）
                    "Edit Interval", min=4, max=48, step=1, initial_value=10
                )
                #编辑引导生效的步数区间
                self.edit_begin_step = self.server.add_gui_slider(
                    "Edit Begin Step", min=0, max=5000, step=100, initial_value=0
                )
                self.edit_until_step = self.server.add_gui_slider(
                    "Edit Until Step", min=0, max=5000, step=100, initial_value=1000
                )

                self.inpaint_scale = self.server.add_gui_slider(
                    "Inpaint Scale", min=0.1, max=10, step=0.1, initial_value=1, visible=False
                )

                self.mask_dilate = self.server.add_gui_slider(
                    "Mask Dilate", min=1, max=30, step=1, initial_value=15, visible=False
                )
                self.fix_holes = self.server.add_gui_checkbox(
                    "Fix Holes", initial_value=True, visible=False
                )
                with self.server.add_gui_folder("Learning Rate Scaler"):
                    self.gs_lr_scaler = self.server.add_gui_slider(
                        "XYZ LR Init", min=0.0, max=10.0, step=0.1, initial_value=3.0
                    )
                    self.gs_lr_end_scaler = self.server.add_gui_slider(
                        "XYZ LR End", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.color_lr_scaler = self.server.add_gui_slider(
                        "Color LR", min=0.0, max=10.0, step=0.1, initial_value=3.0
                    )
                    self.opacity_lr_scaler = self.server.add_gui_slider(
                        "Opacity LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.scaling_lr_scaler = self.server.add_gui_slider(
                        "Scale LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.rotation_lr_scaler = self.server.add_gui_slider(
                        "Rotation LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )

                with self.server.add_gui_folder("Loss Options"):
                    self.lambda_l1 = self.server.add_gui_slider(
                        "Lambda L1", min=0, max=100, step=1, initial_value=10
                    )
                    self.lambda_p = self.server.add_gui_slider(
                        "Lambda Perceptual", min=0, max=100, step=1, initial_value=10
                    )

                    self.anchor_weight_init_g0 = self.server.add_gui_slider(
                        "Anchor Init G0", min=0., max=10., step=0.05, initial_value=0.05
                    )
                    self.anchor_weight_init = self.server.add_gui_slider(
                        "Anchor Init", min=0., max=10., step=0.05, initial_value=0.1
                    )
                    self.anchor_weight_multiplier = self.server.add_gui_slider(
                        "Anchor Multiplier", min=1., max=10., step=0.1, initial_value=1.3
                    )

                    self.lambda_anchor_color = self.server.add_gui_slider(
                        "Lambda Anchor Color", min=0, max=500, step=1, initial_value=0
                    )
                    self.lambda_anchor_geo = self.server.add_gui_slider(
                        "Lambda Anchor Geo", min=0, max=500, step=1, initial_value=50
                    )
                    self.lambda_anchor_scale = self.server.add_gui_slider(
                        "Lambda Anchor Scale", min=0, max=500, step=1, initial_value=50
                    )
                    self.lambda_anchor_opacity = self.server.add_gui_slider(
                        "Lambda Anchor Opacity", min=0, max=500, step=1, initial_value=50
                    )
                    self.anchor_term = [self.anchor_weight_init_g0, self.anchor_weight_init,
                                        self.anchor_weight_multiplier,
                                        self.lambda_anchor_color, self.lambda_anchor_geo,
                                        self.lambda_anchor_scale, self.lambda_anchor_opacity, ]

        @self.inpaint_seed.on_update
        def _(_):
            self.inpaint_again = True

        @self.depth_scaler.on_update
        def _(_):
            self.scale_depth = True

        @self.mask_thres.on_update
        def _(_):
            self.seg_scale = True


        @self.edit_type.on_update
        def _(_):
            if self.edit_type.value == "Edit":
                self.edit_text.visible = True
                self.reward_text.visible = True
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = True
                self.inpaint_scale.visible = False
                self.mask_dilate.visible = False
                self.fix_holes.visible = False
                self.per_editing_step.visible = True
                self.edit_begin_step.visible = True
                self.edit_until_step.visible = True
                self.draw_bbox.visible = False
                self.left_up.visible = False
                self.right_down.visible = False
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = True
                self.guidance_type.visible = True

            elif self.edit_type.value == "Delete_base_image":
                self.edit_text.visible = True
                self.reward_text.visible = False
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = True
                self.inpaint_scale.visible = True
                self.mask_dilate.visible = True
                self.fix_holes.visible = True
                self.edit_cam_num.value = 24
                self.densification_interval.value = 50
                self.per_editing_step.visible = False
                self.edit_begin_step.visible = False
                self.edit_until_step.visible = False
                self.draw_bbox.visible = False
                self.left_up.visible = False
                self.right_down.visible = False
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = True
                self.guidance_type.visible = False

            elif self.edit_type.value == "Delete_base_video":
                self.edit_text.visible = True
                self.reward_text.visible = False
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = True
                self.inpaint_scale.visible = True
                self.mask_dilate.visible = True
                self.fix_holes.visible = True
                self.edit_cam_num.value = 24
                self.densification_interval.value = 50
                self.per_editing_step.visible = False
                self.edit_begin_step.visible = False
                self.edit_until_step.visible = False
                self.draw_bbox.visible = False
                self.left_up.visible = False
                self.right_down.visible = False
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = True
                self.guidance_type.visible = False

            elif self.edit_type.value == "Add":
                self.edit_text.visible = True
                self.reward_text.visible = False
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = False
                self.inpaint_scale.visible = False
                self.mask_dilate.visible = False
                self.fix_holes.visible = False
                self.per_editing_step.visible = True
                self.edit_begin_step.visible = True
                self.edit_until_step.visible = True
                self.draw_bbox.visible = True
                self.left_up.visible = True
                self.right_down.visible = True
                self.center.visible = True
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = False
                self.guidance_type.visible = False

        @self.save_button.on_click
        def _(_):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")
            self.gaussian.save_ply(os.path.join("ui_result", "{}.ply".format(formatted_time)))
        @self.inpaint_end.on_click
        def _(_):
            self.inpaint_end_flag = True

        @self.seg_scale_end_button.on_click
        def _(_):
            self.seg_scale_end = True


        @self.depth_end.on_click
        def _(_):
            self.depth_end_flag = True

        @self.edit_end_button.on_click
        def _(event: viser.GuiEvent):
            self.stop_training = True

        @self.edit_begin_button.on_click
        def _(event: viser.GuiEvent):
            self.edit_begin_button.visible = False
            self.edit_end_button.visible = True
            if self.training:
                return
            self.training = True
            self.configure_optimizers()
            self.gaussian.update_anchor_term(
                anchor_weight_init_g0=self.anchor_weight_init_g0.value,
                anchor_weight_init=self.anchor_weight_init.value,
                anchor_weight_multiplier=self.anchor_weight_multiplier.value,
            )

            if self.edit_type.value == "Add":
                # self.add(self.camera)
                self.add(self.camera)
            else:
                self.edit_frame_show.visible = True

                edit_cameras, train_frames, train_frustums = ui_utils.sample_train_camera(self.colmap_cameras,
                                                                                          self.edit_cam_num.value,
                                                                                          self.server)
                if self.edit_type.value == "Edit":
                    self.edit(edit_cameras, train_frames, train_frustums)

                elif self.edit_type.value == "Delete_base_image":
                    self.delete(edit_cameras, train_frames, train_frustums)

                elif self.edit_type.value == "Delete_base_video":
                    self.delete_video(edit_cameras, train_frames, train_frustums)

                ui_utils.remove_all(train_frames)
                ui_utils.remove_all(train_frustums)
                self.edit_frame_show.visible = False
            self.guidance = None
            self.training = False
            self.gaussian.anchor_postfix()
            self.edit_begin_button.visible = True
            self.edit_end_button.visible = False

        @self.addpreobj.on_click
        def _(event: viser.GuiEvent):
            if self.training:
                return
            self.training = True
            self.configure_optimizers()
            self.gaussian.update_anchor_term(
                anchor_weight_init_g0=self.anchor_weight_init_g0.value,
                anchor_weight_init=self.anchor_weight_init.value,
                anchor_weight_multiplier=self.anchor_weight_multiplier.value,
            )
            self.add_tankM60(self.camera)
            self.guidance = None
            self.training = False
            self.gaussian.anchor_postfix()

        @self.deletepreobj.on_click
        def _(event: viser.GuiEvent):
            self.configure_optimizers()
            self.gaussian.update_anchor_term(
                anchor_weight_init_g0=self.anchor_weight_init_g0.value,
                anchor_weight_init=self.anchor_weight_init.value,
                anchor_weight_multiplier=self.anchor_weight_multiplier.value,
            )
            self.remove_gaussians_by_index()
            self.guidance = None
            self.training = False
            self.gaussian.anchor_postfix()

        ##########################
        #更新用户新加物体的位姿
        @self.user_scale.on_update
        def _(_):
            self.modify_useradd_obj(self.camera, self.user_add_obj, self.user_add_startidx, self.user_add_objlen, self.user_add_initpos, self.user_add_initR, self.user_add_initT)
            
        @self.user_input_rotation_x.on_update
        def _(_):
            self.modify_useradd_obj(self.camera, self.user_add_obj, self.user_add_startidx, self.user_add_objlen, self.user_add_initpos, self.user_add_initR, self.user_add_initT)

        @self.user_input_rotation_y.on_update
        def _(_):
            self.modify_useradd_obj(self.camera, self.user_add_obj, self.user_add_startidx, self.user_add_objlen, self.user_add_initpos, self.user_add_initR, self.user_add_initT)

        @self.user_input_rotation_z.on_update
        def _(_):
            self.modify_useradd_obj(self.camera, self.user_add_obj, self.user_add_startidx, self.user_add_objlen, self.user_add_initpos, self.user_add_initR, self.user_add_initT)

        @self.translate_x.on_update
        def _(_):
            self.modify_useradd_obj(self.camera, self.user_add_obj, self.user_add_startidx, self.user_add_objlen, self.user_add_initpos, self.user_add_initR, self.user_add_initT)

        @self.translate_y.on_update
        def _(_):
            self.modify_useradd_obj(self.camera, self.user_add_obj, self.user_add_startidx, self.user_add_objlen, self.user_add_initpos, self.user_add_initR, self.user_add_initT)

        @self.translate_z.on_update
        def _(_):
            self.modify_useradd_obj(self.camera, self.user_add_obj, self.user_add_startidx, self.user_add_objlen, self.user_add_initpos, self.user_add_initR, self.user_add_initT)
        #############################
            
        @self.submit_seg_prompt.on_click
        def _(_):
            if not self.sam_enabled.value:
                text_prompt = self.text_seg_prompt.value
                print("[Segmentation Prompt]", text_prompt)
                _, semantic_gaussian_mask = self.update_mask(text_prompt)
            else:
                text_prompt = self.sam_group_name.value
                # buggy here, if self.sam_enabled == True, will raise strange errors. (Maybe caused by multi-threading access to the same SAM model)
                self.sam_enabled.value = False
                self.add_sam_points.value = False
                # breakpoint()
                _, semantic_gaussian_mask = self.update_sam_mask_with_point_prompt(
                    save_mask=True
                )

            self.semantic_gauassian_masks[text_prompt] = semantic_gaussian_mask
            if text_prompt not in self.semantic_groups.options:
                self.semantic_groups.options += (text_prompt,)
            self.semantic_groups.value = text_prompt

        @self.semantic_groups.on_update
        def _(_):
            semantic_mask = self.semantic_gauassian_masks[self.semantic_groups.value]
            self.gaussian.set_mask(semantic_mask)
            self.gaussian.apply_grad_mask(semantic_mask)

        @self.edit_frame_show.on_update
        def _(_):
            if self.guidance is not None:
                for _ in self.guidance.train_frames:
                    _.visible = self.edit_frame_show.value
                for _ in self.guidance.train_frustums:
                    _.visible = self.edit_frame_show.value
                self.guidance.visible = self.edit_frame_show.value

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

        @self.server.on_scene_click
        def _(pointer):
            self.click_cb(pointer)

        @self.clear_sam_pins.on_click
        def _(_):
            self.clear_points3d()

    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.edit_train_steps.value,
            lr_scaler = self.gs_lr_scaler.value,
            lr_final_scaler = self.gs_lr_end_scaler.value,
            color_lr_scaler = self.color_lr_scaler.value,
            opacity_lr_scaler = self.opacity_lr_scaler.value,
            scaling_lr_scaler = self.scaling_lr_scaler.value,
            rotation_lr_scaler = self.rotation_lr_scaler.value,

        )
        opt = OmegaConf.create(vars(opt))
        # opt.update(self.training_args)
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)

    def render(
        self,
        cam,
        local=False,
        sam=False,
        train=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        semantic_map = render(
            cam,
            self.gaussian,
            self.pipe,
            self.background_tensor,
            override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
        )["render"]
        semantic_map = torch.norm(semantic_map, dim=0)
        semantic_map = semantic_map > 0.0  # 1, H, W
        semantic_map_viz = image.detach().clone()  # C, H, W
        semantic_map_viz = semantic_map_viz.permute(1, 2, 0)  # 3 512 512 to 512 512 3
        semantic_map_viz[semantic_map] = 0.50 * semantic_map_viz[
            semantic_map
        ] + 0.50 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
        semantic_map_viz = semantic_map_viz.permute(2, 0, 1)  # 512 512 3 to 3 512 512

        render_pkg["sam_masks"] = []
        render_pkg["point2ds"] = []
        if sam:
            if hasattr(self, "points3d") and len(self.points3d) > 0:
                sam_output = self.sam_predict(image, cam)
                if sam_output is not None:
                    render_pkg["sam_masks"].append(sam_output[0])
                    render_pkg["point2ds"].append(sam_output[1])

        self.gaussian.localize = False  # reverse

        render_pkg["semantic"] = semantic_map_viz[None]
        render_pkg["masks"] = semantic_map[None]  # 1, 1, H, W

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C

        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }

    @torch.no_grad()
    def update_mask(self, text_prompt) -> None:

        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        total_view_num = len(self.colmap_cameras)
        random.seed(0)  # make sure same views
        view_index = random.sample(
            range(0, total_view_num),
            min(total_view_num, self.seg_cam_num.value),
        )

        for idx in tqdm(view_index):
            cur_cam = self.colmap_cameras[idx]
            this_frame = render(
                cur_cam, self.gaussian, self.pipe, self.background_tensor
            )["render"]

            # breakpoint()
            # this_frame [c h w]
            this_frame = this_frame.moveaxis(0, -1)[None, ...]
            mask = self.text_segmentor(this_frame, text_prompt)[0].to(get_device())
            if self.use_sam:
                print("Using SAM")
                self.sam_features[idx] = self.sam_predictor.features

            masks.append(mask)
            self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

        weights /= weights_cnt + 1e-7
        self.seg_scale_end_button.visible = True
        self.mask_thres.visible = True
        self.show_semantic_mask.value = True
        while True:
            if self.seg_scale:
                selected_mask = weights > self.mask_thres.value
                selected_mask = selected_mask[:, 0]
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False
        return masks, selected_mask

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def click_cb(self, pointer):
        import torch.nn.functional as F
        if self.sam_enabled.value and self.add_sam_points.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos  # tuple (float, float)  W, H from 0 to 1
            click_pos = torch.tensor(click_pos)
            # if self.save_firstframe_points.value:
            #     self.cam_sam_2dpoint_fcam.append(self.camera)
            #     self.cam_sam_2dpoint.append(click_pos.clone())#保存用户的2D分割点用于后续的删除功能
            # print(f"click_pos{click_pos}")
            self.add_points3d(self.camera, click_pos)
            
            self.viwer_need_update = True
        elif self.draw_bbox.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos
            click_pos = torch.tensor(click_pos)
            cur_cam = self.camera
            if self.draw_flag:
                self.left_up.value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                self.draw_flag = False
            else:
                new_value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                if (self.left_up.value[0] < new_value[0]) and (
                    self.left_up.value[1] < new_value[1]
                ):
                    self.right_down.value = new_value
                    self.draw_flag = True
                else:
                    self.left_up.value = new_value
        elif self.draw_points.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos
            click_pos = torch.tensor(click_pos)
            cur_cam = self.camera
            self.center.value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
            self.viwer_need_update = True


    def set_system(self, system):
        self.system = system

    def clear_points3d(self):
        self.points3d = []
        self.cam_sam_2dpoint=[]
        self.cam_sam_2dpoint_fcam=[]

    def add_points3d(self, camera, points2d, update_mask=False):
        depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
            "depth_3dgs"
        ]
        unprojected_points3d = unproject2(camera, points2d, depth)
        self.points3d += unprojected_points3d.unbind(0)

        if update_mask:
            self.update_sam_mask_with_point_prompt(self.points3d)

    # no longer needed since can be extracted from langsam
    # def sam_encode_all_view(self):
    #     assert hasattr(self, "sam_predictor")
    #     self.sam_features = {}
    #     # NOTE: assuming all views have the same size
    #     for id, frame in self.origin_frames.items():
    #         # TODO: check frame dtype (float32 or uint8) and device
    #         self.sam_predictor.set_image(frame)
    #         self.sam_features[id] = self.sam_predictor.features

    @torch.no_grad()
    def update_sam_mask_with_point_prompt(
        self, points3d=None, save_mask=False, save_name="point_prompt_mask"
    ):
        points3d = points3d if points3d is not None else self.points3d
        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        total_view_num = len(self.colmap_cameras)
        random.seed(0)  # make sure same views
        view_index = random.sample(
            range(0, total_view_num),
            min(total_view_num, self.seg_cam_num.value),
        )
        for idx in tqdm(view_index):
            cur_cam = self.colmap_cameras[idx]
            assert len(points3d) > 0
            points2ds = project(cur_cam, points3d)
            img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor)[
                "render"
            ]

            self.sam_predictor.set_image(
                np.asarray(to_pil_image(img.cpu())),
            )
            self.sam_features[idx] = self.sam_predictor.features
            # print(points2ds)
            mask, _, _ = self.sam_predictor.predict(
                point_coords=points2ds.cpu().numpy(),
                point_labels=np.array([1] * points2ds.shape[0], dtype=np.int64),
                box=None,
                multimask_output=False,
            )
            mask = torch.from_numpy(mask).to(torch.bool).to(get_device())
            self.gaussian.apply_weights(
                cur_cam, weights, weights_cnt, mask.to(torch.float32)
            )
            masks.append(mask)

        weights /= weights_cnt + 1e-7

        self.seg_scale_end_button.visible = True
        self.mask_thres.visible = True
        self.show_semantic_mask.value = True
        while True:
            if self.seg_scale:
                selected_mask = weights > self.mask_thres.value
                selected_mask = selected_mask[:, 0]
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False

        if save_mask:
            for id, mask in enumerate(masks):
                mask = mask.cpu().numpy()[0, 0]
                img = Image.fromarray(mask)
                os.makedirs("tmp",exist_ok=True)
                img.save(f"./tmp/{save_name}-{id}.jpg")

        return masks, selected_mask

    @torch.no_grad()
    def sam_predict(self, image, cam):
        img = np.asarray(to_pil_image(image.cpu()))
        self.sam_predictor.set_image(img)
        if len(self.points3d) == 0:
            return
        _points2ds = project(cam, self.points3d)
        _mask, _, _ = self.sam_predictor.predict(
            point_coords=_points2ds.cpu().numpy(),
            point_labels=np.array([1] * _points2ds.shape[0], dtype=np.int64),
            box=None,
            multimask_output=False,
        )
        _mask = torch.from_numpy(_mask).to(torch.bool).to(get_device())

        return _mask.squeeze(), _points2ds

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        out_img = output[out_key][0]  # H W C
        if out_key == "comp_rgb":
            if self.show_semantic_mask.value:
                out_img = output["semantic"][0].moveaxis(0, -1)
        elif out_key == "masks":
            out_img = output["masks"][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        if self.sam_enabled.value:
            if "sam_masks" in output and len(output["sam_masks"]) > 0:
                try:
                    out_img = torchvision.utils.draw_segmentation_masks(
                        out_img, output["sam_masks"][0]
                    )

                    out_img = torchvision.utils.draw_keypoints(
                        out_img,
                        output["point2ds"][0][None, ...],
                        colors="blue",
                        radius=5,
                    )
                except Exception as e:
                    print(e)

        if (
            self.draw_bbox.value
            and self.draw_flag
            and (self.left_up.value[0] < self.right_down.value[0])
            and (self.left_up.value[1] < self.right_down.value[1])
        ):
            out_img[
                :,
                self.left_up.value[1] : self.right_down.value[1],
                self.left_up.value[0] : self.right_down.value[0],
            ] = 0

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self):
        while True:
            # if self.viewer_need_update:
            self.update_viewer()
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        output = self.render(gs_camera, sam=self.sam_enabled.value)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")

    def delete_video(self, edit_cameras, train_frames, train_frustums):
        # import json
        # from decord import VideoReader, cpu

        # def save_simple_mp4(tensor_list, output_path, is_mask=False):
        #     if not tensor_list:
        #         return
        #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #     height, width = 512, 512
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     fps = 30
        #     video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        #     for tensor in tensor_list:
        #         frame = tensor[0].detach().cpu().numpy()  # 去掉批次维度
        #         print(f"frame_shape:{frame.shape}")
        #         if is_mask:
        #             frame = (frame * 255).astype(np.uint8)
        #             if len(frame.shape) == 2:  # 灰度图
        #                 frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #         else:
        #             if frame.dtype != np.uint8:
        #                 if frame.min() < 0:  # 假设是[-1, 1]范围
        #                     frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        #                 else:  # 假设是[0, 1]范围
        #                     frame = (frame * 255).astype(np.uint8)
        #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #         video.write(frame)
        #     video.release()
        #     print(f"视频已保存: {output_path}")
        # if len(self.cam_sam_2dpoint) == 0:
        #     return 
        # no_prune_frames=[]
        
        # firstcam=self.cam_sam_2dpoint_fcam[0]
        # firstcam_points=self.cam_sam_2dpoint
        # print(f"cam_samkeys:{self.cam_sam_2dpoint_fcam[0]}")
        # print(f"cam_samvalues:{self.cam_sam_2dpoint}")
        
        # first_frameimg=self.render(firstcam)["comp_rgb"]
        # img = first_frameimg.permute(0, 3, 1, 2).contiguous().float()
        # img_512 = F.interpolate(img, size=(512, 512), mode="bilinear", align_corners=False)
        # img_512 = img_512.permute(0, 2, 3, 1).contiguous()
        # no_prune_frames.append(img_512)

        # #把分割的2d写入json中，方便后续传参
        # # points_list = [p.tolist() for p in firstcam_points]  # [[x,y], [x,y], ...]
        # # points_json = json.dumps(points_list)

        # for idx, cam in enumerate(edit_cameras):
        #     res = self.render(cam)
        #     rgb = res["comp_rgb"]
        #     no_prune_frames.append(rgb)
        
        dist_thres = (
            self.inpaint_scale.value * self.cameras_extent * self.gaussian.percent_dense
        )
        valid_remaining_idx = self.gaussian.get_near_gaussians_by_mask(
            self.gaussian.mask, dist_thres*2
        )
        # Prune and update mask to valid_remaining_idx
        self.gaussian.prune_with_mask(new_mask=valid_remaining_idx)

        # inpaint_2D_mask, origin_frames = self.render_all_view_with_mask(
        #     edit_cameras, train_frames, train_frustums
        # )
        #inpaint_2D_mask是需要的mask，origin_frames是原始的rgb图像
        # random_seed = 42
        # video_length=len(no_prune_frames)
        
        # print(f"video_length:{video_length}")

        # save_simple_mp4(no_prune_frames, "./outputs/origin.mp4", is_mask=False)
        # cmd = [
        #         "conda","run","-n","remover",
        #         "python","pipe_rem.py",
        #         "--video_path", "/root/autodl-tmp/GaussianEditor-master/outputs/origin.mp4",
        #         "--video_length", str(video_length),
        #         "--points_json", points_json
        #     ]
        # subprocess.run(cmd, check=True, cwd="/root/autodl-tmp/GaussianEditor-master/remover/gradio_demo")
        
        # def video_to_tensor_list_skip_first(video_path, size=512):
        #     vr = VideoReader(video_path, ctx=cpu(0))
        #     out = []
        #     for i in range(1, len(vr)):  # 跳过第0帧
        #         frame = vr[i].asnumpy()                 # HWC, uint8
        #         frame = cv2.resize(frame, (size, size)) # 统一到 512x512
        #         t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (3,H,W), float[0,1]
        #         out.append(t)
        #     return out
        # remove_frames = video_to_tensor_list_skip_first("./outputs/out.mp4", size=512)
        # remove_frames=[]
        # view_index_stack = list(range(len(edit_cameras)))
        # for step in tqdm(range(self.edit_train_steps.value)):
        #     if not view_index_stack:
        #         view_index_stack = list(range(len(edit_cameras)))
        #     view_index = random.choice(view_index_stack)
        #     view_index_stack.remove(view_index)
        #     rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]
        #     self.gaussian.update_learning_rate(step)
            
        #     remove_frames[view_index] = self.to_tensor(out[view_index]).to("cuda")[None].permute(0,2,3,1)
        #     self.train_frustums[view_index].remove()
        #     visible=True
        #     self.train_frustums[view_index] = ui_utils.new_frustums(view_index, train_frames[view_index],
        #                                                         edit_cameras[view_index], remove_frames[view_index],
        #                                                         visible, self.server)
        #     gt_image = remove_frames[view_index]
        #     perceptual_loss = PerceptualLoss().eval().to(get_device())
        #     lambda_l1=self.lambda_l1.value
        #     lambda_p=self.lambda_p.value
        #     lambda_anchor_color=self.lambda_anchor_color.value
        #     lambda_anchor_geo=self.lambda_anchor_geo.value
        #     lambda_anchor_scale=self.lambda_anchor_scale.value
        #     lambda_anchor_opacity=self.lambda_anchor_opacity.value

        #     loss = lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
        #        lambda_p * perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
        #                                                 gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()
        #     if (self.lambda_anchor_color > 0 or self.lambda_anchor_geo > 0
        #         or self.lambda_anchor_scale > 0
        #         or self.lambda_anchor_opacity > 0):
        #         anchor_out = self.gaussian.anchor_loss()
        #         loss += lambda_anchor_color * anchor_out['loss_anchor_color'] + \
        #                 lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
        #                 lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
        #                 lambda_anchor_scale * anchor_out['loss_anchor_scale']
            
        #     loss.backward()

        #     self.densify_and_prune(step)

        #     self.gaussian.optimizer.step()
        #     self.gaussian.optimizer.zero_grad(set_to_none=True)
        #     if self.stop_training:
        #         self.stop_training = False
        #         return
            
    def delete(self, edit_cameras, train_frames, train_frustums):
        if not self.ctn_inpaint:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DDIMScheduler,
            )

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            pipe.enable_model_cpu_offload()

            self.ctn_inpaint = pipe
            self.ctn_inpaint.set_progress_bar_config(disable=True)
            self.ctn_inpaint.safety_checker = None

        num_channels_latents = self.ctn_inpaint.vae.config.latent_channels
        shape = (
            1,
            num_channels_latents,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
        )

        latents = torch.zeros(shape, dtype=torch.float16, device="cuda")
        # origin_frames=[]
        # for idx, cam in enumerate(edit_cameras):
        #     res = self.render(cam)
        #     rgb = res["comp_rgb"]
        #     origin_frames.append(rgb)

        # dist_thres = (
        #     self.inpaint_scale.value * self.cameras_extent * self.gaussian.percent_dense
        # )
        # valid_remaining_idx = self.gaussian.get_near_gaussians_by_mask(
        #     self.gaussian.mask, dist_thres*2
        # )
        # # Prune and update mask to valid_remaining_idx
        # self.gaussian.prune_with_mask(new_mask=valid_remaining_idx)

        inpaint_2D_mask, origin_frames = self.render_all_view_with_mask(
            edit_cameras, train_frames, train_frustums
        )

        self.guidance = DelGuidance(
            guidance=self.ctn_inpaint,
            latents=latents,
            gaussian=self.gaussian,
            text_prompt=self.edit_text.value,
            lambda_l1=self.lambda_l1.value,
            lambda_p=self.lambda_p.value,
            lambda_anchor_color=self.lambda_anchor_color.value,
            lambda_anchor_geo=self.lambda_anchor_geo.value,
            lambda_anchor_scale=self.lambda_anchor_scale.value,
            lambda_anchor_opacity=self.lambda_anchor_opacity.value,
            train_frames=train_frames,
            train_frustums=train_frustums,
            cams=edit_cameras,
            server=self.server,
        )

        view_index_stack = list(range(len(edit_cameras)))
        for step in tqdm(range(self.edit_train_steps.value)):
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(
                rendering,
                origin_frames[view_index],
                inpaint_2D_mask[view_index],
                view_index,
                step,
            )
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return

    #使用局部mask（inpaint）的编辑函数
    # def edit(self, edit_cameras, train_frames, train_frustums):
    #     from diffusers import (
    #         DiffusionPipeline,
    #         ControlNetModel,
    #         DDIMScheduler,
    #     )

    #     controlnet = ControlNetModel.from_pretrained(
    #         'lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16
    #     )
    #     pipe = DiffusionPipeline.from_pretrained(
    #         "runwayml/stable-diffusion-v1-5",
    #         custom_pipeline="stable_diffusion_controlnet_img2img",
    #         controlnet=controlnet,
    #         torch_dtype=torch.float16,
    #     ).to("cuda") 
    #     pipe.enable_model_cpu_offload()

    #     self.ctn_inpaint = pipe
    #     self.ctn_inpaint.set_progress_bar_config(disable=True)
    #     self.ctn_inpaint.safety_checker = None

    #     origin_frames = self.render_cameras_list(edit_cameras)

    #     self.guidance = EnhanceGuidance(
    #         guidance=self.ctn_inpaint,
    #         gaussian=self.gaussian,
    #         origin_frames=origin_frames,
    #         text_prompt=self.edit_text.value,
    #         per_editing_step=self.per_editing_step.value,
    #         edit_begin_step=self.edit_begin_step.value,
    #         edit_until_step=self.edit_until_step.value,
    #         lambda_l1=self.lambda_l1.value,
    #         lambda_p=self.lambda_p.value,
    #         lambda_anchor_color=self.lambda_anchor_color.value,
    #         lambda_anchor_geo=self.lambda_anchor_geo.value,
    #         lambda_anchor_scale=self.lambda_anchor_scale.value,
    #         lambda_anchor_opacity=self.lambda_anchor_opacity.value,
    #         train_frames=train_frames,
    #         train_frustums=train_frustums,
    #         cams=edit_cameras,
    #         server=self.server,
    #     )

    #     view_index_stack = list(range(len(edit_cameras)))
    #     for step in tqdm(range(self.edit_train_steps.value)):
    #         if not view_index_stack:
    #             view_index_stack = list(range(len(edit_cameras)))
    #         view_index = random.choice(view_index_stack)
    #         view_index_stack.remove(view_index)

    #         rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

    #         loss = self.guidance(rendering, view_index, step)
    #         loss.backward()

    #         self.densify_and_prune(step)

    #         self.gaussian.optimizer.step()
    #         self.gaussian.optimizer.zero_grad(set_to_none=True)
    #         if self.stop_training:
    #             self.stop_training = False
    #             return

    # In local edit, the whole gaussian don't need to be visible for edit 经过多视图一致融合的编辑函数
    # def edit(self, edit_cameras, train_frames, train_frustums):
    #     from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
    #     if self.guidance_type.value == "InstructPix2Pix":
    #         if not self.ip2p:
    #             from threestudio.models.guidance.instructpix2pix_guidance import (
    #                 InstructPix2PixGuidance,
    #             )

    #             self.ip2p = InstructPix2PixGuidance(
    #                 OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98})
    #             )
    #         cur_2D_guidance = self.ip2p
    #         print("using InstructPix2Pix!")
    #     elif self.guidance_type.value == "ControlNet-Pix2Pix":
    #         if not self.ctn_ip2p:
    #             from threestudio.models.guidance.controlnet_guidance import (
    #                 ControlNetGuidance,
    #             )

    #             self.ctn_ip2p = ControlNetGuidance(
    #                 OmegaConf.create({"min_step_percent": 0.05,
    #                                   "max_step_percent": 0.8,
    #                                     "control_type": "p2p"})
    #             )
    #         cur_2D_guidance = self.ctn_ip2p
    #         print("using ControlNet-InstructPix2Pix!")

    #     camera_dist_order, _ = find_nearby_camera(edit_cameras)

    #     #主要拿到mask
    #     frame_2D_masks, origin_frames, rendered_depth_list = self.render_all_view_with_mask_only(edit_cameras)
    #     # inpaint_2D_mask, origin_frames = self.render_all_view_with_mask(
    #     #     edit_cameras, train_frames, train_frustums
    #     # )

    #     prompt_utils = StableDiffusionPromptProcessor(
    #                 {
    #                     "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    #                     "prompt": self.edit_text.value,
    #                 })()

    #     with torch.no_grad():
    #         edited_image_list = []
    #         for step, origin_frame in enumerate(tqdm(origin_frames, desc="Initial editing progress")):
    #             origin_frame = origin_frame.to(device="cuda")

    #             edited_image = cur_2D_guidance(
    #                 origin_frame,
    #                 origin_frame,
    #                 prompt_utils,
    #             )
    #             # safety check
    #             # print(f"edited_image:{edited_image['edit_images'].shape}") #torch.Size([1, 512, 512, 3])
    #             edited_image_list.append(edited_image["edit_images"])

    #     self.guidance = EditGuidance(
    #         guidance=cur_2D_guidance,
    #         gaussian=self.gaussian,
    #         origin_frames=origin_frames,
    #         frame_2D_masks=frame_2D_masks,
    #         text_prompt=prompt_utils,
    #         reward_text=self.reward_text.value,
    #         per_editing_step=self.per_editing_step.value,
    #         edit_begin_step=self.edit_begin_step.value,
    #         edit_until_step=self.edit_until_step.value,
    #         camera_dist_order=camera_dist_order, #相机之间相似排序(N,N)
    #         rendered_depth_list=rendered_depth_list,
    #         lambda_l1=self.lambda_l1.value,
    #         lambda_p=self.lambda_p.value,
    #         lambda_anchor_color=self.lambda_anchor_color.value,
    #         lambda_anchor_geo=self.lambda_anchor_geo.value,
    #         lambda_anchor_scale=self.lambda_anchor_scale.value,
    #         lambda_anchor_opacity=self.lambda_anchor_opacity.value,
    #         train_frames=train_frames,
    #         train_frustums=train_frustums,
    #         cams=edit_cameras,
    #         server=self.server,
    #     )

    #     view_index_stack = list(range(len(edit_cameras)))
    #     for step in tqdm(range(self.edit_train_steps.value)):
    #         if not view_index_stack:
    #             view_index_stack = list(range(len(edit_cameras)))
    #         view_index = random.choice(view_index_stack)
    #         view_index_stack.remove(view_index)

    #         rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

    #         loss = self.guidance(rendering, view_index, step, edited_image_list)
    #         loss.backward()

    #         self.densify_and_prune(step)

    #         self.gaussian.optimizer.step()
    #         self.gaussian.optimizer.zero_grad(set_to_none=True)
    #         if self.stop_training:
    #             self.stop_training = False
    #             return

    def edit(self, edit_cameras, train_frames, train_frustums):
        if self.guidance_type.value == "InstructPix2Pix":
            if not self.ip2p:
                from threestudio.models.guidance.instructpix2pix_guidance import (
                    InstructPix2PixGuidance,
                )

                self.ip2p = InstructPix2PixGuidance(
                    OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98})
                )
            cur_2D_guidance = self.ip2p
            print("using InstructPix2Pix!")
        elif self.guidance_type.value == "ControlNet-Pix2Pix":
            if not self.ctn_ip2p:
                from threestudio.models.guidance.controlnet_guidance import (
                    ControlNetGuidance,
                )

                self.ctn_ip2p = ControlNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.05,
                                      "max_step_percent": 0.8,
                                        "control_type": "p2p"})
                )
            cur_2D_guidance = self.ctn_ip2p
            print("using ControlNet-InstructPix2Pix!")

        origin_frames = self.render_cameras_list(edit_cameras)
        self.guidance = EditGuidance2(
            guidance=cur_2D_guidance,
            gaussian=self.gaussian,
            origin_frames=origin_frames,
            text_prompt=self.edit_text.value,
            reward_text=self.reward_text.value,
            per_editing_step=self.per_editing_step.value,
            edit_begin_step=self.edit_begin_step.value,
            edit_until_step=self.edit_until_step.value,
            lambda_l1=self.lambda_l1.value,
            lambda_p=self.lambda_p.value,
            lambda_anchor_color=self.lambda_anchor_color.value,
            lambda_anchor_geo=self.lambda_anchor_geo.value,
            lambda_anchor_scale=self.lambda_anchor_scale.value,
            lambda_anchor_opacity=self.lambda_anchor_opacity.value,
            train_frames=train_frames,
            train_frustums=train_frustums,
            cams=edit_cameras,
            server=self.server,
        )
        view_index_stack = list(range(len(edit_cameras)))
        for step in tqdm(range(self.edit_train_steps.value)):
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(rendering, view_index, step)
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return

    @torch.no_grad()
    def add(self, cam):
        self.draw_bbox.value = False
        self.inpaint_seed.visible = True    
        self.inpaint_end.visible = True
        self.refine_text.visible = True
        self.draw_bbox.visible = False
        self.left_up.visible = False
        self.right_down.visible = False

        if not self.ctn_inpaint:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DDIMScheduler,
            )
            #这里换成自己的ControlNet
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            pipe.enable_model_cpu_offload()

            self.ctn_inpaint = pipe
            self.ctn_inpaint.set_progress_bar_config(disable=True)
            self.ctn_inpaint.safety_checker = None

        with torch.no_grad():
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)

        image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0))
        # image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0)*255)
        origin_size = image_in.size  # W, H

        frustum = None
        frame = None
        while True:
            if self.inpaint_again:
                if frustum is not None:
                    frustum.remove()
                    frame.remove()
                mask_in = torch.zeros(
                    (origin_size[1], origin_size[0]),
                    dtype=torch.float32,
                    device=get_device(),
                )  # H, W
                mask_in[
                    self.left_up.value[1] : self.right_down.value[1],
                    self.left_up.value[0] : self.right_down.value[0],
                ] = 1.0

                image_in_pil = to_pil_image(
                    ui_utils.resize_image_ctn(np.asarray(image_in), 512)
                )
                mask_in = to_pil_image(mask_in)  # .resize((1024, 1024))
                mask_in_pil = to_pil_image(
                    ui_utils.resize_image_ctn(np.asarray(mask_in)[..., None], 512)
                )

                image = np.array(image_in_pil.convert("RGB")).astype(np.float32) / 255.0
                image_mask = (
                    np.array(mask_in_pil.convert("L")).astype(np.float32) / 255.0
                )

                image[image_mask > 0.5] = -1.0  # set as masked pixel
                image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
                control_image = torch.from_numpy(image).to("cuda")
                generator = torch.Generator(device="cuda").manual_seed(
                    self.inpaint_seed.value
                )
                out = self.ctn_inpaint(
                    self.edit_text.value+", high quality, extremely detailed",
                    num_inference_steps=25,
                    generator=generator,
                    eta=1.0,
                    image=image_in_pil,
                    mask_image=mask_in_pil,
                    control_image=control_image,
                ).images[0]
                out = cv2.resize(
                    np.asarray(out),
                    origin_size,
                    interpolation=cv2.INTER_LANCZOS4
                    if out.width / origin_size[0] > 1
                    else cv2.INTER_AREA,
                )
                out = to_pil_image(out)
                frame, frustum = ui_utils.new_frustum_from_cam(
                    list(self.server.get_clients().values())[0].camera,
                    self.server,
                    np.asarray(out),
                )
                self.inpaint_again = False
            else:
                if self.stop_training:
                    self.stop_training = False
                    return
                time.sleep(0.1)
            if self.inpaint_end_flag:
                self.inpaint_end_flag = False
                break
        self.inpaint_seed.visible = False
        self.inpaint_end.visible = False
        self.edit_text.visible = False

        removed_bg = rembg.remove(out)
        inpainted_image = to_tensor(out).to("cuda")
        frustum.remove()
        frame.remove()
        frame, frustum = ui_utils.new_frustum_from_cam(
            list(self.server.get_clients().values())[0].camera,
            self.server,
            np.asarray(removed_bg),
        )

        cache_dir = Path("trellis_tmp_add").absolute().as_posix()
        os.makedirs(cache_dir, exist_ok=True)
        inpaint_path = os.path.join(cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(cache_dir, "removed_bg.png")
        gs_path = os.path.join(cache_dir, "sample.ply")
        out.save(inpaint_path)
        removed_bg.save(removed_bg_path)

        #这里直接换成TRELLIS
        p1 = subprocess.Popen(
            [
                f"{sys.prefix}/bin/python",
                "example.py",
                "--image_path",
                removed_bg_path,
                "--save_path",
                cache_dir,
            ],
            cwd="TRELLIS",
        )
        p1.wait()

        # p3 = subprocess.Popen(
        #     [
        #         f"{sys.prefix}/bin/python",
        #         "train_from_mesh.py",
        #         "--mesh",
        #         mesh_path,
        #         "--save_path",
        #         gs_path,
        #         "--prompt",
        #         self.refine_text.value,
        #     ]
        # )
        # p3.wait()
        #############################
        
        frustum.remove()
        frame.remove()

        object_mask = np.array(removed_bg)
        object_mask = object_mask[:, :, 3] > 0
        object_mask = torch.from_numpy(object_mask)
        bbox = masks_to_boxes(object_mask[None])[0].to("cuda")

        depth_estimator = DPT(get_device(), mode="depth")

        estimated_depth = depth_estimator(
            inpainted_image.moveaxis(0, -1)[None, ...]
        ).squeeze()
        # ui_utils.vis_depth(estimated_depth.cpu())
        object_center = (bbox[:2] + bbox[2:]) / 2

        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)

        object_center = (
            object_center
            - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
        ) / torch.tensor([fx, fy]).to("cuda")

        rendered_depth = render_pkg["depth_3dgs"][..., ~object_mask]

        inpainted_depth = estimated_depth[~object_mask]
        object_depth = estimated_depth[..., object_mask]

        min_object_depth = torch.quantile(object_depth, 0.05)
        max_object_depth = torch.quantile(object_depth, 0.95)
        obj_depth_scale = (max_object_depth - min_object_depth) * 1

        min_valid_depth_mask = (min_object_depth - obj_depth_scale) < inpainted_depth
        max_valid_depth_mask = inpainted_depth < (max_object_depth + obj_depth_scale)
        valid_depth_mask = torch.logical_and(min_valid_depth_mask, max_valid_depth_mask)
        valid_percent = valid_depth_mask.sum() / min_valid_depth_mask.shape[0]
        print("depth valid percent: ", valid_percent)

        rendered_depth = rendered_depth[0, valid_depth_mask]
        inpainted_depth = inpainted_depth[valid_depth_mask.squeeze()]

        ## assuming rendered_depth = a * estimated_depth + b
        y = rendered_depth
        x = inpainted_depth
        a = (torch.sum(x * y) - torch.sum(x) * torch.sum(y)) / (
            torch.sum(x**2) - torch.sum(x) ** 2
        )
        b = torch.sum(y) - a * torch.sum(x)

        z_in_cam = object_depth.min() * a + b

        self.depth_scaler.visible = True
        self.depth_end.visible = True
        self.refine_text.visible = True

        new_object_gaussian = None
        while True:
            if self.scale_depth:
                if new_object_gaussian is not None:
                    self.gaussian.prune_with_mask()
                scaled_z_in_cam = z_in_cam * self.depth_scaler.value
                x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
                T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)

                bbox = bbox.cuda()
                real_scale = (
                    (bbox[2:] - bbox[:2])
                    / torch.tensor([fx, fy], device="cuda")
                    * scaled_z_in_cam
                )

                new_object_gaussian = VanillaGaussianModel(self.gaussian.max_sh_degree)
                new_object_gaussian.load_f0_ply(gs_path)
                new_object_numspoints=new_object_gaussian._xyz.shape[0]
                print(new_object_numspoints)
                original_gaussian_numspoints=self.gaussian._xyz.shape[0]
                print(original_gaussian_numspoints)
                self.user_add_startidx=original_gaussian_numspoints
                self.user_add_objlen=new_object_numspoints

                new_object_gaussian._opacity.data = (
                    torch.ones_like(new_object_gaussian._opacity.data) * 99.99
                )

                new_object_gaussian._xyz.data -= new_object_gaussian._xyz.data.mean(
                    dim=0, keepdim=True
                )
                # rotate_gaussians(new_object_gaussian, default_model_mtx.T)

                object_scale = (
                    new_object_gaussian._xyz.data.max(dim=0)[0]
                    - new_object_gaussian._xyz.data.min(dim=0)[0]
                )[:2]

                relative_scale = (real_scale / object_scale).mean()
                print(relative_scale)

                self.user_add_obj=copy.deepcopy(new_object_gaussian)

                scale_gaussians(new_object_gaussian, relative_scale)

                new_object_gaussian._xyz.data += T_in_cam
                self.user_add_initpos=T_in_cam

                R = torch.from_numpy(cam.R).float().cuda()
                T = -R @ torch.from_numpy(cam.T).float().cuda()

                rotate_gaussians(new_object_gaussian, R)
                translate_gaussians(new_object_gaussian, T)
                self.user_add_initR=R
                self.user_add_initT=T

                self.gaussian.concat_gaussians(new_object_gaussian)
                self.scale_depth = False
            else:
                if self.stop_training:
                    self.stop_training = False
                    return
                time.sleep(0.01)
            if self.depth_end_flag:
                self.depth_end_flag = False
                break
        self.depth_scaler.visible = False
        self.depth_end.visible = False
    
    @torch.no_grad()
    def add_tankM60(self, cam):
        gs_tankm60_path="./dataset/tank/sample.ply"
        with torch.no_grad():
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0))
        origin_size = image_in.size
        origin_size=torch.tensor(origin_size,device="cuda")
        # object_center=self.center.value
        object_center = torch.tensor(self.center.value, device="cuda")
        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)
        object_center = (
            object_center
            - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
        ) / torch.tensor([fx, fy]).to("cuda")
        rendered_depth = render_pkg["depth_3dgs"]
        z_in_cam=rendered_depth.min()
        scaled_z_in_cam=z_in_cam*self.depth_scaler.value
        x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
        T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)
        real_scale=((origin_size//2)/torch.tensor([fx,fy],device="cuda")*scaled_z_in_cam)
        #开始加载坦克M60高斯模型
        print(self.gaussian.max_sh_degree)
        new_object_gaussian=VanillaGaussianModel(self.gaussian.max_sh_degree)
        new_object_gaussian.load_f0_ply(gs_tankm60_path)

        #后续用于制作索引
        new_object_numspoints=new_object_gaussian._xyz.shape[0]
        print(new_object_numspoints)
        original_gaussian_numspoints=self.gaussian._xyz.shape[0]
        print(original_gaussian_numspoints)
        self.user_add_startidx=original_gaussian_numspoints
        self.user_add_objlen=new_object_numspoints

        new_object_gaussian._opacity.data = (torch.ones_like(new_object_gaussian._opacity.data) * 99.99)
        new_object_gaussian._xyz.data -= new_object_gaussian._xyz.data.mean(dim=0, keepdim=True)
        rotate_gaussians(new_object_gaussian, default_model_mtx.T)
        object_scale = (new_object_gaussian._xyz.data.max(dim=0)[0]- new_object_gaussian._xyz.data.min(dim=0)[0])[:2]

        relative_scale = (real_scale / object_scale).mean()
        print(relative_scale)
        scale_gaussians(new_object_gaussian, relative_scale)
        #在没有移动之前赋值，确保以物体为中心的性质不变
        self.user_add_obj=copy.deepcopy(new_object_gaussian)

        new_object_gaussian._xyz.data += T_in_cam
        self.user_add_initpos=T_in_cam
        #########
        print(T_in_cam.shape)
        #########
        R = torch.from_numpy(cam.R).float().cuda()
        T = -R @ torch.from_numpy(cam.T).float().cuda()
        self.user_add_initR=R
        self.user_add_initT=T

        rotate_gaussians(new_object_gaussian, R)
        translate_gaussians(new_object_gaussian, T)
        self.gaussian.concat_gaussians(new_object_gaussian)

    @torch.no_grad()
    def modify_useradd_obj(self, cam, obj, startidx, obj_len, initpos, initR, initT):
        if self.user_add:
            new_object_gaussian=copy.deepcopy(obj)

            user_scale = torch.tensor(self.user_scale.value, device="cuda")

            rotation_x = np.radians(self.user_input_rotation_x.value)
            rotation_y = np.radians(self.user_input_rotation_y.value)
            rotation_z = np.radians(self.user_input_rotation_z.value)
            # 使用弧度的旋转矩阵
            R_x = torch.tensor([
                [1, 0, 0],
                [0, np.cos(rotation_x), -np.sin(rotation_x)],
                [0, np.sin(rotation_x), np.cos(rotation_x)]
            ], dtype=torch.float32).cuda()
            R_y = torch.tensor([
                [np.cos(rotation_y), 0, np.sin(rotation_y)],
                [0, 1, 0],
                [-np.sin(rotation_y), 0, np.cos(rotation_y)]
            ], dtype=torch.float32).cuda()
            R_z = torch.tensor([
                [np.cos(rotation_z), -np.sin(rotation_z), 0],
                [np.sin(rotation_z), np.cos(rotation_z), 0],
                [0, 0, 1]
            ], dtype=torch.float32).cuda()

            R = torch.mm(R_z, torch.mm(R_y, R_x))
            T = (torch.tensor([self.translate_x.value,self. translate_y.value, self.translate_z.value])).cuda()
            scale_gaussians(new_object_gaussian, user_scale)
            rotate_gaussians(new_object_gaussian, R)
            translate_gaussians(new_object_gaussian, T)
            #****这里是一个小细节，必须先在物体世界进行旋转，在转回场景世界*****
            new_object_gaussian._xyz.data += initpos
            rotate_gaussians(new_object_gaussian, initR)
            translate_gaussians(new_object_gaussian,initT)

            self.gaussian._scaling.data[startidx:startidx+obj_len]=new_object_gaussian._scaling.data
            self.gaussian._xyz.data[startidx:startidx+obj_len]=new_object_gaussian._xyz.data



    @torch.no_grad()
    def remove_gaussians_by_index(self):
        """删除指定索引范围的高斯点"""
        if self.user_add_startidx==None or self.user_add_objlen==None or self.user_add_initpos==None or self.user_add_obj==None:
            print("没有物体可删除")
            return
        # 存储所有需要删除索引的高斯属性
        end_idx = self.user_add_startidx + self.user_add_objlen
        indices_to_keep = torch.cat([
            torch.arange(0, self.user_add_startidx, device="cuda"),
            torch.arange(end_idx, self.gaussian._xyz.shape[0], device="cuda")
        ])
        
        # 更新所有高斯属性
        self.gaussian._xyz = self.gaussian._xyz[indices_to_keep]
        self.gaussian._scaling = self.gaussian._scaling[indices_to_keep]
        self.gaussian._rotation = self.gaussian._rotation[indices_to_keep]
        self.gaussian._opacity = self.gaussian._opacity[indices_to_keep]
        self.gaussian._features_dc = self.gaussian._features_dc[indices_to_keep]
        
        # 如果有其他属性也需要更新
        if hasattr(self.gaussian, '_features_rest'):
            self.gaussian._features_rest = self.gaussian._features_rest[indices_to_keep]
        print(f"删除高斯点: [{self.user_add_startidx}:{end_idx}], 剩余点数: {len(self.gaussian._xyz)}")
        self.user_add_startidx=None
        self.user_add_objlen=None
        self.user_add_obj=None
        self.user_add_initpos=None
        self.user_add_initR=None
        self.user_add_initT=None

    def densify_and_prune(self, step):
        if step <= self.densify_until_step.value:
            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval.value == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent.value,
                    min_opacity=self.min_opacity.value,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )

    @torch.no_grad()
    def render_cameras_list(self, edit_cameras):
        origin_frames = []
        for cam in edit_cameras:
            out = self.render(cam)["comp_rgb"]
            origin_frames.append(out)

        return origin_frames

    @torch.no_grad()
    def render_all_view_with_mask(self, edit_cameras, train_frames, train_frustums):
        inpaint_2D_mask = []
        origin_frames = []

        for idx, cam in enumerate(edit_cameras):
            res = self.render(cam)
            rgb, mask = res["comp_rgb"], res["masks"]
            mask = dilate_mask(mask.to(torch.float32), self.mask_dilate.value)
            if self.fix_holes.value:
                mask = fill_closed_areas(mask)
            inpaint_2D_mask.append(mask)
            origin_frames.append(rgb)
            train_frustums[idx].remove()
            mask_view = torch.stack([mask] * 3, dim=3)  # 1 H W C
            train_frustums[idx] = ui_utils.new_frustums(
                idx, train_frames[idx], cam, mask_view, True, self.server
            )
        return inpaint_2D_mask, origin_frames
    
    @torch.no_grad()
    def render_all_view_with_mask_only(self, edit_cameras):
        inpaint_2D_mask = []
        origin_frames = []
        rendered_depth_list = []
        for idx, cam in enumerate(edit_cameras):
            res = self.render(cam)
            rgb, mask, depth = res["comp_rgb"], res["masks"], res["depth_3dgs"]
            mask = dilate_mask(mask.to(torch.float32), self.mask_dilate.value)
            if self.fix_holes.value:
                mask = fill_closed_areas(mask)
            inpaint_2D_mask.append(mask)
            origin_frames.append(rgb)
            rendered_depth_list.append(depth)

        return inpaint_2D_mask, origin_frames, rendered_depth_list

    def add_theme(self):
        buttons = (
            TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md",
            ),
            TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/buaacyw/GaussianEditor",
            ),
            TitlebarButton(
                text="Yiwen Chen",
                icon=None,
                href="https://buaacyw.github.io/",
            ),
            TitlebarButton(
                text="Zilong Chen",
                icon=None,
                href="https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en",
            ),
        )
        image = TitlebarImage(
            image_url_light="https://github.com/buaacyw/gaussian-editor/raw/master/static/images/logo.png",
            image_alt="GaussianEditor Logo",
            href="https://buaacyw.github.io/gaussian-editor/",
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (7, 0, 8), visible=False)

        self.server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    parser.add_argument("--port", type=int, default=7000, help="Port to run the server on")

    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()
#python webui.py --colmap_dir ./dataset/bonsai --gs_source ./dataset/bonsai/point_cloud/iteration_30000/point_cloud.ply
#Remove the black stain/ghosting on the purple cloth. Fill the masked area with the same purple fabric as surrounding: identical weave texture, identical purple color tone, consistent lighting and shading, seamless transition, no change outside the mask.
#python webui.py --colmap_dir ./dataset/m60 --gs_source ./dataset/m60/point_cloud/iteration_7000/point_cloud.ply
#python webui.py --colmap_dir ./dataset/face --gs_source ./dataset/face/point_cloud.ply