# import os
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
# from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
# from huggingface_hub import snapshot_download
# from transformers import AutoTokenizer, CLIPTextModel

# tokenizer = AutoTokenizer.from_pretrained(
#     pretrained_model_name_or_path="/root/autodl-tmp/GaussianEditor-master/st21", subfolder="tokenizer"
# )
# text_encoder = CLIPTextModel.from_pretrained(
#     pretrained_model_name_or_path="/root/autodl-tmp/GaussianEditor-master/st21",
#     subfolder="text_encoder",
#     device_map="auto",
# )kaggle config set -n path -v /root/autodl-tmp/GaussianEditor-master/.cache

# import kagglehub
# #KGAT_fb137e64571d65bcfebdfe2f048022b9
# #export KAGGLE_API_TOKEN=KGAT_fb137e64571d65bcfebdfe2f048022b9
# # Download latest version。export KAGGLE_HOME=/root/autodl-tmp/GaussianEditor-master/.cache
# path = kagglehub.dataset_download("jinnywjy/tanks-and-temple-m60-colmap-preprocessed",path='.kaggle')

# print("Path to dataset files:", path)
# from plyfile import PlyData, PlyElement
# import numpy as np

# def load_ply(path):
#     plydata = PlyData.read(path)

#     xyz = np.stack(
#         (
#             np.asarray(plydata.elements[0]["x"]),
#             np.asarray(plydata.elements[0]["y"]),
#             np.asarray(plydata.elements[0]["z"]),
#         ),
#         axis=1,
#     )
#     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

#     features_dc = np.zeros((xyz.shape[0], 3, 1))
#     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
#     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
#     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

#     extra_f_names = [
#         p.name
#         for p in plydata.elements[0].properties
#         if p.name.startswith("f_rest_")
#     ]
#     extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
#     assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
#     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
#     for idx, attr_name in enumerate(extra_f_names):
#         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
#     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
#     features_extra = features_extra.reshape(
#         (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
#     )

#     scale_names = [
#         p.name
#         for p in plydata.elements[0].properties
#         if p.name.startswith("scale_")
#     ]
#     scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
#     scales = np.zeros((xyz.shape[0], len(scale_names)))
#     for idx, attr_name in enumerate(scale_names):
#         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

#     rot_names = [
#         p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
#     ]
#     rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
#     rots = np.zeros((xyz.shape[0], len(rot_names)))
#     for idx, attr_name in enumerate(rot_names):
#         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

#     self._xyz = nn.Parameter(
#         torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
#     )
#     self._features_dc = nn.Parameter(
#         torch.tensor(features_dc, dtype=torch.float, device="cuda")
#         .transpose(1, 2)
#         .contiguous()
#         .requires_grad_(True)
#     )
#     self._features_rest = nn.Parameter(
#         torch.tensor(features_extra, dtype=torch.float, device="cuda")
#         .transpose(1, 2)
#         .contiguous()
#         .requires_grad_(True)
#     )
#     self._opacity = nn.Parameter(
#         torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
#             True
#         )
#     )
#     self._scaling = nn.Parameter(
#         torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
#     )
#     self._rotation = nn.Parameter(
#         torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
#     )

#     self.active_sh_degree = self.max_sh_degree

# if __name__=='__main__':
#     load_ply("./dataset/tank/sample.ply")

import cv2
import os
import glob

def images_to_mp4(image_folder, output_file, fps=30):
    """
    将图片文件夹转换为MP4视频
    
    参数:
    image_folder: 图片文件夹路径
    output_file: 输出视频文件路径（如：output.mp4）
    fps: 帧率（每秒帧数），默认30
    """
    
    # 获取所有.JPG图片文件
    image_files = glob.glob(os.path.join(image_folder, "*.JPG"))
    
    # 按文件名排序，确保正确的顺序
    image_files.sort()
    
    if not image_files:
        print("在指定文件夹中未找到.JPG图片文件！")
        return False
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张图片获取尺寸
    img = cv2.imread(image_files[0])
    if img is None:
        print(f"无法读取图片：{image_files[0]}")
        return False
    
    height, width = img.shape[:2]
    
    # 输出文件
    if not output_file.endswith('.mp4'):
        output_file = output_file.rsplit('.', 1)[0] + '.mp4'
    
    # 尝试不同的编码器
    codecs_to_try = ['mp4v', 'avc1', 'XVID', 'MJPG']
    
    for codec in codecs_to_try:
        try:
            print(f"尝试使用 {codec} 编码器...")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            if out.isOpened():
                print(f"编码器 {codec} 可用，开始创建视频...")
                
                # 处理每一张图片
                for i, image_file in enumerate(image_files):
                    img = cv2.imread(image_file)
                    if img is not None:
                        out.write(img)
                        
                        # 显示进度
                        if (i + 1) % 10 == 0:
                            print(f"处理进度: {i + 1}/{len(image_files)}")
                    
                out.release()
                cv2.destroyAllWindows()
                
                print(f"\n视频创建完成：{output_file}")
                print(f"总帧数：{len(image_files)}")
                print(f"视频时长：{len(image_files)/fps:.2f} 秒")
                
                # 检查文件大小
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file) / (1024 * 1024)
                    print(f"文件大小：{file_size:.2f} MB")
                
                return True
            else:
                print(f"编码器 {codec} 不可用，尝试下一个...")
                
        except Exception as e:
            print(f"编码器 {codec} 失败：{e}")
    
    print("所有编码器都失败了！")
    return False


if __name__ == "__main__":
    # 设置你的图片文件夹路径
    image_folder = "/root/autodl-tmp/GaussianEditor-master/dataset/bonsai/images"
    output_file = "output.mp4"
    fps = 30
    
    # 转换为视频
    images_to_mp4(image_folder, output_file, fps)