import os
# 确保清理掉可能冲突的代理，并强制指定国内镜像源
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# 1. 编写边界框生成函数（使用 OpenCV 基础操作）
def create_bbox_hint(width=512, height=512, bbox=[128, 128, 384, 384]):
    """
    在纯黑背景上绘制白色矩形框作为 ControlNet 的 Canny 信号
    bbox 格式: [xmin, ymin, xmax, ymax]
    """
    # 创建黑底画布
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # 画一个白色的矩形框（颜色为 (255, 255, 255)，线条宽度设为 2）
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
    return Image.fromarray(canvas)

# 2. 加载专用的 ControlNet Canny 模型
print("正在从镜像站加载 ControlNet Canny 模型...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=torch.float16
)

# 3. 将 ControlNet 注入到 Stable Diffusion v1.5 管道中
print("正在初始化可控图像生成管道...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 4. 算法性能优化（换用更高效的调度器，并开启显存优化）
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# 5. 准备实验输入
prompt = "a vibrant professional digital art of a ripe red apple, isolated on a clean background"
# 在图像中央指定一个边界框
bbox_coordinate = [128, 128, 384, 384] 
hint_image = create_bbox_hint(bbox=bbox_coordinate)

# 将控制框保存下来，方便后续做 COCO 数据集的对比图展示
hint_image.save("bbox_hint.png") 

# 6. 运行控制推理
print("开始根据边界框条件生成图像...")
output_image = pipe(
    prompt,
    image=hint_image,
    num_inference_steps=20,
    controlnet_conditioning_scale=1.2  # 控制权重，值越大越严格遵守方框边界
).images[0]

# 7. 保存最终成果
output_image.save("bbox_output.png")
print("成功！空间控制图片已保存为 bbox_output.png")