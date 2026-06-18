import os
# 1. 指定国内镜像源
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# 创建结果保存目录
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# 2. 核心辅助函数：绘制黑底白框控制图
def create_bbox_hint(width=512, height=512, bbox=[128, 128, 384, 384]):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
    return Image.fromarray(canvas)

# 3. 核心辅助函数：将控制图与结果图横向拼接（方便后续在 GitHub README 中展示）
def image_grid(img1, img2, x_pad=10):
    w, h = img1.size
    # 创建一个带间距的画布
    grid = Image.new('RGB', (w + w + x_pad, h), color=(255, 255, 255))
    grid.paste(img1, (0, 0))
    grid.paste(img2, (w + x_pad, 0))
    return grid

# 4. 精心设计 10 组多元化的空间控制测试集（模拟真实场景测试）
test_cases = [
    {"prompt": "a professional photo of a red apple, isolated clean background", "bbox": [128, 128, 384, 384]}, # 中央大苹果
    {"prompt": "a cute fluffy white cat sleeping, cinematic lighting", "bbox": [50, 250, 300, 480]},        # 左下角睡猫
    {"prompt": "a modern yellow ceramic coffee mug, high detailed", "bbox": [250, 100, 480, 350]},       # 右上角马克杯
    {"prompt": "a vintage brown leather football, studio light", "bbox": [300, 300, 500, 500]},          # 右下角橄榄球
    {"prompt": "a majestic bald eagle flying, photorealistic, clear sky", "bbox": [50, 50, 450, 200]},     # 顶部飞鹰
    {"prompt": "a sleek black sports car, side view, dark background", "bbox": [100, 200, 480, 400]},     # 中下部跑车
    {"prompt": "a delicious chocolate cupcake with a cherry on top", "bbox": [180, 150, 330, 400]},       # 偏左长条形蛋糕
    {"prompt": "a bright green tennis ball on the grass, close up", "bbox": [50, 300, 200, 450]},         # 左下角小网球
    {"prompt": "a simple wooden chair, minimalistic design", "bbox": [200, 200, 320, 480]},               # 中央偏下椅子
    {"prompt": "a round blue wall clock, modern style", "bbox": [150, 50, 360, 260]}                      # 中上部圆钟
]

# 5. 初始化模型（这次运行会直接秒开，因为上一把已经缓存好了）
print("正在加载控制模型与主管道...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# 6. 开始自动化批量循环推理
print(f"开始批量跑测，共 {len(test_cases)} 组样本...")
for i, case in enumerate(test_cases):
    print(f"\n[正在处理第 {i+1}/10 组] Prompt: {case['prompt']}")
    
    # 生成控制 Hint
    hint = create_bbox_hint(bbox=case["bbox"])
    
    # 扩散推理生成图像
    output_image = pipe(
        case["prompt"],
        image=hint,
        num_inference_steps=20,
        controlnet_conditioning_scale=1.2
    ).images[0]
    
    # 将控制白框和生成的画面拼在一起
    comparison_result = image_grid(hint, output_image)
    
    # 保存结果
    save_path = os.path.join(output_dir, f"case_{i+1}_comparison.png")
    comparison_result.save(save_path)
    print(f"成功保存对比图至: {save_path}")

print("\n🎉 所有 10 组实验批量运行完毕！成果已保存在 output_results 文件夹中。")