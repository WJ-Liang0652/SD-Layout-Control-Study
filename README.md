# Stable Diffusion 空间布局控制实战评测

本仓库包含了针对文本生成图像模型（Stable Diffusion v1.5）进行空间几何控制（Layout Control）的实战研究。

## 🚀 项目目标
实现根据文本描述（Prompt）与显式边界框（Bounding Box），在图像指定空间坐标内精准生成指定物体的功能。

## 🛠️ 技术路线
* **Base Model**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
* **Control Pipeline**: `StableDiffusionControlNetPipeline`
* **Conditioning**: ControlNet Canny 边缘检测 (`lllyasviel/sd-controlnet-canny`)
* **Optimization**: UniPCMultistepScheduler 调度器加速推理，结合 Attention Slicing 优化显存。

## 📊 10-Case 批量评测结果与定量分析
通过构建涵盖多元化物体、不同方位及尺寸的 10 组测试集（测试结果保存在 `output_results/` 文件夹中），独立发现了当前基于 **Canny 刚性线条进行空间约束** 的核心局限性：

1. **物理实体误判现象 (Case 1, 2, 10)**：模型易将抽象的 bounding box 边界框误解为物体的物理轮廓。例如在生成时钟和猫咪时，在边界框处强行生成了方形的实体外壳。
2. **柔性语义逃逸隐患 (Case 5, 8)**：当文本语义（如老鹰展翅、圆形网球）与死板的矩形线条发生刚性冲突时，扩散模型在去噪过程中倾向于优先满足文本权重，导致物体在框外生成，而框内被敷衍为水波或网线。

## 💡 科研思考
传统的显式线条约束易破坏原生扩散模型的自注意力机制。更优的解法应倾向于在 U-Net 的 Cross-Attention（交叉注意力机制）层直接进行潜在空间的“能量注入”或注意力矩阵操纵（如 Training-Free Layout Control 机制），在不引入生硬线条的前提下实现完美的非侵入式布局引导。