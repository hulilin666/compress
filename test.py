import torch
import compressai.zoo as zoo
from PIL import Image
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import numpy as np

# 选择模型的质量级别和优化指标
quality = 3
metric = 'mse'

# 载入模型
model = zoo.bmshj2018_factorized(quality=quality, metric=metric, pretrained=True, progress=True)

# 将模型移动到指定设备（如 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 将模型设置为评估模式
model.eval()

# 定义图像预处理转换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载图像
image_path = "kodak\\train\\kodim18.png"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# 进行推理
with torch.no_grad():
    output = model(image)
    compressed = model.compress(image)

# 获取重建图像
reconstructed_image = output["x_hat"]

# 将图像从 Tensor 转换为 NumPy 数组
original_image_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
reconstructed_image_np = reconstructed_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
reconstructed_image_np = np.clip(reconstructed_image_np, 0, 1)

# 计算 PSNR
mse = torch.mean((image - reconstructed_image) ** 2)
psnr = 10 * math.log10(1 / mse.item())

# 计算 BPP
y_likelihoods = output["likelihoods"]["y"]
total_bits = torch.sum(torch.log2(y_likelihoods)).item() * -1
num_pixels = image.size(2) * image.size(3)
bpp = total_bits / num_pixels
real_bpp = sum(len(s[0]) for s in compressed["strings"]) * 8.0 / num_pixels

print(f'PSNR: {psnr:.2f} dB')
print(f'BPP: {bpp:.4f}, Real BPP: {real_bpp:.4f}')

# 可视化图像
plt.figure(figsize=(10, 5))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(original_image_np)
plt.title("Original Image")
plt.axis("off")

# 显示重建图像
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image_np)
plt.title(f"Reconstructed Image\nPSNR: {psnr:.2f} dB, BPP: {bpp:.4f}")
plt.axis("off")

plt.show()