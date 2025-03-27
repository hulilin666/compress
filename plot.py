import math
from matplotlib import pyplot as plt
import numpy as np
from model import Network
import torch
import torchvision.transforms as transforms
from PIL import Image
from compressai.zoo import image_models
import os
import compressai.zoo as zoo

# model = image_models["bmshj2018-factorized"](quality=3)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # 载入模型权重
# checkpoint_path = "checkpoint_best_loss.pth.tar"
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint["state_dict"])

quality = 3
metric = 'mse'

# 载入模型
model = zoo.bmshj2018_factorized(quality=quality, metric=metric, pretrained=True, progress=True)

# 将模型移动到指定设备（如 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 将模型设置为评估模式
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

# 载入图像
image_path = "kodak\\test\kodim19.png"
# 获取原始图片文件大小（字节）
original_image_size = os.path.getsize(image_path)
print(f"原始图片文件大小: {original_image_size} 字节")

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# 进行推理
with torch.no_grad():
    output = model(image)
    compressed = model.compress(image)

# 获取重建图像
reconstructed_image = output["x_hat"]

# 将图像从Tensor转换为NumPy数组
original_image_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
reconstructed_image_np = reconstructed_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
reconstructed_image_np = np.clip(reconstructed_image_np, 0, 1)
# 计算PSNR
mse = torch.mean((image - reconstructed_image) ** 2)
psnr = 10 * math.log10(1 / mse.item())

# 计算BPP
y_likelihoods = output["likelihoods"]["y"]
total_bits = torch.sum(torch.log2(y_likelihoods)).item() * -1
num_pixels = image.size(2) * image.size(3)
bpp = total_bits / num_pixels
real_bpp = sum(len(s[0]) for s in compressed["strings"]) * 8.0 / num_pixels
num_bytes = sum(len(s[0]) for s in compressed["strings"])
print(f'bpp: {bpp:.4f}, real bpp: {real_bpp:.4f}')
print(f'压缩后字节数: {num_bytes} 字节')

# 计算原始图片的 bpp
original_bpp = original_image_size * 8 / num_pixels
print(f'原始图片 bpp: {original_bpp:.4f}')

# 计算压缩倍率
compression_ratio = original_image_size / num_bytes
print(f'压缩倍率: {compression_ratio:.2f}')

# 按通道计算比特数
channel_bits = []
for channel in range(y_likelihoods.size(1)):
    channel_likelihood = y_likelihoods[:, channel, :, :]
    channel_bits.append(torch.sum(torch.log2(channel_likelihood)).item() * -1)

# 绘制每个通道的比特数图表
plt.figure(figsize=(12, 6))

# 显示图像
plt.subplot(1, 2, 1)
plt.imshow(original_image_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image_np)
plt.title(f"Reconstructed Image\nPSNR: {psnr:.2f} dB, BPP: {bpp:.4f}")
plt.axis("off")

plt.figure(figsize=(8, 6))
plt.bar(range(len(channel_bits)), channel_bits)
plt.xlabel('Channel')
plt.ylabel('Bits')
plt.title('Bits per Channel')
plt.xticks(range(len(channel_bits)))
plt.show()