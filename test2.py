import numpy as np
import matplotlib.pyplot as plt

# 定义函数 T(x)
def T(x):
    return 0.001 * np.exp(4.382 * x)

# 生成 x 值（例如从 0 到 2，步长 0.01）
x_values = np.arange(0, 1.01, 0.01)
y_values = T(x_values)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label=r"$T(x) = 0.001 \cdot e^{4.382x}$", color="blue")

# 添加标题和标签
plt.title("Exponential Growth Function: $T(x) = 0.001 \cdot e^{4.382x}$", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("T(x)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)

# 显示图形
plt.show()