import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 加载数据
data = np.load('ns_tori_taylor_green-uvp.npy', allow_pickle=True).item()

# 提取物理量和时间信息
u = data['u']
v = data['v']
w = data['w']
p = data['p']
t = data['t']
x = data['x']
y = data['y']

# 创建网格
X, Y = np.meshgrid(x, y)

# 计算 u 和 v 合并的流场速度大小
speed = np.sqrt(u**2 + v**2)

# 定义物理量名称和对应的数组
quantities = {
    'u': u,
    'v': v,
    'w': w,
    'p': p,
    'speed': speed  # 新增流场速度大小
}

# 为每个物理量创建一个子图
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
axes = axes.flatten()

# 初始化图像
ims = []
for i, (name, quantity) in enumerate(quantities.items()):
    im = axes[i].imshow(quantity[0].T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].set_title(f'{name} at Time: {t[0]:.2f}')
    plt.colorbar(im, ax=axes[i], label=name)
    ims.append(im)

# 更新函数，用于动画
def update(frame):
    for i, (name, quantity) in enumerate(quantities.items()):
        # 更新图像数据
        ims[i].set_data(quantity[frame].T)
        # 更新标题中的时间
        axes[i].set_title(f'{name} at Time: {t[frame]:.2f}')
    return ims

# 创建动画，将 blit 设置为 False
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=False)

# 显示动画
plt.show()

# 保存动画
ani.save('ns_tori_taylor_green_all.gif', writer='imagemagick', fps=20)