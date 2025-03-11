import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 加载数据
# data = np.load('data/ns_tori1.npy', allow_pickle=True).item()
data = np.load('ns_tori_taylor_green10.npy', allow_pickle=True).item()

# 提取涡度场和时间信息
w = data['w']
t = data['t']
x = data['x']
y = data['y']

# 创建网格
X, Y = np.meshgrid(x, y)

# 初始化图形
fig, ax = plt.subplots()
im = ax.imshow(w[0].T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
# 初始标题使用正确的时间
ax.set_title(f'Time: {t[0]:.2f}')
plt.colorbar(im, label='Vorticity')

# 更新函数，用于动画
def update(frame):
    # 更新图像数据
    im.set_data(w[frame].T)
    # 更新标题中的时间
    title = ax.set_title(f'Time: {t[frame]:.2f}')
    print(f'frame: {frame}')
    print(f't: {t[frame]}')
    return im, title  # 返回标题对象以确保标题更新

# 创建动画，将 blit 设置为 False
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=False)

# 显示动画
plt.show()

# 保存动画
ani.save('ns_tori_taylor_green-fast.gif', writer='imagemagick', fps=20)