import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time

# 物理参数
viscosity = 1e-2  # 粘性系数
grid_size = 1024  # 网格大小
domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))  # 定义域
grid = jnp.meshgrid(jnp.linspace(domain[0][0], domain[0][1], grid_size),
                    jnp.linspace(domain[1][0], domain[1][1], grid_size),
                    indexing='ij')
x, y = grid

# 时间参数
t_end = 10.0  # 模拟结束时间
num_steps = 101  # 时间步数
dt = t_end / num_steps

# 泰勒 - 格林涡解析解函数
def taylor_green_solution(x, y, t, nu):
    """
    计算泰勒 - 格林涡的解析解。

    参数:
    x (jnp.ndarray): x 坐标网格
    y (jnp.ndarray): y 坐标网格
    t (float): 时间
    nu (float): 粘性系数

    返回:
    u (jnp.ndarray): x 方向速度场
    v (jnp.ndarray): y 方向速度场
    w (jnp.ndarray): 涡度场
    """
    u = jnp.cos(x) * jnp.sin(y) * jnp.exp(-2 * nu * t)
    v = -jnp.sin(x) * jnp.cos(y) * jnp.exp(-2 * nu * t)
    w = -2 * jnp.cos(x) * jnp.cos(y) * jnp.exp(-2 * nu * t)
    p = -1/4 * (jnp.cos(2 * x) + jnp.cos(2 * y))* jnp.exp(-4 * nu * t)
    return u, v, w, p
    

# 生成不同时间步的解析解
u_all = []
v_all = []
w_all = []
p_all = []
for i in range(num_steps):
    t = i * dt
    u, v, w ,p= taylor_green_solution(x, y, t, viscosity)
    u_all.append(u)
    v_all.append(v)
    w_all.append(w)
    p_all.append(p)

u_all = jnp.stack(u_all)
v_all = jnp.stack(v_all)
w_all = jnp.stack(w_all)
p_all = jnp.stack(p_all)

# 生成坐标轴
t = jnp.linspace(0, t_end, num_steps)

# 获取初始时刻的物理量
u0 = u_all[0, :, :]
v0 = v_all[0, :, :]
w0 = w_all[0, :, :]
p0 = p_all[0, :, :]

# 保存的数据
data = {
    'w': w_all,
    'u': u_all,
    'v': v_all,
    'p': p_all,  
    'u0': u0,
    'v0': v0,
    'w0': w0,
    'p0': p0,
    'x': x[:, 0],
    'y': y[0, :],
    't': t,
    'viscosity': viscosity
}

# 降采样
res = 8  # 分辨率
downsampled_data = {
    'u': u_all[:, ::res, ::res],
    'v': v_all[:, ::res, ::res],
    'w': w_all[:, ::res, ::res],
    'p': p_all[:, ::res, ::res],
    'u0': u0[::res, ::res],
    'v0': v0[::res, ::res],
    'w0': w0[::res, ::res],
    'p0': p0[::res, ::res],
    'x': x[::res, 0],
    'y': y[0, ::res],
    't': t,
    'viscosity': viscosity
}
print(
    f"降采样后的数据形状:\n"
    f"u: {downsampled_data['u'].shape}\n"
    f"v: {downsampled_data['v'].shape}\n"
    f"w: {downsampled_data['w'].shape}\n"
    f"p: {downsampled_data['p'].shape}\n"
    f"u0: {downsampled_data['u0'].shape}\n"
    f"v0: {downsampled_data['v0'].shape}\n"
    f"w0: {downsampled_data['w0'].shape}\n"
    f"p0: {downsampled_data['p0'].shape}\n"
    f"x: {downsampled_data['x'].shape}\n"
    f"y: {downsampled_data['y'].shape}\n"
    f"t: {downsampled_data['t'].shape}"
)

# 保存降采样后的数据
print("正在保存数据到 ns_tori_taylor_green.npy...")
jnp.save('ns_tori_taylor_green-uvp+1.npy', downsampled_data)
print("数据保存完成！")