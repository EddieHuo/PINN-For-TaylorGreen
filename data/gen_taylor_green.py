import jax
import jax.numpy as jnp
from time import time

# 泰勒格林涡解析解
def taylor_green(x, y, t, nu):
    u = jnp.cos(x) * jnp.sin(y) * jnp.exp(-2*nu*t)
    v = -jnp.sin(x) * jnp.cos(y) * jnp.exp(-2*nu*t)
    w = -2 * jnp.cos(x) * jnp.cos(y) * jnp.exp(-2*nu*t)
    return u, v, w

# 物理参数
viscosity = 1e-2
T = 10.0  # 总时间
num_steps = 201  # 与原文件保持一致
inner_steps = 100
dt = 5e-4

# 生成网格
grid_size = 1024
x = jnp.linspace(0, 2*jnp.pi, grid_size)
y = jnp.linspace(0, 2*jnp.pi, grid_size)
X, Y = jnp.meshgrid(x, y, indexing='ij')

# 生成时间轴
t = jnp.linspace(0, T, num_steps)

print("正在计算时空演化场...")
start_time = time()

# 计算所有时间步的场
u_all = jnp.stack([taylor_green(X, Y, ti, viscosity)[0] for ti in t])
v_all = jnp.stack([taylor_green(X, Y, ti, viscosity)[1] for ti in t])
w_all = jnp.stack([taylor_green(X, Y, ti, viscosity)[2] for ti in t])

# 获取初始场
u0 = u_all[0]
v0 = v_all[0]
w0 = w_all[0]

# 降采样处理
res = 8
downsampled_data = {
    'u': u_all[:, ::res, ::res],
    'v': v_all[:, ::res, ::res],
    'w': w_all[:, ::res, ::res],
    'x': x[::res],
    'y': y[::res],
    't': t,
    'viscosity': viscosity,
    'u0': u0[::res, ::res],
    'v0': v0[::res, ::res],
    'w0': w0[::res, ::res]
}

print(f"降采样数据形状:\n"
      f"u: {downsampled_data['u'].shape}\n"
      f"v: {downsampled_data['v'].shape}\n"
      f"w: {downsampled_data['w'].shape}\n"
      f"x: {downsampled_data['x'].shape}\n"
      f"y: {downsampled_data['y'].shape}\n"
      f"t: {downsampled_data['t'].shape}")

# 保存数据
print("正在保存到ns_tori_taylor_green.npy...")
jnp.save('ns_tori_taylor_green1.npy', downsampled_data)
print(f"数据保存完成! 耗时: {time()-start_time:.2f}秒")