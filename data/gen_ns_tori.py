import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from jax_cfd.spectral import utils as spectral_utils

from jax import vmap, jit

# 物理参数
viscosity = 1e-2  # 粘性系数
max_velocity = 3  # 最大速度
grid = grids.Grid((1024, 1024), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))  # 1024x1024 网格
dt = 5e-4  # 时间步长

# 时间步进函数
smooth = True  # 是否使用抗锯齿
# 时间步进函数，使用了谱方法的 crank_nicolson_rk4 函数，用于求解非线性对流方程  
# 输入：spectral.equations.NavierStokes2D(viscosity, grid, smooth=smooth)
# 其中 viscosity 是粘性系数，grid 是网格，smooth 是是否使用抗锯齿  
# 输出：step_fn，用于求解非线性对流方程的时间步进函数
step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.NavierStokes2D(viscosity, grid, smooth=smooth), dt)

outer_steps = 201  # 外部步数
inner_steps = 100  # 内部步数

# 生成模拟轨迹的函数
# 输入：step_fn，用于求解非线性对流方程的时间步进函数
# 输出：trajectory_fn，用于求解非线性对流方程的时间步进函数
trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)


# 生成初始速度场
#cfd.initial_conditions.filtered_velocity_field ：
# 这是 jax_cfd 库中的一个函数，用于生成经过过滤的速度场。
# 过滤的目的通常是为了确保生成的速度场具有特定的属性，例如在频谱上的分布特征等。

v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(0), grid, max_velocity, 2)

vorticity0 = cfd.finite_differences.curl_2d(v0).data

vorticity_hat0 = jnp.fft.rfftn(vorticity0)

# 运行模拟
_, trajectory = trajectory_fn(vorticity_hat0)

# 逆 FFT 得到涡度场
w = jnp.fft.irfftn(trajectory, axes=(1, 2))

# 计算速度场
velocity_solve = spectral_utils.vorticity_to_velocity(grid)
u_hat, v_hat = vmap(velocity_solve)(trajectory)
u = vmap(jnp.fft.irfftn)(u_hat)
v = vmap(jnp.fft.irfftn)(v_hat)

# 生成坐标轴
x = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # 生成 x 轴坐标 0 到 2π 范围内的等间距坐标点
y = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # 生成 y 轴坐标 0 到 2π 范围内的等间距坐标点
t = dt * jnp.arange(outer_steps) * inner_steps # 生成时间轴坐标，从 0 到 (outer_steps-1) * inner_steps * dt 范围内的等间距坐标点

# 获取初始时刻的物理量
u0 = u[0, :, :]
v0 = v[0, :, :]
w0 = w[0, :, :]

# 保存的数据
data = {
    'w': w,
    'u': u,
    'v': v,
    'u0': u0,
    'v0': v0,
    'w0': w0,
    'x': x,
    'y': y,
    't': t,
    'viscosity': viscosity
}

# 降采样
res = 8  # 降采样分辨率
downsampled_data = {
    'u': u[:, ::res, ::res],
    'v': v[:, ::res, ::res],
    'w': w[:, ::res, ::res],
    'x': x[::res],
    'y': y[::res],
    't': t,
    'viscosity': viscosity
}

# 确保降采样后的数据形状正确
print(f"Downsampled data shapes:")
print(f"u: {downsampled_data['u'].shape}")
print(f"v: {downsampled_data['v'].shape}")
print(f"w: {downsampled_data['w'].shape}")
print(f"x: {downsampled_data['x'].shape}")
print(f"y: {downsampled_data['y'].shape}")
print(f"t: {downsampled_data['t'].shape}")

# 保存降采样后的数据
print("Saving data to ns_tori1.npy")
jnp.save('ns_tori1.npy', downsampled_data)
