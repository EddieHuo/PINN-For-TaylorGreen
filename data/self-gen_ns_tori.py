import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time  # 用来显示运行时间

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from jax_cfd.spectral import utils as spectral_utils

from jax import vmap, jit

# 物理参数
viscosity = 1e-2  # 粘性系数
max_velocity = 3  # 最大速度
grid = grids.Grid((1024, 1024), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))  # 定义模拟的网格
dt = 5e-4  # 时间步长

# 时间步进函数
smooth = True  # 是否使用抗锯齿
step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.NavierStokes2D(viscosity, grid, smooth=smooth), dt)

outer_steps = 201  # 外部步数
inner_steps = 100  # 内部步数

# 生成模拟轨迹的函数
trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

# 生成初始速度场
print("正在生成初始速度场...")
v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(0), grid, max_velocity, 2)
vorticity0 = cfd.finite_differences.curl_2d(v0).data
vorticity_hat0 = jnp.fft.rfftn(vorticity0)
print("初始涡度场生成完成！")

# 运行模拟
print(f"开始模拟，总步数: {outer_steps} * {inner_steps} 步，每 {inner_steps} 步保存一个时间帧...")
start_time = time()
_, trajectory = trajectory_fn(vorticity_hat0)
end_time = time()
print(f"模拟完成！耗时: {end_time - start_time:.2f} 秒")

# 逆 FFT 得到涡度场
w = jnp.fft.irfftn(trajectory, axes=(1, 2))
print(f"涡度场形状: {w.shape}")

# 计算速度场
start_time = time()
print("正在计算速度场...")
velocity_solve = spectral_utils.vorticity_to_velocity(grid)
u_hat, v_hat = vmap(velocity_solve)(trajectory)
u = vmap(jnp.fft.irfftn)(u_hat)
v = vmap(jnp.fft.irfftn)(v_hat)
end_time = time()
print(f"速度场计算完成！耗时: {end_time - start_time:.2f} 秒")
print(f"速度场 u 形状: {u.shape}, v 形状: {v.shape}")

# 生成坐标轴
x = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
y = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
t = dt * jnp.arange(outer_steps) * inner_steps
print(f"x 形状: {x.shape}, y 形状: {y.shape}, t 形状: {t.shape}")

# 获取初始时刻的物理量
u0 = u[0, :, :]
v0 = v[0, :, :]
w0 = w[0, :, :]
print(f"初始时刻 u0 形状: {u0.shape}, v0 形状: {v0.shape}, w0 形状: {w0.shape}")

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
res = 8  # 分辨率
downsampled_data = {
    'u': u[:, ::res, ::res],
    'v': v[:, ::res, ::res],
    'w': w[:, ::res, ::res],
    'x': x[::res],
    'y': y[::res],
    't': t,
    'viscosity': viscosity
}
print(
    f"降采样后的数据形状:\n"
    f"u: {downsampled_data['u'].shape}\n"
    f"v: {downsampled_data['v'].shape}\n"
    f"w: {downsampled_data['w'].shape}\n"
    f"x: {downsampled_data['x'].shape}\n"
    f"y: {downsampled_data['y'].shape}\n"
    f"t: {downsampled_data['t'].shape}"
)

# 保存降采样后的数据
print("正在保存数据到 ns_tori2.npy...")
jnp.save('ns_tori2.npy', downsampled_data)
print("数据保存完成！")