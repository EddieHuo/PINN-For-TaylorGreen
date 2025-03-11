# Simplified_Navier–Stokes flow in a 2D torus(TaylorGreenVortex)

## Problem Setup 

没有完整的NS方程，只加入了以下简化的NS方程。

在`/data`文件夹下的`TG-self.py`文件中，定义了一个名为 `TaylorGreenVortex` 的类，用于生成Taylor-Green涡旋的初始条件及其解析解。

The partial differential equation is defined as

$$\begin{aligned}
w_t +\mathbf{u} \cdot \nabla w &= \frac{1}{\text{Re}} \Delta w,   \quad \text{ in }  [0, T] \times \Omega,  \\
\nabla \cdot \mathbf{u}  &=0,  \quad \text{ in }  [0, T] \times \Omega, \\
w(0, x, y) &=w_{0}(x, y),   \quad \text{ in }  \Omega,
\end{aligned}$$

For this example, we set Re=100 and aim to simulate the system up to T=10.
