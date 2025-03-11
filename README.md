# Navier–Stokes flow in a 2D torus

## 项目介绍
本项目旨在使用物理信息神经网络（PINN）方法来模拟二维环面中的纳维-斯托克斯流。通过PINN，我们可以将物理方程融入到神经网络的训练过程中，从而在不依赖大量标注数据的情况下求解偏微分方程。


## 问题描述

偏微分方程：

$$\begin{aligned}
w_t +\mathbf{u} \cdot \nabla w &= \frac{1}{\text{Re}} \Delta w,   \quad \text{ in }  [0, T] \times \Omega,  \\
\nabla \cdot \mathbf{u}  &=0,  \quad \text{ in }  [0, T] \times \Omega, \\
w(0, x, y) &=w_{0}(x, y),   \quad \text{ in }  \Omega,
\end{aligned}$$


对于这个例子，我们将Re设置为100，目标是在T=10内模拟。


# #结果

下面的动画显示了精确解和预测之间的比较。
模型参数可以在[Google链接](https://drive.google.com/drive/folders/1n2k2613BWWLcug3CI4i3ZQnBvgrHS1Ph?usp=drive_link)上找到。
有关损失和权重的全面记录，请访问[我们的权重和偏差仪表板](https://wandb.ai/jaxpi/ns_tori?workspace=user-)

![ns_tori](figures/ns_animation.gif)


## 使用方法

###  训练
python main.py --config=configs/default.py --workdir=./output

python main.py --config=configs/default.py --workdir=./output --mode=eval




### 切换到评估模式
在训练完成后，需要切换到评估模式以验证模型性能。切换步骤如下：

1. **修改配置文件**：
   - 将 `sota.py`(看'main.py'中选择了哪个配置文件) 中的 `config.mode` 设置为 `"eval"`：
     ```python
     config.mode = "eval"
     ```

2. **修改评估脚本路径**：
   - 在 `eval.py` 中，确保评估脚本的路径正确指向训练后的模型文件。例如：
     ```python
     model_path = "20250305-215326sota-ModifiedMlp"
     ```
3. **运行评估脚本**：
   - 在命令行中运行 `main.py`：(确保在 `main.py` 所在的目录下运行，同时已经选择了与训练时一致的配置文件)
     ```bash
     python main.py
     ```

### 注意事项
- **及时评估**：训练完成后，务必及时运行评估脚本。如果在训练后修改了模型的配置文件（如网络结构、超参数等），即使评估成功，结果也可能是错误的，无法真实反映模型的实际性能。
- **保持配置一致**：确保训练和评估时使用的配置文件一致，避免因配置差异导致评估结果不准确。
- **检查模型路径**：在评估时，确保 `model_path` 指向正确的训练模型文件。

### 配置文件说明
配置文件（如 `sota.py`）定义了模型的结构和训练参数。以下是一些关键配置项：
- `config.arch`：定义模型架构（如 `DeepONet`）及其参数（如层数、隐藏层维度等）。
- `config.optim`：定义优化器及其参数（如学习率、衰减率等）。
- `config.training`：定义训练参数（如最大训练步数、批量大小等）。
- `config.weighting`：定义损失权重的配置。


# 代码功能补充

## 自动评估功能
在 `main.py` 中，加入了自动评估功能。当训练完成后，会自动调用 `eval.py` 进行模型评估。
### 保存训练的配置文件
在 `main.py` 中，在训练完成后，会将训练的配置文件保存为 `.yaml` 文件，以便后续分析和比较。
### 训练结束自动评估功能
在 `main.py` 中，加入了训练结束自动评估功能。当训练完成后，会自动调用 `eval.py` 进行模型评估。


## eval.py生成的gif不连续问题
每整数秒的图像不连续，怀疑是因为在传入下一个时间窗口的初始值并非上一个时间窗口的最后一帧，而是第一帧
在eval.py中，增加了 对比每个时间窗口的第一帧以及与上一个时间窗口的最后一帧的误差就可以知道是否是以上问题

### 解决gif不连续问题及原因
在 `eval.py` 中:

    # Remove the last time step
    u_ref = u_ref[:-1, :]
    v_ref = v_ref[:-1, :]
    w_ref = w_ref[:-1, :]

删去最后一帧即可使gif连续

同时在生成data时

    num_steps = 101  # 时间步数  时间步数加1可以应对取时间窗口时，最后一个时间步的时间窗口无法取到
    #100是0-99，但是取时间窗口时是0-10 10-20 20-30 30-40 40-50 50-60 60-70 70-80 80-90 90-100，但是没有100
    #一定要加1，0-99因为在eval.py中加入了删去最后一帧，# Remove the last time step否则帧不连续

也一定要把时间步数加1，因为在’’eval.py中加入了删去最后一帧，# Remove the last time step，num_steps = 100删去后是0-99，num_steps = 101删去后是0-100