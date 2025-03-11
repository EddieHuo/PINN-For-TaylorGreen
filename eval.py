import os
import matplotlib.animation as animation

from absl import logging
import ml_collections

import jax.numpy as jnp

import scipy.io
import matplotlib.pyplot as plt

import wandb

from jaxpi.utils import restore_checkpoint
import models

from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, w_ref, p_ref, t_star, x_star, y_star, nu = get_dataset()

    # # Remove the last time step
    # u_ref = u_ref[:-1, :]
    # v_ref = v_ref[:-1, :]
    # w_ref = w_ref[:-1, :]
    # p_ref = p_ref[:-1, :]

    u0 = u_ref[0, :]
    v0 = v_ref[0, :]
    w0 = w_ref[0, :]
    p0 = p_ref[0, :]

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Initialize the model
    # Warning: t must be the same as the one used in training, otherwise the prediction will be wrong
    # This is because the input t is scaled inside the model forward pass
    model = models.NavierStokes(config, t, x_star, y_star, u0, v0, w0, p0, nu)

    u_pred_list = []
    v_pred_list = []
    w_pred_list = []
    p_pred_list = []

    # eval_ckpt_path = '20250311-011551sotaMLP-4*256'
    eval_ckpt_path = config.wandb.name
    for idx in range(config.training.num_time_windows):

        start = num_time_steps * idx
        end = num_time_steps * (idx + 1)
        print(f"Time window {idx + 1}: start={start}, end={end}")
        u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
        v_star = v_ref[start: end, :, :]
        w_star = w_ref[start: end, :, :]
        p_star = p_ref[start: end, :, :]
        print(f"num_time_steps: {num_time_steps}")
        print(f"id: {idx}")
        print(f"u_star shape: {u_star.shape}")
        print(f"u_ref shape: {u_ref.shape}")
        # 其他代码...


        # Get the reference solution for the current time window
        u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
        v_star = v_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
        w_star = w_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
        p_star = p_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]

        # Restore the checkpoint
        # ckpt_path = os.path.join(
        #     workdir, "ckpt", config.wandb.name, "time_window_{}".format(idx + 1)
        # )
        # 修改为使用绝对路径
        ckpt_path = os.path.join(
            os.path.abspath(workdir), eval_ckpt_path, 'ckpt', "time_window_{}".format(idx + 1)
        )
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute the L2 error for the current time window
        u_error, v_error, w_error, p_error = model.compute_l2_error(
            params, t, x_star, y_star, u_star, v_star, w_star, p_star
        )
        logging.info(
            "Time window: {}, u error: {:.3e}, v error: {:.3e}, w error: {:.3e}, p error: {:.3e}".format(
                idx + 1, u_error, v_error, w_error, p_error
            )
        )

        u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)
        w_pred = model.w_pred_fn(params, model.t_star, model.x_star, model.y_star)
        p_pred = model.p_pred_fn(params, model.t_star, model.x_star, model.y_star)

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        w_pred_list.append(w_pred)
        p_pred_list.append(p_pred)

    # Get the full prediction
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)
    w_pred = jnp.concatenate(w_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)

    

    u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
    w_error = jnp.linalg.norm(w_pred - w_ref) / jnp.linalg.norm(w_ref)
    p_error = jnp.linalg.norm(p_pred - p_ref) / jnp.linalg.norm(p_ref)

    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))
    logging.info("L2 error of the full prediction of w: {:.3e}".format(w_error))
    logging.info("L2 error of the full prediction of p: {:.3e}".format(p_error))

    # Plot the results

    # 创建 evaluation_results 文件夹，如果不存在则创建
    eval_results_dir = os.path.join(workdir, 'evaluation_results')
    # 获取当前时间并格式化为字符串
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 在 evaluation_results 文件夹中创建以当前时间命名的子文件夹
    time_named_dir = os.path.join(eval_results_dir, eval_ckpt_path)
    os.makedirs(time_named_dir, exist_ok=True)

    XX, YY = jnp.meshgrid(x_star, y_star, indexing="ij")  # Grid for plotting

    # Plot at the last time step
    fig = plt.figure(figsize=(18, 5))
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.pcolor(XX, YY, w_ref[-1], cmap="jet")
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Exact")
    plt.tight_layout()

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.pcolor(XX, YY, w_pred[-1], cmap="jet")
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Predicted")
    plt.tight_layout()

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.pcolor(XX, YY, jnp.abs(w_pred[-1] - w_ref[-1]), cmap="jet")
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Absolute error")
    plt.tight_layout()

    # 添加时间文本
    time_text = fig.text(0.5, 0.02, f"Time: {t_star[-1]:.2f}", ha='center', va='bottom')

    # 保存图像
    # 修改保存路径为新创建的时间命名文件夹
    image_path = os.path.join(time_named_dir, 'evaluation.png')
    plt.savefig(image_path)
    

    # 选取有代表性的帧保存
    # representative_frames = [0, len(w_ref) // 2, len(w_ref) - 1]\
    representative_frames = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    for frame in representative_frames:
        # 更新精确解的图像
        im1.set_array(w_ref[frame].flatten())
        # 更新预测解的图像
        im2.set_array(w_pred[frame].flatten())
        # 更新绝对误差的图像
        im3.set_array(jnp.abs(w_pred[frame] - w_ref[frame]).flatten())
        # 更新时间文本，显示当前帧对应的时间
        time_text.set_text(f"Time: {t_star[frame]:.2f}")
        # 确保时间文本被正确绘制
        time_text.set_visible(True)
    
        # 保存当前帧为图片
        frame_image_path = os.path.join(time_named_dir, f'evaluation_frame_{frame}.png')
        plt.savefig(frame_image_path)

        # #每整数秒的图像不连续，怀疑是因为在传入下一个时间窗口的初始值并非上一个时间窗口的最后一帧，而是第一帧
        # #对比每个时间窗口的第一帧以及与上一个时间窗口的最后一帧的误差就可以知道是否是以上问题
        
        # if frame > 0 :
        #     last_frame = frame - 1
        #     # 更新本时间窗口的第一帧的图像
        #     im1.set_array(w_pred[frame].flatten())
        #     # 更新上一个时间窗口第一帧的图像
        #     im2.set_array(w_pred[last_frame].flatten())
        #     # 更新绝对误差的图像
        #     im3.set_array(jnp.abs(w_pred[frame] - w_pred[last_frame]).flatten())
        #     # 更新时间文本，显示当前帧对应的时间
        #     time_text.set_text(f"Time: {t_star[frame]:.2f}")
        #     # 确保时间文本被正确绘制
        #     time_text.set_visible(True)

        #     # 保存当前帧为图片
        #     frame_image_path = os.path.join(time_named_dir, f'evaluation_frame_{frame}and_frame{last_frame}_.png')
        #     plt.savefig(frame_image_path)

        
    
    # 移除 plt.show()
    # plt.show()
    
    # Animation
    def update(frame):
        """
        更新动画的每一帧。
    
        Args:
            frame (int): 当前帧的索引。
    
        Returns:
            tuple: 包含更新后的图像对象和时间文本对象。
        """
        # 更新精确解的图像
        im1.set_array(w_ref[frame].flatten())
        # 更新预测解的图像
        im2.set_array(w_pred[frame].flatten())
        # 更新绝对误差的图像
        im3.set_array(jnp.abs(w_pred[frame] - w_ref[frame]).flatten())
        # 更新时间文本，显示当前帧对应的时间
        time_text.set_text(f"Time: {t_star[frame]:.2f}")
        # 确保时间文本被正确绘制
        time_text.set_visible(True)
        return im1, im2, im3, time_text

    # 创建动画对象，指定更新函数、帧数、帧间隔和是否使用 blitting 优化
    ani = animation.FuncAnimation(fig, update, frames=len(w_ref), interval=200, blit=False)

    # 保存 w 变量的动画为 GIF
    gif_path_w = os.path.join(time_named_dir, 'evaluation_w.gif')
    ani.save(gif_path_w, writer='pillow')

    # 额外生成 u 变量的动画
    fig_u = plt.figure(figsize=(18, 5))

    ax1_u = plt.subplot(1, 3, 1)
    im1_u = ax1_u.pcolor(XX, YY, u_ref[-1], cmap="jet")
    plt.colorbar(im1_u, ax=ax1_u)
    ax1_u.set_xlabel("x")
    ax1_u.set_ylabel("y")
    ax1_u.set_title("Exact")
    plt.tight_layout()

    ax2_u = plt.subplot(1, 3, 2)
    im2_u = ax2_u.pcolor(XX, YY, u_pred[-1], cmap="jet")
    plt.colorbar(im2_u, ax=ax2_u)
    ax2_u.set_xlabel("x")
    ax2_u.set_ylabel("y")
    ax2_u.set_title("Predicted")
    plt.tight_layout()

    ax3_u = plt.subplot(1, 3, 3)
    im3_u = ax3_u.pcolor(XX, YY, jnp.abs(u_pred[-1] - u_ref[-1]), cmap="jet")
    plt.colorbar(im3_u, ax=ax3_u)
    ax3_u.set_xlabel("x")
    ax3_u.set_ylabel("y")
    ax3_u.set_title("Absolute error")
    plt.tight_layout()

    # 添加时间文本
    time_text_u = fig_u.text(0.5, 0.02, f"Time: {t_star[-1]:.2f}", ha='center', va='bottom')

    def update_u(frame):
        """
        更新 u 变量动画的每一帧。
    
        Args:
            frame (int): 当前帧的索引。
    
        Returns:
            tuple: 包含更新后的图像对象和时间文本对象。
        """
        # 更新精确解的图像
        im1_u.set_array(u_ref[frame].flatten())
        # 更新预测解的图像
        im2_u.set_array(u_pred[frame].flatten())
        # 更新绝对误差的图像
        im3_u.set_array(jnp.abs(u_pred[frame] - u_ref[frame]).flatten())
        # 更新时间文本，显示当前帧对应的时间
        time_text_u.set_text(f"Time: {t_star[frame]:.2f}")
        # 确保时间文本被正确绘制
        time_text_u.set_visible(True)
        return im1_u, im2_u, im3_u, time_text_u

    # 创建动画对象，指定更新函数、帧数、帧间隔和是否使用 blitting 优化
    ani_u = animation.FuncAnimation(fig_u, update_u, frames=len(u_ref), interval=200, blit=False)

    # 保存动画为 GIF
    gif_path_u = os.path.join(time_named_dir, 'evaluation_u.gif')
    ani_u.save(gif_path_u, writer='pillow')

    results_path = os.path.join(time_named_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        # 记录标题
        f.write("Evaluation Results\n\n")
        # 记录当前时间
        f.write(f"Current Time: {current_time}\n\n")
        # 记录工作目录
        f.write(f"Work Directory: {workdir}\n\n")
        # 记录评估的检查点路径
        f.write(f"Evaluation Checkpoint Path: {eval_ckpt_path}\n\n")
        # 记录配置信息
        f.write(f"Configuration:\n{config}\n\n")
        # 记录时间窗口数量
        f.write(f"Number of time windows: {config.training.num_time_windows}\n\n")
        # 记录每个时间窗口的误差
        for idx in range(config.training.num_time_windows):
            u_star = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
            v_star = v_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
            w_star = w_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
            u_error, v_error, w_error, p_error = model.compute_l2_error(
                params, t, x_star, y_star, u_star, v_star, w_star, p_star
            )
            f.write(f"Time window: {idx + 1}, u error: {u_error:.3e}, v error: {v_error:.3e}, w error: {w_error:.3e}\n")
        # 记录完整预测的误差
        f.write(f"\nL2 error of the full prediction of u: {u_error:.3e}\n")
        f.write(f"L2 error of the full prediction of v: {v_error:.3e}\n")
        f.write(f"L2 error of the full prediction of w: {w_error:.3e}\n")
        f.write(f"L2 error of the full prediction of p: {p_error:.3e}\n")

    return u_error, v_error, w_error, p_error
