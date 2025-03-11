import ml_collections

import jax.numpy as jnp
from datetime import datetime


def get_config():
    """
    获取默认的超参数配置。

    Returns:
        ml_collections.ConfigDict: 包含所有超参数配置的字典。
    """
    # 创建一个配置字典
    config = ml_collections.ConfigDict()

    # 设置运行模式为训练模式
    # config.mode = "train"
    config.mode = "eval"


    # Weights & Biases 配置
    config.wandb = wandb = ml_collections.ConfigDict()
    # 设置 Weights & Biases 项目名称
    wandb.project = "PINN-NS_tori"
    # 设置 Weights & Biases 运行名称为当前运行时间
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.name = current_time
    # 设置 Weights & Biases 标签，默认为 None
    wandb.tag = None

    # 模型架构配置
    config.arch = arch = ml_collections.ConfigDict()
    # 设置模型架构名称为多层感知机（Mlp）
    arch.arch_name = "Mlp"
    # 设置模型的层数
    arch.num_layers = 4
    # 设置模型隐藏层的维度
    arch.hidden_dim = 64
    # 设置模型输出层的维度
    arch.out_dim = 2
    # 设置模型的激活函数为双曲正切函数
    arch.activation = "tanh"
    # 设置模型的周期性配置
    arch.periodicity = ml_collections.ConfigDict(
        {
            # 每个轴的周期
            "period": (1.0, 1.0),
            # 应用周期性的轴
            "axis": (1, 2),
            # 周期是否可训练
            "trainable": (False, False)
        }
    )
    # 定义傅里叶嵌入的配置，embed_scale 为嵌入的缩放因子，embed_dim 为嵌入的维度
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 256})
    # 设置模型的重参数化配置
    arch.reparam = ml_collections.ConfigDict(
        {
            # 重参数化类型
            "type": "weight_fact",
            # 重参数化的均值
            "mean": 0.5,
            # 重参数化的标准差
            "stddev": 0.1
        }
    )

    # 优化器配置
    config.optim = optim = ml_collections.ConfigDict()
    # 设置优化器为 Adam 优化器
    optim.optimizer = "Adam"
    # 设置 Adam 优化器的第一个动量参数
    optim.beta1 = 0.9
    # 设置 Adam 优化器的第二个动量参数
    optim.beta2 = 0.999
    # 设置 Adam 优化器的数值稳定性参数
    optim.eps = 1e-8
    # 设置初始学习率
    optim.learning_rate = 1e-3
    # 设置学习率的衰减率
    optim.decay_rate = 0.9
    # 设置学习率衰减的步数
    optim.decay_steps = 200
    # 设置梯度累积的步数
    optim.grad_accum_steps = 0

    # 训练配置
    config.training = training = ml_collections.ConfigDict()
    # 设置最大训练步数
    training.max_steps = 5000
    # 设置每个设备的批量大小
    training.batch_size_per_device = 1024
    # 设置时间窗口的数量
    training.num_time_windows = 5

    # 损失权重配置
    config.weighting = weighting = ml_collections.ConfigDict()
    # 设置损失权重的调整方案为梯度范数
    weighting.scheme = "grad_norm"
    # 设置初始损失权重
    weighting.init_weights = ml_collections.ConfigDict(
        {
            "u_ic": 1.0,
            "v_ic": 1.0,
            "w_ic": 1.0,
            "rm": 1.0,
            "rc": 1.0
        }
    )
    # 设置损失权重调整的动量参数
    weighting.momentum = 0.9
    # 设置损失权重更新的步数间隔
    weighting.update_every_steps = 100
    # 设置是否使用因果权重调整
    weighting.use_causal = True
    # 设置因果权重调整的容差
    weighting.causal_tol = 1.0
    # 设置数据分块的数量
    weighting.num_chunks = 32

    # 日志记录配置
    config.logging = logging = ml_collections.ConfigDict()
    # 设置日志记录的步数间隔
    logging.log_every_steps = 100
    # 设置是否记录误差
    logging.log_errors = True
    # 设置是否记录损失
    logging.log_losses = True
    # 设置是否记录权重
    logging.log_weights = True
    # 设置是否记录预测结果
    logging.log_preds = False
    # 设置是否记录梯度
    logging.log_grads = False
    # 设置是否记录 NTK（Neural Tangent Kernel）
    logging.log_ntk = False

    # 模型保存配置
    config.saving = saving = ml_collections.ConfigDict()
    # 设置模型保存的步数间隔，None 表示不按步数保存
    saving.save_every_steps = 1000
    # 设置保留的检查点数量
    saving.num_keep_ckpts = 10

    # 输入维度，用于初始化 Flax 模型
    config.input_dim = 3

    # 随机数生成器的种子
    config.seed = 42

    return config
