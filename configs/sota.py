import ml_collections

import jax.numpy as jnp
from datetime import datetime


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # config.mode = "train"
    config.mode = "eval"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-NS_tori"
    # 设置 Weights & Biases 运行名称为当前运行时间拼接 'sota'
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.name = current_time + 'sotaMLP-4*256' 
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    # arch.arch_name = "ModifiedMlp"
    arch.num_layers = 4
    # arch.num_branch_layers = 4
    # arch.num_trunk_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 3
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0, 1.0), "axis": (1, 2), "trainable": (False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 256 })
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-4
    optim.decay_rate = 0.9
    optim.decay_steps = 200
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 50000
    training.batch_size_per_device = 128
    training.num_time_windows = 10

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"u_ic": 1.0, "v_ic": 1.0, "p_ic": 1.0, "w_ic": 1.0, "rm": 1.0, "rc": 1.0, "ru": 1.0, "rv": 1.0}
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
