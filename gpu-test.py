import jax
import torch
import os
import jax
print(jax.devices())

# # # 手动指定 JAX 后端为 GPU
# os.environ["JAX_PLATFORM_NAME"] = "gpu"

# print("jax-version=", jax.__version__)
# print("torch-version=", torch.__version__)
# print("jax.devices()=", jax.devices())
# print("jax.devices('cpu')=", jax.devices('cpu'))
# print("jax.local_devices()=", jax.local_devices())
# print("jax.devices=", jax.devices())
# print("cuda.is_available=", torch.cuda.is_available())
# print("torch.cuda.device_count()=", torch.cuda.device_count())

# import torch

# if torch.cuda.is_available():
#     cudnn_version = torch.backends.cudnn.version()
#     print(f"cuDNN version: {cudnn_version}")
# else:
#     print("CUDA is not available.")
    
# print("jax.devices('gpu')=", jax.devices('gpu'))
