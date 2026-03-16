import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前 CUDA 版本: {torch.version.cuda}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")



import numpy, scipy, diffusers, huggingface_hub
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("diffusers:", diffusers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)

import numpy as np
print(np.__version__, torch.__version__)
print(torch.from_numpy(np.arange(5)))
