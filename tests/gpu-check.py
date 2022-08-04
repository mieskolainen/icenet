# Test GPU support of tensorflow, torch

import tensorflow
import torch

from tensorflow.python.client import device_lib

print(f'   >>> tensorflow: {tensorflow.test.is_gpu_available()} {device_lib.list_local_devices()}')
print(f'   >>> torch:      {torch.cuda.get_device_name(0)} {torch.cuda.is_available()}')
