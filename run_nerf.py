import os
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import axes3d
from torch import nn
from tqdm import trange

from data_handling import NeRF_Data_Loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")


nerf_data = NeRF_Data_Loader()
nerf_data.debug_information()
nerf_data.testimg_show()
