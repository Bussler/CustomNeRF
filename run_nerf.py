import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running on: {device}")