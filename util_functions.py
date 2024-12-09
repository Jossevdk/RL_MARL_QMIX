from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from gymPacMan import gymPacMan_parallel_env
import torch.nn.functional as F
from collections import deque
import random
import os
from dataclasses import dataclass

import cProfile
import pstats

torch.autograd.set_detect_anomaly(True)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



