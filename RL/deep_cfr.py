# implement deep cfr to play poker 
# partially based on https://arxiv.org/pdf/1811.00164

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
