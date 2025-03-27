import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
 


class LightSteeringNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # descobrir o tamanho do flatten dinamicamente
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            flatten_size = self.features(dummy).view(1, -1).size(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x).squeeze()

