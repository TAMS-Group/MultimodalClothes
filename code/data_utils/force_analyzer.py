#!/usr/bin/python
import csv
import os
import typing

import numpy
import numpy as np
import warnings
import pickle
import torch
import sys
import json
import open3d as o3d
import random
import cv2
import PIL.Image
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List, Union, Iterable, Optional
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
from enum import Enum
from data_utils.MultiClothesDataLoader import MultiClothesDataLoader

if __name__ == '__main__':
    import torch
    import argparse
    import sys

    data = MultiClothesDataLoader(
        '/srv/fiedler/niklas_data',
        (1, 2, 4, 6, 8, 9, 10, 12, 14, 16, 18, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54),
        sample_mappings={
              1: 6,
              2: 8,
              4: 4,
              6: 8,
              8: 1,
              9: 6,
              10: 6,
              12: 9,
              14: 8,
              16: 9,
              18: 0,
              25: 5,
              27: 5,
              28: 0,
              29: 0,
              30: 4,
              32: 5,
              33: 7,
              34: 7,
              35: 2,
              38: 3,
              39: 3,
              40: 10,
              41: 10,
              42: 11,
              43: 11,
              44: 1,
              45: 1,
              46: 3,
              47: 2,
              48: 4,
              49: 7,
              51: 2,
              52: 11,
              53: 9,
              54: 10,
        },
        modalities=[
            #MultiClothesDataLoader.Modalities.DEPTH,
            #MultiClothesDataLoader.Modalities.RGB,
            #MultiClothesDataLoader.Modalities.POINT_CLOUD,
            MultiClothesDataLoader.Modalities.FORCE_TORQUE,
        ],
        areas=(1, 2, 3),
        point_cloud_numpy_color=True
    )
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)
    l = list()
    for d, label in tqdm(data):
        l.append([d['label'], d['sample']][d[MultiClothesDataLoader.Modalities.FORCE_TORQUE]])
    np.save('ft_data', np.array(l), allow_pickle=False)
