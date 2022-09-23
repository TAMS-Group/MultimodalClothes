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
import math
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

    def grasp_area(name):
        return int(name[-5:-3])

    sample_mappings = {
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
        }
    modalities = [MultiClothesDataLoader.Modalities.FILENAME]

    train_samples = (1, 2, 4, 6, 8, 9, 12, 18, 25, 27, 28, 30, 33, 35, 38, 40, 42, 43, 44, 46, 47, 49, 53, 54)
    test_samples = (10, 14, 16, 29, 32, 34, 39, 41, 45, 48, 51, 52, )
    # train_samples = (1, 2, 6, 9, 12, 18, 25, 27, 28, 38, 46, 53)
    # test_samples = (10, 14, 16, 29, 32, 39)

    train_data = MultiClothesDataLoader(
        '/srv/ssd_nvm/dataset/MultiModalClothes/niklas_data',
        train_samples,
        sample_mappings=sample_mappings,
        modalities=modalities,
        areas=(1, 2, 3),
        point_cloud_numpy_color=True
    )

    test = MultiClothesDataLoader(
        '/srv/ssd_nvm/dataset/MultiModalClothes/niklas_data',
        test_samples,
        sample_mappings=sample_mappings,
        modalities=modalities,
        areas=(1, 2, 3),
        point_cloud_numpy_color=True
    )

    n_val = int(len(train_data) * 0.2)
    n_train = len(train_data) - n_val
    train, validation = torch.utils.data.random_split(train_data, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_numbers = defaultdict(lambda: 0)
    validation_numbers = defaultdict(lambda: 0)
    test_numbers = defaultdict(lambda: 0)
    l = list()
    for d, label in tqdm(train):
        train_numbers[d['label']] += 1
        l.append(d[MultiClothesDataLoader.Modalities.FILENAME] + '\n')
    print('train:')
    print(json.dumps(dict(train_numbers)))
    with open('train_samples.txt', 'w') as f:
        f.writelines(l)

    l = list()
    for d, label in tqdm(validation):
        validation_numbers[d['label']] += 1
        l.append(d[MultiClothesDataLoader.Modalities.FILENAME] + '\n')
    print('validation:')
    print(json.dumps(dict(validation_numbers)))
    with open('validation_samples.txt', 'w') as f:
        f.writelines(l)

    l = list()
    for d, label in tqdm(test):
        test_numbers[d['label']] += 1
        l.append(d[MultiClothesDataLoader.Modalities.FILENAME] + '\n')
    print('test:')
    print(json.dumps(dict(test_numbers)))
    with open('test_samples.txt', 'w') as f:
        f.writelines(l)
