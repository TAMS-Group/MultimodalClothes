#!/usr/bin/python

import os
import typing

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



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class MultiClothesDataReducer:

    class Modalities(Enum):
        RGB = 'rbg_img'
        DEPTH = 'depth_img'
        POINT_CLOUD = 'point_cloud'
        TACTILE = 'tactile_img'

    def __init__(self,
                 root: str,
                 samples: Tuple[Union[int, str]],
                 areas: Tuple[Union[int, str]] = (1, 2, 3),
                 point_cloud_centered: bool = False,
                 point_cloud_num_points: int = 8192,
                 rgb_crop: Optional[Tuple[int]] = None,
                 rgb_depth_mask: Optional[Tuple[int, int]] = None,
                 depth_crop: Optional[Tuple[int]] = None,
                 ):
        self.root = root
        self.samples = samples
        self.point_cloud_centered = point_cloud_centered
        self.point_cloud_num_points = point_cloud_num_points
        self.rgb_crop = rgb_crop
        self.rgb_depth_mask = rgb_depth_mask
        self.depth_crop = depth_crop
        self.all_classes = list()

        self.modality_keys = {
            MultiClothesDataReducer.Modalities.RGB: '_rgb.png',
            MultiClothesDataReducer.Modalities.DEPTH: '_depth.png',
            MultiClothesDataReducer.Modalities.POINT_CLOUD: '_pc.pcd',
            MultiClothesDataReducer.Modalities.TACTILE: ('_finger_2.png', '_finger_3.png'),
        }

        sample = {
            'rbg_img': None,
            'depth_img': None,
            'point_cloud': None,
            'tactile_img': None,
        }

        root_files = os.listdir(root)
        # TODO: filter class dirs
        class_dirs = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and len(f) == 2 and f.isdigit()]
        self.sample_strgs = [str(sample).zfill(2) for sample in self.samples]
        sample_dirs = {cls: [f for f in os.listdir(os.path.join(root, cls)) if os.path.isdir(os.path.join(root, cls, f)) if f in self.sample_strgs and len(f) == 2 and f.isdigit()] for cls in class_dirs}
        sample_dirs = {cls: samples for cls, samples in sample_dirs.items() if sample_dirs[cls]}
        # TODO: handle return
        self.area_strgs = [str(area).zfill(2) for area in areas]
        self.sample_paths = {cls: dict({sample: dict({area: sorted((self.get_base_names(os.path.join(self.root, cls, sample, area)))) for area in self.area_strgs}) for sample in samples}) for cls, samples in sample_dirs.items() if sample_dirs[cls]}
        # print(class_dirs)
        # print(sample_dirs)
        # print(sample_paths)
        # print(json.dumps(self.sample_paths, indent=2))
        class_sample_paths = dict()

        self.classlength: Dict[str:int] = defaultdict(lambda: 0)
        self.arealength: Dict[str:int] = defaultdict(lambda: 0)
        self.samplelength: Dict[str:int] = defaultdict(lambda: 0)
        self._length = 0
        self.sample_path_list: List[Tuple[str, int]] = list()

        for cls in self.sample_paths:
            p1 = os.path.join(self.root, cls)
            for sample in self.sample_paths[cls]:
                p2 = os.path.join(p1, sample)
                for area in self.sample_paths[cls][sample]:
                    p3 = os.path.join(p2, area)
                    s_paths = self.sample_paths[cls][sample][area]
                    l = len(s_paths)
                    self.sample_path_list += [(os.path.join(p3, s), cls) for s in s_paths]
                    self._length += l
                    self.classlength[cls] += l
                    self.arealength[area] += l
                    self.samplelength[sample] += l
        # print(self._length)
        # print(self.classlength)
        # print(self.arealength)
        # print(self.samplelength)
        # random.shuffle(self.sample_path_list)
        # for s in self.sample_path_list:
        #     print(s)

    def __len__(self):
        return self._length

    def get_num_classes(self) -> int:
        return len(self.sample_paths.keys())

    def get_base_names(self, path: str) -> typing.Iterable[str]:
        key = '_rgb.png'
        return (f.replace(key, '') for f in os.listdir(path) if key in f)

    def _get_item(self, index: int):
        base, label = self.sample_path_list[index]
        sample = dict()
        sample['label'] = label
        sample[MultiClothesDataReducer.Modalities.RGB] = cv2.imwrite(base + '_rgb_s.png', cv2.imread(base + self.modality_keys[MultiClothesDataReducer.Modalities.RGB])[ 430:1400, 620:1200, :])
        sample[MultiClothesDataReducer.Modalities.DEPTH] = np.array(PIL.Image.open(base + self.modality_keys[MultiClothesDataReducer.Modalities.DEPTH]))[130:540, 150:420]
        np.savez_compressed(base + '_depth_s', depth=sample[MultiClothesDataReducer.Modalities.DEPTH])
        o3d_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
            base + self.modality_keys[MultiClothesDataReducer.Modalities.POINT_CLOUD],
            remove_nan_points=True,
            remove_infinite_points=True
        )
        np_cloud = np.asarray(o3d_cloud.points)
        np_cloud = np.concatenate((np_cloud, np.asarray(o3d_cloud.colors)), axis=1)
        np.random.shuffle(np_cloud)
        np_cloud = np_cloud[:self.point_cloud_num_points]
        sample[MultiClothesDataReducer.Modalities.POINT_CLOUD] = np_cloud
        # print(base)
        np.save(base + '_pc_s', np_cloud)
        return sample


    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    import argparse
    import sys

    data = MultiClothesDataReducer(
        '/srv/fiedler/niklas_data',
        (1, 2, 4, 6, 8, 9, 10, 12, 14, 16, 18, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54),
    )
    for i in tqdm(range(len(data))):
        data[i]
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)

