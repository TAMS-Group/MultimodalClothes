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


class MultiClothesDataLoader(Dataset):

    class Modalities(Enum):
        RGB = 'rbg_img'
        DEPTH = 'depth_img'
        POINT_CLOUD = 'point_cloud'
        TACTILE = 'tactile_img'
        FORCE_TORQUE = 'ft_data'
        FILENAME = 'filename'

    def __init__(self,
                 root: str,
                 samples: Tuple[Union[int, str], ...],
                 areas: Tuple[Union[int, str], ...] = (1, 2, 3),
                 modalities: Union[Tuple[Modalities], List[Modalities]] = (Modalities.RGB, Modalities.POINT_CLOUD),
                 small_data: bool = True,
                 sample_mappings: Optional[Dict[Union[int, str], int]] = None,
                 class_mappings: Optional[Dict[Union[int, str], int]] = None,
                 point_cloud_centered: bool = False,
                 point_cloud_num_points: Optional[int] = None,
                 point_cloud_numpy: bool = True,
                 point_cloud_numpy_color: bool = False,
                 point_cloud_random_scaling: bool = False,
                 rgb_crop: Optional[Tuple[int]] = None,
                 rgb_depth_mask: Optional[Tuple[int, int]] = None,
                 depth_crop: Optional[Tuple[int]] = None,
                 ):
        self.root = root
        self.samples = samples
        self.modalities = modalities
        self.small_data = small_data
        self.sample_mappings = sample_mappings
        self.class_mappings = class_mappings
        self.point_cloud_centered = point_cloud_centered
        self.point_cloud_num_points = point_cloud_num_points
        self.point_cloud_numpy = point_cloud_numpy
        self.point_cloud_numpy_color = point_cloud_numpy_color
        self.point_cloud_random_scaling = point_cloud_random_scaling
        self.rgb_crop = rgb_crop
        self.rgb_depth_mask = rgb_depth_mask
        self.depth_crop = depth_crop
        self.all_classes = list()

        if self.small_data:
            self.modality_keys = {
                MultiClothesDataLoader.Modalities.RGB: '_rgb_s.png',
                MultiClothesDataLoader.Modalities.DEPTH: '_depth_s.npz',
                MultiClothesDataLoader.Modalities.POINT_CLOUD: '_pc_s.npy',
                MultiClothesDataLoader.Modalities.TACTILE: ('_finger_2.png', '_finger_3.png'),
                MultiClothesDataLoader.Modalities.FORCE_TORQUE: '_ft.csv',
            }
        else:
            self.modality_keys = {
                MultiClothesDataLoader.Modalities.RGB: '_rgb.png',
                MultiClothesDataLoader.Modalities.DEPTH: '_depth.png',
                MultiClothesDataLoader.Modalities.POINT_CLOUD: '_pc.pcd',
                MultiClothesDataLoader.Modalities.TACTILE: ('_finger_2.png', '_finger_3.png'),
                MultiClothesDataLoader.Modalities.FORCE_TORQUE: '_ft.csv',
            }


        sample = {
            'rbg_img': None,
            'depth_img': None,
            'point_cloud': None,
            'tactile_img': None,
            'ft_data': None,
        }

        root_files = os.listdir(root)
        # TODO: filter class dirs
        class_dirs = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and len(f) == 2 and f.isdigit()]
        self.sample_strgs = [str(sample).zfill(2) for sample in self.samples]
        sample_dirs = {cls: [f for f in os.listdir(os.path.join(root, cls)) if os.path.isdir(os.path.join(root, cls, f)) if f in self.sample_strgs and len(f) == 2 and f.isdigit()] for cls in class_dirs}
        sample_dirs = {cls: samples for cls, samples in sample_dirs.items() if sample_dirs[cls]}
        # TODO: handle return
        self.area_strgs = [str(area).zfill(2) for area in areas]
        self.check_areas(self.area_strgs, sample_dirs)
        self.sample_paths = {cls: dict({sample: dict({area: sorted((self.get_base_names(os.path.join(self.root, cls, sample, area)))) for area in self.area_strgs}) for sample in samples}) for cls, samples in sample_dirs.items() if sample_dirs[cls]}
        # print(class_dirs)
        # print(sample_dirs)
        # print(sample_paths)
        # print(json.dumps(self.sample_paths, indent=2))
        class_sample_paths = dict()

        if self.class_mappings:
            if self.sample_mappings:
                raise Exception("cannot define both class and sample mappings!")
            self.class_mappings = {str(o_cls).zfill(2): cls for o_cls, cls in self.class_mappings.items()}
            for cls in sample_dirs.keys():
                if cls not in self.class_mappings.keys():
                    raise Exception(f'No class mapping for original class {cls}!')
        else:
            if not self.sample_mappings:
                raise Exception('You need to define either class or sample mappings!')
            self.sample_mappings = {str(sample).zfill(2): cls for sample, cls in self.sample_mappings.items()}
            for sample in self.sample_strgs:
                if sample not in self.sample_mappings.keys():
                    raise Exception(f'No class mapping for sample {sample}!')

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
                    if self.sample_mappings:
                        self.sample_path_list += [(os.path.join(p3, s), self.sample_mappings[sample]) for s in s_paths]
                    else:
                        self.sample_path_list += [(os.path.join(p3, s), self.class_mappings[cls]) for s in s_paths]
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


    def check_areas(self, areas: typing.Iterable[str], sample_dirs: Dict):
        ret = True
        for cls in sample_dirs.keys():
            for sample in sample_dirs[cls]:
                for area in areas:
                    if not os.path.exists(os.path.join(self.root, cls, sample, area)):
                        ret = False
                        print(f'Data init error: area \"{area}\" for sample \"{sample}\" of class \"{cls}\" not found.')
        return ret

    def get_num_classes(self) -> int:
        return len(self.sample_paths.keys())

    def get_base_names(self, path: str) -> typing.Iterable[str]:
        key = '_rgb.png'
        return (f.replace(key, '') for f in os.listdir(path) if key in f)

    def _get_item(self, index: int):
        base, label = self.sample_path_list[index]
        sample = dict()
        sample['label'] = int(label)
        sample['sample'] = int(base[-8:-6])
        if MultiClothesDataLoader.Modalities.FILENAME in self.modalities:
            sample[MultiClothesDataLoader.Modalities.FILENAME] = base[-31:]
        if MultiClothesDataLoader.Modalities.RGB in self.modalities:
            sample[MultiClothesDataLoader.Modalities.RGB] = cv2.imread(base + self.modality_keys[MultiClothesDataLoader.Modalities.RGB])
        if MultiClothesDataLoader.Modalities.DEPTH in self.modalities:
            if self.modality_keys[MultiClothesDataLoader.Modalities.DEPTH].endswith('png'):
                sample[MultiClothesDataLoader.Modalities.DEPTH] = np.array(PIL.Image.open(base + self.modality_keys[MultiClothesDataLoader.Modalities.DEPTH]))
            elif self.modality_keys[MultiClothesDataLoader.Modalities.DEPTH].endswith('npz'):
                sample[MultiClothesDataLoader.Modalities.DEPTH] = np.load(base + self.modality_keys[MultiClothesDataLoader.Modalities.DEPTH])['depth']
            else:
                print('error with pointcloud file name')
        if MultiClothesDataLoader.Modalities.POINT_CLOUD in self.modalities:
            if self.modality_keys[MultiClothesDataLoader.Modalities.POINT_CLOUD].endswith('pcd'):
                o3d_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
                    base + self.modality_keys[MultiClothesDataLoader.Modalities.POINT_CLOUD],
                    remove_nan_points=True,
                    remove_infinite_points=True
                )
                if self.point_cloud_numpy:
                    np_cloud = np.asarray(o3d_cloud.points)
                    if self.point_cloud_numpy_color:
                        np_cloud = np.concatenate((np_cloud, np.asarray(o3d_cloud.colors)), axis=1)
                    np.random.shuffle(np_cloud)
                    np_cloud = np_cloud[:self.point_cloud_num_points]
                    sample[MultiClothesDataLoader.Modalities.POINT_CLOUD] = np_cloud
                else:
                    sample[MultiClothesDataLoader.Modalities.POINT_CLOUD] = o3d_cloud
                    if self.point_cloud_num_points:
                        raise NotImplementedError('point number not implemented for open3d point clouds.')
            elif self.modality_keys[MultiClothesDataLoader.Modalities.POINT_CLOUD].endswith('npy'):
                np_cloud = np.load(base + self.modality_keys[MultiClothesDataLoader.Modalities.POINT_CLOUD])
                if not self.point_cloud_numpy_color:
                    np_cloud = np_cloud[:, :3]
                np.random.shuffle(np_cloud)
                np_cloud = np_cloud[:self.point_cloud_num_points]
                sample[MultiClothesDataLoader.Modalities.POINT_CLOUD] = np_cloud
            else:
                print('error with pointcloud file name')
        if MultiClothesDataLoader.Modalities.FORCE_TORQUE in self.modalities:
            index = int(base[-2:])
            l = list()
            with open(base[:-3] + self.modality_keys[MultiClothesDataLoader.Modalities.FORCE_TORQUE]) as f:
                c = csv.reader(f, delimiter=',')
                for i in c:
                    l.append(i)
            n = np.array(l[1:], dtype=np.float32)
            if index < 25:
                n = np.array_split(n[n[:, 7] > 0.01], 25)[index]
            else:
                n = np.array_split(n[n[:, 7] < -0.01], 25)[index-25]
            sample[MultiClothesDataLoader.Modalities.FORCE_TORQUE] = np.mean(n[:, :6], axis=0)
        return sample, label

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    import argparse
    import sys

    data = MultiClothesDataLoader(
        '/run/media/nfiedler/214a284f-5d1a-4897-a171-ef1d07f3ff98/niklas_data',
        (33, 34, 38, 39, 46, 49),
        sample_mappings={33: 0, 34: 1, 38: 2, 39: 3, 46: 4, 49: 5},
        modalities=[
            #MultiClothesDataLoader.Modalities.DEPTH,
            #MultiClothesDataLoader.Modalities.RGB,
            #MultiClothesDataLoader.Modalities.POINT_CLOUD,
            MultiClothesDataLoader.Modalities.FORCE_TORQUE,
        ],
        areas=[1],
        point_cloud_numpy_color=True
    )
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)

