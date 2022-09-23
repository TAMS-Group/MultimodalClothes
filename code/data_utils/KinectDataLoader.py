#!/usr/bin/python

import os
import numpy as np
import warnings
import pickle
import torch

from tqdm import tqdm
from torch.utils.data import Dataset



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


class KinectDataLoader(Dataset):
    def __init__(self, root, split='train', include_normals=False, num_points=1024, center_pointclouds=False, random_scaling=False):
        self.root = root
        self.include_normals = include_normals
        self.path = os.path.join(self.root, split)
        self.num_points = num_points
        self.datapath = list()
        self.label = list()
        self.centering = center_pointclouds
        self.random_scaling = random_scaling

        with open(os.path.join(self.path, 'labels.txt')) as f:
            for line in f.readlines():
                s = line.replace('\n', '').split(' ')
                self.datapath.append(s[0])
                self.label.append(s[1])

    def __len__(self):
        return len(self.datapath)

    def get_num_classes(self):
        return len(set(self.label))

    def _get_item(self, index):
        x = 6 if self.include_normals else 3
        a = np.load(os.path.join(self.path, os.path.basename(self.datapath[index])))['points'][:self.num_points, :x]
        t = torch.tensor(a)
        if self.centering:
            t[:, :3] = (t[:, :3] - (t[:, :3].min(0)[0] + (t[:, :3].max(0)[0] - t[:, :3].min(0)[0]) / 2)) * (1 + (torch.rand(1) * .2 - .1))
        if self.random_scaling:
            t[:, :3] *= (1 + (torch.rand(1) * .2 - .1))
        return t, int(self.label[index])

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    import argparse
    import sys


    def parse_args():
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('Dataloader Test')
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
        parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
        parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
        parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
        parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
        parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
        parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
        # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
        parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
        parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
        parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
        parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
        parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
        return parser.parse_args()

    data = KinectDataLoader(sys.argv[1], num_points=1, split='train', center_pointclouds=True, random_scaling=True, include_normals=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)
    for point, label in DataLoader:
        pass
        print(point)
        print(label)
