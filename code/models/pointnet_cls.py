import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

from typing import Dict
from data_utils.MultiClothesDataLoader import MultiClothesDataLoader
import provider


def get_model(num_class: int, config_dict: Dict):
    return PointNet(num_class, config_dict['dropout'], config_dict['emb_dims'], config_dict['use_colors'], config_dict['use_cpu'])


class PointNet(nn.Module):
    def __init__(self, num_class, dropout: float = 0.4, emb_dims: int = 1024, normal_channel: bool = False, use_cpu: bool = True):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.use_cpu = use_cpu
        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel, emb_dims=emb_dims)
        self.fc1 = nn.Linear(emb_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[MultiClothesDataLoader.Modalities.POINT_CLOUD]
        if self.training:
            # points = provider.random_point_dropout(points)
            x[:, :, 0:3] = provider.random_scale_point_cloud(x[:, :, 0:3])
            x[:, :, 0:3] = provider.shift_point_cloud(x[:, :, 0:3])

        x = x.transpose(2, 1)

        if not self.use_cpu:
            x = x.cuda()
        x, trans, trans_feat = self.feat(x.float())
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
