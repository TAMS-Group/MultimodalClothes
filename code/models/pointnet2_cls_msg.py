import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from typing import Dict
from data_utils.MultiClothesDataLoader import MultiClothesDataLoader
import provider

def get_model(num_class: int, config_dict: Dict):
    return PointNet2MSG(num_class, config_dict['dropout'], config_dict['emb_dims'], config_dict['use_colors'], config_dict['use_cpu'])

class PointNet2MSG(nn.Module):
    def __init__(self, num_class, dropout, emb_dims: int = 1024, normal_channel=True, use_cpu=True):
        super(PointNet2MSG, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.use_cpu = use_cpu
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(emb_dims, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = x[MultiClothesDataLoader.Modalities.POINT_CLOUD]

        x = x.transpose(2, 1).float()
        if self.training:
            # points = provider.random_point_dropout(points)
            x[:, :, 0:3] = provider.random_scale_point_cloud(x[:, :, 0:3])
            x[:, :, 0:3] = provider.shift_point_cloud(x[:, :, 0:3])
        if not self.use_cpu:
            x = x.cuda()
        B, _, _ = x.shape
        if self.normal_channel:
            norm = x[:, 3:, :]
            xyz = x[:, :3, :]
        else:
            norm = None
            xyz = x

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


