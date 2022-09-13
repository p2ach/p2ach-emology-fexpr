import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

class FaceExprNet(nn.Module):
    def __init__(self, landmark_num,chns):
        super(FaceExprNet, self).__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(landmark_num*chns, 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 7)



        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(landmark_num*chns, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 7),
        #     nn.ReLU()
        # )
        # self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.flatten(x)
                # x에 대해서 max pooling을 실행합니다.
        # x = F.max_pool2d(x, 2)
        # 데이터가 dropout1을 지나갑니다.
        # x = self.dropout1(x)
        # start_dim=1으로 x를 압축합니다.

        # 데이터가 fc1을 지나갑니다.
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)


        # logits = self.linear_relu_stack(x)
        return x