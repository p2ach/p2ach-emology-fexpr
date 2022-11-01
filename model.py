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
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6_1 = nn.Linear(32, 32)
        self.fc6_2 = nn.Linear(32, 32)
        self.fc6_3 = nn.Linear(32, 32)
        self.fc6_4 = nn.Linear(32, 32)
        self.fc6_5 = nn.Linear(32, 32)
        self.fc6_6 = nn.Linear(32, 32)
        self.fc6_7 = nn.Linear(32, 32)


        self.fc7_1 = nn.Linear(32, 1)
        self.fc7_2 = nn.Linear(32, 1)
        self.fc7_3 = nn.Linear(32, 1)
        self.fc7_4 = nn.Linear(32, 1)
        self.fc7_5 = nn.Linear(32, 1)
        self.fc7_6 = nn.Linear(32, 1)
        self.fc7_7 = nn.Linear(32, 1)


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
        self.dropout2 = nn.Dropout(0.5)

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
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)

        x_1 = self.fc6_1(x)
        x_2 = self.fc6_2(x)
        x_3 = self.fc6_3(x)
        x_4 = self.fc6_4(x)
        x_5 = self.fc6_5(x)
        x_6 = self.fc6_6(x)
        x_7 = self.fc6_7(x)

        x_1 = F.relu(x_1)
        x_2 = F.relu(x_2)
        x_3 = F.relu(x_3)
        x_4 = F.relu(x_4)
        x_5 = F.relu(x_5)
        x_6 = F.relu(x_6)
        x_7 = F.relu(x_7)

        x_1 = self.fc7_1(x_1)
        x_2 = self.fc7_2(x_2)
        x_3 = self.fc7_3(x_3)
        x_4 = self.fc7_4(x_4)
        x_5 = self.fc7_5(x_5)
        x_6 = self.fc7_6(x_6)
        x_7 = self.fc7_7(x_7)

        x_1 = F.relu(x_1)
        x_2 = F.relu(x_2)
        x_3 = F.relu(x_3)
        x_4 = F.relu(x_4)
        x_5 = F.relu(x_5)
        x_6 = F.relu(x_6)
        x_7 = F.relu(x_7)

        y= torch.cat((x_1, x_2, x_3, x_4, x_5, x_6, x_7), dim=1)

        # logits = self.linear_relu_stack(x)
        return x_1, x_2, x_3, x_4, x_5, x_6, x_7, y

class FaceExprNet_2(nn.Module):
    def __init__(self, landmark_num,chns):
        super(FaceExprNet_2, self).__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(landmark_num*chns, 256)
        # self.fc2 = nn.Linear(512, 512)
        # self.fc2_2 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.fc3_2 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 64)
        self.fc6_1 = nn.Linear(64, 64)
        self.fc6_2 = nn.Linear(64, 64)
        self.fc6_3 = nn.Linear(64, 64)
        self.fc6_4 = nn.Linear(64, 64)
        self.fc6_5 = nn.Linear(64, 64)
        self.fc6_6 = nn.Linear(64, 64)
        self.fc6_7 = nn.Linear(64, 64)


        self.fc7_1 = nn.Linear(64, 1)
        self.fc7_2 = nn.Linear(64, 1)
        self.fc7_3 = nn.Linear(64, 1)
        self.fc7_4 = nn.Linear(64, 1)
        self.fc7_5 = nn.Linear(64, 1)
        self.fc7_6 = nn.Linear(64, 1)
        self.fc7_7 = nn.Linear(64, 1)


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
        self.dropout2 = nn.Dropout(0.5)

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
        # x = self.fc2(x)
        # x = F.relu(x)

        # x = self.fc2_2(x)
        # x = self.dropout2(x)

        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        x = self.fc3_2(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        # x = self.fc5(x)
        # x = F.relu(x)

        x_1 = self.fc6_1(x)
        x_2 = self.fc6_2(x)
        x_3 = self.fc6_3(x)
        x_4 = self.fc6_4(x)
        x_5 = self.fc6_5(x)
        x_6 = self.fc6_6(x)
        x_7 = self.fc6_7(x)

        x_1 = F.relu(x_1)
        x_2 = F.relu(x_2)
        x_3 = F.relu(x_3)
        x_4 = F.relu(x_4)
        x_5 = F.relu(x_5)
        x_6 = F.relu(x_6)
        x_7 = F.relu(x_7)

        x_1 = self.fc7_1(x_1)
        x_2 = self.fc7_2(x_2)
        x_3 = self.fc7_3(x_3)
        x_4 = self.fc7_4(x_4)
        x_5 = self.fc7_5(x_5)
        x_6 = self.fc7_6(x_6)
        x_7 = self.fc7_7(x_7)

        x_1 = F.relu(x_1)
        x_2 = F.relu(x_2)
        x_3 = F.relu(x_3)
        x_4 = F.relu(x_4)
        x_5 = F.relu(x_5)
        x_6 = F.relu(x_6)
        x_7 = F.relu(x_7)

        x_1 = torch.minimum(x_1,torch.ones_like(x_1))
        x_2 = torch.minimum(x_2,torch.ones_like(x_2))
        x_3 = torch.minimum(x_3,torch.ones_like(x_3))
        x_4 = torch.minimum(x_4,torch.ones_like(x_4))
        x_5 = torch.minimum(x_5,torch.ones_like(x_5))
        x_6 = torch.minimum(x_6,torch.ones_like(x_6))
        x_7 = torch.minimum(x_7,torch.ones_like(x_7))

        y= torch.cat((x_1, x_2, x_3, x_4, x_5, x_6, x_7), dim=1)

        # logits = self.linear_relu_stack(x)
        return x_1, x_2, x_3, x_4, x_5, x_6, x_7, y

class FaceExprNet_each(nn.Module):
    def __init__(self, landmark_num,chns):
        super(FaceExprNet_each, self).__init__()
        self.flatten = nn.Flatten()

        self.fc1_1 = nn.Linear(landmark_num*chns, 256)
        self.fc1_2 = nn.Linear(landmark_num*chns, 256)
        self.fc1_3 = nn.Linear(landmark_num*chns, 256)
        self.fc1_4 = nn.Linear(landmark_num*chns, 256)
        self.fc1_5 = nn.Linear(landmark_num*chns, 256)
        self.fc1_6 = nn.Linear(landmark_num*chns, 256)
        self.fc1_7= nn.Linear(landmark_num*chns, 256)
        # self.fc2 = nn.Linear(512, 512)
        # self.fc2_2 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.fc3_1 = nn.Linear(256, 128)
        self.fc3_2 = nn.Linear(256, 128)
        self.fc3_3 = nn.Linear(256, 128)
        self.fc3_4 = nn.Linear(256, 128)
        self.fc3_5 = nn.Linear(256, 128)
        self.fc3_6 = nn.Linear(256, 128)
        self.fc3_7 = nn.Linear(256, 128)
        self.fc4_1 = nn.Linear(128, 64)
        self.fc4_2 = nn.Linear(128, 64)
        self.fc4_3 = nn.Linear(128, 64)
        self.fc4_4 = nn.Linear(128, 64)
        self.fc4_5 = nn.Linear(128, 64)
        self.fc4_6 = nn.Linear(128, 64)
        self.fc4_7 = nn.Linear(128, 64)




        # self.fc5 = nn.Linear(64, 64)
        self.fc6_1 = nn.Linear(64, 64)
        self.fc6_2 = nn.Linear(64, 64)
        self.fc6_3 = nn.Linear(64, 64)
        self.fc6_4 = nn.Linear(64, 64)
        self.fc6_5 = nn.Linear(64, 64)
        self.fc6_6 = nn.Linear(64, 64)
        self.fc6_7 = nn.Linear(64, 64)


        self.fc7_1 = nn.Linear(64, 1)
        self.fc7_2 = nn.Linear(64, 1)
        self.fc7_3 = nn.Linear(64, 1)
        self.fc7_4 = nn.Linear(64, 1)
        self.fc7_5 = nn.Linear(64, 1)
        self.fc7_6 = nn.Linear(64, 1)
        self.fc7_7 = nn.Linear(64, 1)


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
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
                # x에 대해서 max pooling을 실행합니다.
        # x = F.max_pool2d(x, 2)
        # 데이터가 dropout1을 지나갑니다.
        # x = self.dropout1(x)
        # start_dim=1으로 x를 압축합니다.

        # 데이터가 fc1을 지나갑니다.
        x1 = self.fc1_1(x)
        x1 = F.relu(x1)
        x2 = self.fc1_2(x)
        x2 = F.relu(x2)
        x3 = self.fc1_3(x)
        x3 = F.relu(x3)
        x4 = self.fc1_4(x)
        x4 = F.relu(x4)
        x5 = self.fc1_5(x)
        x5 = F.relu(x5)
        x6 = self.fc1_6(x)
        x6 = F.relu(x6)
        x7 = self.fc1_7(x)
        x7 = F.relu(x7)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = F.relu(x)

        # x = self.fc2_2(x)
        # x = self.dropout2(x)

        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        x1 = self.fc3_1(x1)
        x2 = self.fc3_2(x2)
        x3 = self.fc3_3(x3)
        x4 = self.fc3_4(x4)
        x5 = self.fc3_5(x5)
        x6 = self.fc3_6(x6)
        x7 = self.fc3_7(x7)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)
        x4 = F.relu(x4)
        x5 = F.relu(x5)
        x6 = F.relu(x6)
        x7 = F.relu(x7)
        x1 = self.fc4_1(x1)
        x2 = self.fc4_2(x2)
        x3 = self.fc4_3(x3)
        x4 = self.fc4_4(x4)
        x5 = self.fc4_5(x5)
        x6 = self.fc4_6(x6)
        x7 = self.fc4_7(x7)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)
        x4 = F.relu(x4)
        x5 = F.relu(x5)
        x6 = F.relu(x6)
        x7 = F.relu(x7)
        # x = self.fc5(x)
        # x = F.relu(x)

        x_1 = self.fc6_1(x1)
        x_2 = self.fc6_2(x2)
        x_3 = self.fc6_3(x3)
        x_4 = self.fc6_4(x4)
        x_5 = self.fc6_5(x5)
        x_6 = self.fc6_6(x6)
        x_7 = self.fc6_7(x7)

        x_1 = F.relu(x_1)
        x_2 = F.relu(x_2)
        x_3 = F.relu(x_3)
        x_4 = F.relu(x_4)
        x_5 = F.relu(x_5)
        x_6 = F.relu(x_6)
        x_7 = F.relu(x_7)

        x_1 = self.fc7_1(x_1)
        x_2 = self.fc7_2(x_2)
        x_3 = self.fc7_3(x_3)
        x_4 = self.fc7_4(x_4)
        x_5 = self.fc7_5(x_5)
        x_6 = self.fc7_6(x_6)
        x_7 = self.fc7_7(x_7)

        x_1 = F.relu(x_1)
        x_2 = F.relu(x_2)
        x_3 = F.relu(x_3)
        x_4 = F.relu(x_4)
        x_5 = F.relu(x_5)
        x_6 = F.relu(x_6)
        x_7 = F.relu(x_7)

        x_1 = torch.minimum(x_1,torch.ones_like(x_1))
        x_2 = torch.minimum(x_2,torch.ones_like(x_2))
        x_3 = torch.minimum(x_3,torch.ones_like(x_3))
        x_4 = torch.minimum(x_4,torch.ones_like(x_4))
        x_5 = torch.minimum(x_5,torch.ones_like(x_5))
        x_6 = torch.minimum(x_6,torch.ones_like(x_6))
        x_7 = torch.minimum(x_7,torch.ones_like(x_7))

        y= torch.cat((x_1, x_2, x_3, x_4, x_5, x_6, x_7), dim=1)

        # logits = self.linear_relu_stack(x)
        return x_1, x_2, x_3, x_4, x_5, x_6, x_7, y