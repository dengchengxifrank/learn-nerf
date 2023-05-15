import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

#这个version下的NeRF没有加上原来的坐标而是完全按照paper的60 dim来处理(x,y,z) 24 dim来处理方向
class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.linear1 = nn.Linear(60,256,bias=True)
        self.linear2 = nn.Linear(256, 256,bias=True)
        self.linear3 = nn.Linear(256, 256,bias=True)
        self.linear4 = nn.Linear(256, 256,bias=True)
        self.linear5 = nn.Linear(256, 256,bias=True)
        self.linear6 = nn.Linear(316, 256,bias=True)
        self.linear7 = nn.Linear(256, 256,bias=True)
        self.linear8 = nn.Linear(256, 256,bias=True)
        self.linear_alpha = nn.Linear(256,1,bias=True)
        self.linear9 = nn.Linear(256,256,bias=True)
        self.linear10 = nn.Linear(280,128,bias=True)
        self.linear11 = nn.Linear(128,3, bias=True)

    def forward(self,x,dir):
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear2(x1))
        x3 = F.relu(self.linear3(x2))
        x4 = F.relu(self.linear4(x3))
        x4 = x4 + x1
        x5 = F.relu(self.linear5(x4))
        x5 = torch.concat((x5,x),dim=1)
        x6 = F.relu(self.linear6(x5))
        x7 = F.relu(self.linear7(x6))
        x8 = F.relu(self.linear8(x7))
        alpha = self.linear_alpha(x8)
        x9 = F.relu(self.linear9(x8))

        x10 = torch.concat((x9,dir),dim=1)
        x11 = F.relu(self.linear10(x10))
        rgb = F.sigmoid(self.linear11(x11))

        return alpha,rgb

if __name__ == "__main__":
     x = torch.rand(4,60)
     dir = torch.rand(4,24)
     net = NeRF()
     # with SummaryWriter(comment='MutipleInput') as w:
     #     w.add_graph(net,(x,dir),)
     alpha,rgb = net(x,dir)
     print(alpha.shape,rgb.shape)
