import torch
import matplotlib.pyplot as plt
import numpy as np
from Network import NeRF
from Sample_Ray import sample_rays
# from Render import render_rays
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
np.random.seed(3407)
n_train = 100

data = np.load('./tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
print("images.shape:", images.shape)
print("poses.shape:", poses.shape)
print("focal:", focal)

test_img, test_pose = images[101], poses[101]
images = images[:n_train]   #images  (100, 100, 100, 3) (100张image,H,W,rgb)
poses = poses[:n_train]     #poses  (100, 4, 4)
#print(images.shape,poses.shape)


# plt.imshow(images[0])
# plt.show()
rays_o_list = list()
rays_d_list = list()
rays_rgb_list = list()

#对于 n_train image进行初始化光线的起点和方向
for i in range(n_train):
    img = images[i]
    pose = poses[i]
    # 对每张图像的每个pixel进行远点 ray_o 以及 ray_d 的计算,其中ray_d对于每个pixel而言是一个三维的向量
    # [H,W,3]
    #rays_o , rays_d = sample_rays(H,W,focal,pose)
    rays_o , rays_d = sample_rays(H,W,focal,pose)
    #需要考虑np.reshape的是,多维数组在计算机内部仍然是以一维数组组成的
    #所以reshape的形状只是在一维度数组上的变化index从0并且从内往外增大的过程就是检索原来一维数组的过程
    #rays_o的第一个index是0-99 所以如果我们在第二维开始索引,每一个二维就对应了一个rays_o的三维向量,所以从多维度向量转为1维考虑时也应该是最近的index先转为1维度向量
    #这里-1并不奇怪,-1就看作一个num,不决定reshape排列的具体算法
    rays_o_list.append(rays_o.reshape(-1, 3))

    rays_d_list.append(rays_d.reshape(-1, 3))
    rays_rgb_list.append(img.reshape(-1, 3))

rays_o_npy = np.concatenate(rays_o_list, axis=0)
rays_d_npy = np.concatenate(rays_d_list, axis=0)
rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
#ray在第二维进行了concatenate所以最后dim是9
rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=device) #torch.Size([1000000, 9])


#o,d,rgb采用了相同的处理方式,这样index能对应上
num_rays = rays.shape[0]
#batchsize就是每次渲染多少条光线
batchsize = 4096

#/保留小数部分     //只保留整数部分
iter = num_rays//batchsize
#define distance of near and far
bound = (2., 6.)
#for test
test_rays_o, test_rays_d = sample_rays(H, W, focal, test_pose)
test_rays_o = torch.tensor(test_rays_o, device=device)
test_rays_d = torch.tensor(test_rays_d, device=device)
test_rgb = torch.tensor(test_img, device=device)

net = NeRF.to(device)

#


