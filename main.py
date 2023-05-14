import torch
import matplotlib.pyplot as plt
import numpy as np
# from Network import NeRF
# from Sample import sample_rays_np
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
