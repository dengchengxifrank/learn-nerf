import numpy as np


def sample_rays(H, W, focal, camera_to_world):

     # 用 np.meshgrid来创建网格,实际上size和原始的img是一样的大小 i,j分别成为了图像宽和搞的索引
     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
     dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
     rays_d = np.sum(dirs[..., None, :] * camera_to_world[:3, :3], -1)
     rays_tmp = camera_to_world[0:3, 3]  # first [-0.05379832  3.8454704   1.2080823   1.]
     rays_o = np.broadcast_to(rays_tmp, (np.shape(rays_d)))

     return rays_o,rays_d




     

