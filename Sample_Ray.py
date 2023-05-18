import numpy as np
import torch
import torch.nn.functional as F

def sample_rays(H, W, focal, camera_to_world):
     # 用 np.meshgrid来创建网格,实际上size和原始的img是一样的大小 i,j分别成为了图像宽和搞的索引
     # 并且这里对于方向向量dir只做了旋转,因为在三维空间我们方向只是一个vecter,rotation变换后就只有在dir方向上的伸缩变换
     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
     dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
     rays_d = np.sum(dirs[..., None, :] * camera_to_world[:3, :3], -1)
     rays_tmp = camera_to_world[0:3, 3]  # first [-0.05379832  3.8454704   1.2080823   1.]
     rays_o = np.broadcast_to(rays_tmp, (np.shape(rays_d)))

     return rays_o,rays_d

def  uniform_sample(near,far,N_samples):

     result = torch.linspace(start=near,end=far,steps=N_samples)
     return result


def positional_embedding(input,L):
     #input [input_len,dim]
     input_len = input.shape[-1] #0 dim是batchsize
     output = list()
     for i in range(L):
          for function in [torch.sin,torch.cos]:
               tmp = function(2**i*torch.pi*input)
               output.append(tmp)

     return torch.concat(output,dim=-1)



def render_output(net,pts,rays_o,rays_d,near,z_vals):
     # pts => tensor(Batch_Size, uniform_N, 3)
     # rays_d => tensor(Batch_Size, 3)
     #pts[4096,64,3] z_vals [64]
     space_cordinate = torch.reshape(pts,shape=[-1,3])
     pts_embedding = positional_embedding(space_cordinate,L=10)
     '''
     print('pts',pts_embedding.shape) #pts torch.Size([262144, 60])
     direction_norm = F.normalize(rays_d,p=2,dim=1)
     这里是先在第二维进行增维,然后赋值以下pts的shape,因为原来rays_d是需要考虑
     这里需要对rays_d做归一化,因为不同的位置pts对应了不同的embedding,但是这里rays_d只表示方向,这意味三维空间的同一个方向应该有一样的embedding
     比如一个点(x,y,z) 方向是dir,则已经可以确定最后render的结果,而dir的embedding需要作为network输入,这就意味这需要dir需要有一个规则化的表示
     '''
     direction_norm = F.normalize(torch.reshape(rays_d.unsqueeze(-2).expand_as(pts), [-1, 3]), p=2, dim=-1)
     dir_embedding = positional_embedding(direction_norm,L=4)

     rgb , sigma = net(pts_embedding,dir_embedding)
     # get the interval
     # z_vals 是 near 和 far 之间的采样点 是均匀采样,采了 N_samples 个点
     a = z_vals[..., 1:]  # a 这里是从z_vals 的第二个点到了最后一个点
     b = z_vals[..., :-1]  # b 这里是从这里是从z_vals 的第一个点到了倒数第二个点
     delta = z_vals[..., 1:] - z_vals[..., :-1]
     INF = torch.ones(delta[..., :1].shape).fill_(1e10)
     delta = torch.cat([delta, INF], -1)  # 在delta后面拼接了一个INF
     # 5.1569e-02, 8.6743e-02
     #norm = torch.norm(rays_d, dim=-1, keepdim=True)
     delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)
     sigma = torch.reshape(sigma,shape=(delta.shape[0],delta.shape[1]))
     alpha = 1. - torch.exp(-sigma * delta)
     ones = torch.ones(alpha[..., :1].shape)
     weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]

     return rgb, weights

def Render_rays(net,rays_o,rays_d,bound,N_samples):
     near,far = bound[0],bound[1]
     sample_uniform = N_samples
     batchsize = rays_o.shape[0]
     # [4096,3]
     rays_o = rays_o
     rays_d = rays_d
     z_vals = uniform_sample(near,far,N_samples)
     # pts就是world view下的三维点
     # pts => tensor(Batch_Size, uniform_N, 3)
     # rays_o, rays_d => tensor(Batch_Size, 3)
     # z_vals 是在near 和 far 之间进行的采样 这里是均匀采样Uniform的
     rays_d_norm = F.normalize(rays_d, p=2, dim=-1)

     pts = rays_o[..., None, :] + rays_d_norm[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

     rgb , weights = render_output(net,pts,rays_o,rays_d,near,z_vals)
     rgb = torch.reshape(rgb,shape=(weights.shape[0],weights.shape[1],-1))
     rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

     return rgb_map

# if __name__ == '__main__':
#      input = torch.ones(2,2)
#      res = input[-1]
#      out = positional_embedding(input,L=3)
#      print(out)
#






     

