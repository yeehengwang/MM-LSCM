# -*- coding: utf-8 -*-
"""MM_LSCM NN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
# from einops import rearrange

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2me = lambda x, y : torch.mean(abs(x - y))
sig2mse = lambda x, y : torch.mean((x - y) ** 2)
csi2snr = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1, 2)) ** 2 /
    torch.norm(y, dim=(1, 2)) ** 2
)




class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, is_embeded=True, input_dims=3):
    """get positional encoding function

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 1 for default positional encoding, 0 for none
    input_dims : input dimension of gamma


    Returns
    -------
        embedding function; output_dims
    """
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



class MM_LSCM(nn.Module):

    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts':3, 'view':3, 'tx':3,'pc':1},
                 multires = {'pts':10, 'view':10, 'tx':10,'pc':30},
                 is_embeded={'pts':True, 'view':True, 'tx':False,'pc':True},
                 attn_output_dims=1):
        """NeRF2 model

        Parameters
        ----------
        D : int, hidden layer number, default by 8
        W : int, Dimension per hidden layer, default by 256
        skip : list, skip layer index
        input_dims: dict, input dimensions
        multires: dict, log2 of max freq for position, view, and tx position positional encoding, i.e., (L-1)
        is_embeded : dict, whether to use positional encoding
        attn_output_dims : int, output dimension of attenuation
        sig_output_dims : int, output dimension of signal
        """

        super().__init__()
        self.skips = skips

        # set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])
        self.embed_pc_fn, input_pc_dim = get_embedder(multires['pc'], is_embeded['pc'], input_dims['pc'])

        ## attenuation network
        # self.attenuation_linears = torch.nn.Sequential(nn.Linear(input_pts_dim, W), torch.nn.ReLU(inplace=True),
        #                                                nn.Linear(W, W), torch.nn.ReLU(inplace=True),
        #                                                nn.Linear(W, W), torch.nn.ReLU(inplace=True),
        #                                                nn.Linear(W+input_pc_dim, W), torch.nn.ReLU(inplace=True),
        #                                                nn.Linear(W+input_pts_dim, W), torch.nn.ReLU(inplace=True),
        #                                                nn.Linear(W, W), torch.nn.ReLU(inplace=True),
        #                                                nn.Linear(W+input_pc_dim, W), torch.nn.ReLU(inplace=True),
        #                                                nn.Linear(W, W), torch.nn.ReLU(inplace=True),
        #
        #                                                )

        # self.final_output_linears=nn.ModuleList(
        #     [nn.Linear(100, 50)]+
        #     [nn.Linear(50, 100)]
        # )

        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W)] +
            [nn.Linear(W, W)] +
            [nn.Linear(W, W)] +
            [nn.Linear(W+input_pc_dim, W)] +

            [nn.Linear(W+input_pts_dim, W)] +

            [nn.Linear(W, W)] +
            [nn.Linear(W+input_pc_dim, W)]

            # [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
            #  for i in range(D - 1)]
        )


        ## signal network
        self.signal_linears = nn.ModuleList(
            [nn.Linear( input_tx_dim + W, W)] +
            [nn.Linear(W, W//2)]
        )

        self.degree=3

        ## output head, 2 for amplitude and phase
        self.attenuation_output = nn.Linear(W, attn_output_dims)
        self.feature_layer = nn.Linear(W, W)
        # self.signal_output = nn.Linear(W//2, sig_output_dims)

        self.signal_output = nn.Linear(W//2, (self.degree+1)**2+1)

        


    def init_point_cloud(self,pc,world_sclar):
        self.point_cloud=pc/pc.max()
        self.world_scale = world_sclar

    @torch.no_grad()
    def compute_PC_feature(self, xyz,world_scale=300):

        matrics = self.point_cloud.to(xyz.device)
        world_scale = self.world_scale
        # 根据输入坐标计算插值权重
        self.aabb = torch.tensor([[-300., -350., -2.], [300., 350., 111.]])
# 
        # pdb.set_trace()

        # x, y, z = (-self.aabb[0].to(xyz.device) +  xyz).unbind(dim=-1)
        x, y, z = (-self.aabb[0].to(xyz.device) + world_scale*xyz).unbind(dim=-1)
        z = (z) * 0.5  # 0->0
        y = (y) * 0.5
        x = (x) * 0.5  # * (width - 1)

        # pdb.set_trace()

        # c =  trilinear_interpolation(self.point_cloud,x,y,z)
        #
        # def trilinear_interpolation(matrics, x, y, z):
        x = torch.clamp(x, 0, matrics.shape[0] - 1)
        y = torch.clamp(y, 0, matrics.shape[1] - 1)
        z = torch.clamp(z, 0, matrics.shape[2] - 1)

        x0 = torch.floor(x).long()
        x1 = torch.where(x0 < matrics.shape[0] - 1, x0 + 1, x0)
        y0 = torch.floor(y).long()
        y1 = torch.where(y0 < matrics.shape[1] - 1, y0 + 1, y0)
        z0 = torch.floor(z).long()
        z1 = torch.where(z0 < matrics.shape[2] - 1, z0 + 1, z0)

        wx = x - x0.float()
        wy = y - y0.float()
        wz = z - z0.float()
        N = x.shape[0]
        c000 = matrics[x0, y0, z0]
        c001 = matrics[x1, y0, z0]
        c010 = matrics[x0, y1, z0]
        c011 = matrics[x1, y1, z0]
        c100 = matrics[x0, y0, z1]
        c101 = matrics[x1, y0, z1]
        c110 = matrics[x0, y1, z1]
        c111 = matrics[x1, y1, z1]

        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        c = c0 * (1 - wz) + c1 * wz
        # pdb.set_trace()

        return c

    #     return x


    def forward(self, pts, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_samples, 3], position of voxels
        view: [batchsize, n_samples, 3], view direction
        tx: [batchsize, n_samples, 3], position of transmitter

        Returns
        ----------
        outputs: [batchsize, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """
       
        pts /= self.world_scale
    
        pc0 = self.compute_PC_feature(pts,self.world_scale).unsqueeze(-1)
        # pdb.set_trace()
       
        # position encoding

        pts = self.embed_pts_fn(pts).contiguous()
        # view = self.embed_view_fn(view).contiguous()
        tx = self.embed_tx_fn(tx).contiguous()
        pc = self.embed_pc_fn(pc0).contiguous()
        # pdb.set_trace()

        # pdb.set_trace()

       

        # shape = pts.shape
        # pts = pts.view(-1, list(pts.shape)[-1])
        # pc = pc.view(-1, list(pc.shape)[-1])
        # # view = view.view(-1, list(view.shape)[-1])
        # tx = tx.view(-1, list(tx.shape)[-1])

        # pdb.set_trace()

        x = pts
        for i, layer in enumerate(self.attenuation_linears):
            # print(i)
            # pdb.set_trace()
            x = F.relu(layer(x))
            # pdb.set_trace()
            # print(i)
            if i in self.skips:
                x = torch.cat([pts, x], -1)
            if i == 4-1 or i==7-1:
                x = torch.cat([pc,x],-1)
        # pdb.set_trace()

        
        attn = self.attenuation_output(x)    # (batch_size, 2)
        feature = self.feature_layer(x)
        featuret=feature[None, ...].expand(tx.shape[:1]+feature.shape) 
        # pdb.set_trace()
        x = torch.cat([featuret, tx], -1)
        # pdb.set_trace()

        for i, layer in enumerate(self.signal_linears):
            # pdb.set_trace()
            x = F.relu(layer(x))
        signal = self.signal_output(x)    #[batchsize, n_samples, sh_coeff+1]
        attnt=attn[None, ...].expand(tx.shape[:1]+attn.shape) 
        # pdb.set_trace()

        outputs = torch.cat([attnt, signal], -1).contiguous()    # [batchsize, n_samples, 4]
        # pdb.set_trace()
        # pdb.set_trace()
        return outputs,pc0,self.degree
