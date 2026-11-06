# -*- coding: utf-8 -*-
"""code for ray marching and signal rendering
"""
import torch
import numpy as np
import torch.nn.functional as F
import scipy.constants as sc
import pdb
# from einops import rearrange, repeat
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def build_rsrp( shs,degree, rays_d_normarlized):
        # rays_d_normarlized=view/view.norm(dim=3,keepdim=True)
        # pdb.set_trace()
        rsrp = eval_sh(degree, shs, rays_d_normarlized)
      
        # if rsrp.shape
        rsrp = (rsrp + 0.5).clip(min=0.0)
        return rsrp

class Renderer():

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """

        ## Rendering parameters
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']
        self.near = kwargs['near']
        self.far = kwargs['far']


    def sample_points(self, rays_o, rays_d,world_scaler=300):
        """sample points along rays

        Parameters
        ----------
        rays_o : tensor. [n_rays, 3]. The origin of rays
        rays_d : tensor. [n_rays, 3]. The direction of rays

        Returns
        -------
        pts : tensor. [n_rays, n_samples, 3]. The sampled points along rays
        t_vals : tensor. [n_rays, n_samples]. The distance from origin to each sampled point
        """
        # pdb.set_trace()
        shape = list(rays_o.shape)
        shape[-1] = 1
        near, far = torch.full(shape, self.near), torch.full(shape, self.far)
        t_vals = torch.linspace(0.001, 1., steps=self.n_samples) * (far - near) + near  # scale t with near and far
        t_vals = (world_scaler* t_vals).to(rays_o.device)
        pts = world_scaler*rays_o[...,None,:] + rays_d[...,None,:].to(rays_o.device) * t_vals[...,:,None] # p = o + td, [n_rays, n_samples, 3]
        # pdb.set_trace()

        return pts, t_vals


class Renderer_RSRP(Renderer):
    """Renderer for RSSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)


    # def render_x(self, tx, rays_o, rays_d, angle_index):
    # def render_x(self, rays_o, tx,  rays_d, angle_index):
    def render_x(self, rays_o, tx,  rays_d,angle_index):
        """render the RSSI for each gateway. To avoid OOM, we split the rays into chunks

        Parameters
        ----------
        tx: tensor. [batchsize, 3]. The position of the transmitter
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 6552]. The direction of rays
        """
        # pdb.set_trace()
        batchsize, _ = tx.shape
        rays_d= torch.reshape(rays_d, (-1, 3))
        # rays_d = rays_d[angle_index, :]
        # rays_d = torch.reshape(rays_d, (batchsize, -1, 3))  # [batchsize, 9x36, 3]
        # rays_d = rays_d[..., angle_index, :]
        chunks =100  # 600
        chunks_num = angle_index.shape[0] // chunks
        rays_o_chunk= rays_o.expand(chunks, -1)
        # rays_o = rays_o.expand(batchsize, -1).cuda()
        # rays_o_chunk = rays_o.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
        tags_chunk = tx.expand(chunks, -1, -1).permute(1, 0, 2).cuda()  # [bs, cks, 3]
        recv_signal = torch.zeros(batchsize, 6552).cuda()
        # pdb.set_trace()
        
        L1_pc = 0
        for i in range(chunks_num):
            rays_d_chunk = rays_d[ i * chunks:(i + 1) * chunks, :]  # [bs, cks, 3]
            
            pts, t_vals = self.sample_points(rays_o_chunk, rays_d_chunk)  # [bs, cks, pts, 3]
            # pdb.set_trace()
            tx_chunk = tags_chunk[..., None, :].expand(tags_chunk.shape[:1]+pts.shape)  # [bs, cks, pts, 3]

            # views_chunk = rays_d_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
            # pdb.set_trace()
            
            # rays_d_chunk = rays_d[:, i * chunks:(i + 1) * chunks, :]  # [bs, cks, 3]
            # ##########################################
            # pts, t_vals = self.sample_points(rays_o_chunk, rays_d_chunk)  # [bs, cks, pts, 3]
            # ##########################################
            # # pdb.set_trace()
            # views_chunk = rays_d_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
            # tx_chunk = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
            # pdb.set_trace()

            # Run network and compute outputs
            ##########################################
            raw, pc0,degree = self.network_fn(pts, tx_chunk)  # [batchsize, chunks, n_samples, 4]
            # pdb.set_trace()
           
            ##########################################
            # estimated_depth = torch.zeros(pc0.shape[0], pc0.shape[1]).cuda() #从基站作为接收机出发，不同的栅格都一样没有影响
            depth = torch.zeros(pc0.shape[0]).cuda()
           
            for iteration in range(pc0.shape[0]):
                temp = torch.nonzero(pc0[iteration, :, :])
                # pdb.set_trace()
                

               
                if len(temp) > 0:
                    # depth[iteration] = torch.min(temp[:,0])/pc0.shape[2]
                    # pdb.set_trace()
                    # print(torch.min(temp[:,0]))
                    depth[iteration] = torch.min(temp[:,0])*(t_vals[0,1]-t_vals[0,0])
                    
                    # pdb.set_trace()
                    # estimated_depth[iteration] = 
                    # print(torch.min(temp[:,0])/pc0.shape[2])
                    # pdb.set_trace()
                else:
                    # estimated_depth[iteration] = pc0.shape[1] + 1
                    # depth[iteration] = 1.5
                    depth[iteration] = 300
                    # pc0.shape[1] + 1
                # print(depth[iteration])
            # pdb.set_trace()

            # depth=depth/300
            # pdb.set_trace()
            # normalize_estimated_depth = estimated_depth / (estimated_depth.max())
            # pdb.set_trace()
            views_chunk=rays_d_chunk
            # pdb.set_trace()
            recv_signal_chunks, estimated_depth = self.raw2outputs_signal(raw, t_vals,views_chunk,degree, rays_d_chunk)
            # pdb.set_trace()
            # pdb.set_trace()  # [bs]
            # rec_signal_chunks_new=self.network_fn.final_out(recv_signal_chunks)
            # pdb.set_trace()
            recv_signal[...,angle_index[i * chunks:(i + 1) * chunks]] = recv_signal_chunks
            # normalize_depth = depth[0, :] / (depth.max()+1e-6)
            # normalize_depth = depth / (depth.max()+1e-6)
            # L1_pc += torch.mean(abs(estimated_depth - depth))
            # pdb.set_trace()
            L1_pc+=(torch.sum((estimated_depth - depth)**2)/rays_d.shape[0])/estimated_depth.shape[0]
            # L1_pc +=torch.mean(torch.norm(estimated_depth[0] - depth,2))
            if torch.isnan(L1_pc):
        
                pdb.set_trace()
        

        return recv_signal, L1_pc/chunks_num  # [batchsize,]

    def raw2outputs_signal(self, raw, r_vals,view,degree, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Parameters
        ----------
        raw : [batchsize, chunks,n_samples,  4]. Prediction from model.
        r_vals : [batchsize, chunks, n_samples]. Integration distance.
        rays_d : [batchsize,chunks, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize]. abs(singal of each ray)
        """
 

        # pdb.set_trace()

        dists = (r_vals[...,1:] - r_vals[...,:-1])
        # pdb.set_trace()
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
        # pdb.set_trace()
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].
        
        # pdb.set_trace()

        # validsigma3, _, s_a, s_p = raw[...,0]+1e-30, raw[...,1]+1e-30, raw[...,2]+1e-30, raw[...,3:]+1e-30    # [batchsize,chunks, N_samples]
        validsigma3, s_a, s_p = raw[...,0], raw[...,1], raw[...,2:]    # [batchsize,chunks, N_samples]
        # att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [batchsize,chunks, N_samples]
        # att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi
        # att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))
        # amp = raw2amp(att_a, dists)  # [batchsize,chunks, N_samples]
        # phase = raw2phase(att_p, dists)
        sigma=torch.nn.functional.relu(validsigma3 ) 
        alpha = 1.0 - torch.exp(-sigma * dists)
        # alpha = 1.0 - torch.exp(- torch.nn.Softplus()(validsigma3 ) * dists)
        # pdb.set_trace()
        T=self.cumprod_exclusive((1. - alpha) + 1e-10, dim=-1)
        weights = alpha * T
        # pdb.set_trace()
        # depth = (weights * r_vals).sum(dim=-1)  # [n_rays]
        tn=r_vals[None,:].expand(T.shape[0],T.shape[1],T.shape[2])
        gp=r_vals[0,1]-r_vals[0,0]
        tnn=tn+gp
        
        # pdb.set_trace()
        depth=(T*(alpha/(sigma+1e-10)+tn-tnn*torch.exp(-sigma*gp))).sum(dim=-1)
        # pdb.set_trace()

        
        
        # phs_shift = torch.exp(-1j * (2 * np.pi * fc * 1e9 / 3e8) * dists)
        # Compute weight for each sample along each ray. [n_rays, n_samples]
        # coeffs = alpha * self.cumprod_exclusive((1. - alpha) * phs_shift + 1e-10, dim=-1)
        # amp_decay = 3e8 / (r_vals * fc * 1e9 * 4 * np.pi)
        # amp_decay = 3e8 / (r_vals * fc * 1e9 * 4 * torch.pi  + 1e-10)

        # att_i = torch.cumprod(torch.cat([torch.ones((al_shape[:-1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        # amp_i = torch.exp(torch.cumsum(amp, -1))            # [batchsize,chunks, N_samples]
        # phase_i = torch.exp(1j*torch.cumsum(phase, -1))                # [batchsize,chunks, N_samples]
        #
        # recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*amp_i*phase_i, -1)  # integral along line [batchsize,chunks]
        # recv_signal = torch.sum(recv_signal, -1)   # integral along direction [batchsize,]
        s_phi = torch.sigmoid(s_a )  # [N_dir, N_samples]
        s_coef = torch.sigmoid(s_p )  # [N_dir, N_samples]
        # im_ch = torch.tanh(rgb[..., 1])  # [N_dir, N_samples]
        # Produce CFR
        # pdb.set_trace()
        # sum_along_rays = abs(torch.sum((re_ch + 1j * im_ch) * amp_decay * coeffs, dim=-1))
        
        viewt=view[..., None, :].expand(s_coef.shape[1:3]+view.shape[-1:])
        viewtt=viewt[None,...].expand(s_coef.shape[:1]+viewt.shape)
        # pdb.set_trace()
        signal= build_rsrp(s_coef,degree,viewtt)*torch.exp(1j * s_phi)
        # pdb.set_trace()
        # pdb.set_trace()
        sum_along_rays = abs(torch.sum(signal * weights, dim=-1))**2
        # sum_along_rays = abs(torch.sum(signal * amp_decay * coeffs, dim=-1))
        # pdb.set_trace()

        return sum_along_rays , depth


    def cumprod_exclusive(
        self,
        tensor: torch.Tensor,
        dim
    ) -> torch.Tensor:
        # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
        cumprod = torch.cumprod(tensor, dim)
        # "Roll" the elements along dimension 'dim' by 1 element.
        cumprod = torch.roll(cumprod, 1, dim)
        # Replace the first element by "0".
        cumprod[..., 0] = torch.zeros(tensor.shape[:-1])

        return cumprod





# renderer_dict = {"spectrum": Renderer_spectrum, "rssi": Renderer_RSSI, "csi": Renderer_CSI,'rsrp':Renderer_RSRP}
renderer_dict = {'rsrp':Renderer_RSRP}