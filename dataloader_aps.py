"""dataset processing and loading
"""
import os
import random

# import imageio
import numpy as np
# import pandas as pd
import torch
import yaml
# from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
# from einops import rearrange
from utils.logger import logger_config
import pdb

# def rssi2amplitude(rssi):
#     """convert rssi to amplitude
#     """
#     return 100 * 10 ** (rssi / 20)


# def amplitude2rssi(amplitude):
#     """convert amplitude to rssi
#     """
#     return 20 * np.log10(amplitude / 100)

def recover_rsrp(rsrp):
    return rsrp*100 - 180

def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)  # rssi~[0,-100] ->[1,0]


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


def rsrp2amp2(rsrp):
    RS = (10 ** (rsrp / 10) / (10 ** (-6))) ** 0.5
    return RS


def rsrp2amp(rsrp):
    amp = rsrp.clone()
    # result[result!=0]
    amp = 1 + (rsrp + 30) / 170
    return amp


def amp2rsrp(amp):
    return -30 - 170 * (1 - amp)


def amp2rsrp2(amp):
    # RS = (10 ** (rsrp / 10) / (10 ** (-5))) ** 0.05
    rsrp = 10 * torch.log10((amp ** 2) * (10 ** (-9)))
    # rsrp = 10*torch.log10((amp**2)*(10**(-5)))
    return rsrp


def split_dataset(datadir, ratio=0.8, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)
    elif dataset_type == "ble":
        rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        index = pd.read_csv(rssi_dir).index.values
        random.shuffle(index)
    elif dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.pt')
        index = [i for i in range(torch.load(csi_dir).shape[0])]
        random.shuffle(index)
    elif dataset_type == "rsrp":
        rsrpdata_dir = os.path.join(datadir, 'RSRP.txt')
        index = [i for i in range(np.loadtxt(rsrpdata_dir).shape[0])]
        random.shuffle(index)
    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')

def split_realdataset(datadir, ratio=0.8, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)
    elif dataset_type == "ble":
        rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        index = pd.read_csv(rssi_dir).index.values
        random.shuffle(index)
    elif dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.pt')
        index = [i for i in range(torch.load(csi_dir).shape[0])]
        random.shuffle(index)
    elif dataset_type == "rsrp":
        rsrpdata_dir = os.path.join(datadir, 'rsrp_train.pt')
        index = [i for i in range(torch.load(rsrpdata_dir).shape[0])]
        random.shuffle(index)
    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')

class RSRP_dataset(Dataset):

    def __init__(self, datadir, indexdir, scale_worldsize=1):
        """ datasets [datalen*8, up+down+r_o+r_d] --> [datalen*8, 26+26+3+36*3]
        """
        super().__init__()
        self.datadir = datadir

        self.rsrpdata_dir = './data/RSRP_after.npy'
        self.apsdata_dir='./data/angular_power_spectrum.npy'
  

        # else raise
        self.location = './data/location.txt'
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 91, 72  # resulution of rays
        self.logger = logger_config(log_savepath="log_savepath", logging_name='nerf3_shanghai')
        self.rx_poses = torch.from_numpy(np.loadtxt(
            self.location))  
        self.rx_poses = self.rx_poses / scale_worldsize

        self.APS=torch.from_numpy(np.load(
            self.apsdata_dir))
        self.RSRPs = torch.from_numpy(np.load(
            self.rsrpdata_dir))  # torch.tensor(pd.read_csv(self.rssi_dir).values, dtype=torch.float32)  #[N, 8, 52]


        self.nn_inputs, self.nn_labels, self.APSs = self.load_data()

    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, ?975]. The inputs for training
                    ray_o:3 , ray_d:9*36*3

        nn_labels : tensor. [n_samples, n_rsrp = 40]. The RSRP as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3 + 3 * self.alpha_res * self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 32)), dtype=torch.float32)
        nn_APSs = torch.tensor(np.zeros((len(self), 6552)), dtype=torch.float32)

        ## generate rays origin at gateways
        rx_ray_o, rx_rays_d = self.gen_rays_gateways()
        ###bs_ray_o = rearrange(bs_ray_o, 'n g c -> n (g c)')   # [n_bs, 1, 3] --> [n_bs, 3]
        ###bs_rays_d = rearrange(bs_rays_d, 'n g c -> n (g c)') # [n_bs, n_rays, 3] --> [n_bs, n_rays*3]

        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):  # sample from dataset_index
            #  rsrps = self.RSRPs[idx]
            # tx_pos = self.tx_poses[idx].view(-1)  # [3]
            # for i_gateway, rsrp in enumerate(rsrps):
            # for i_grid in enumerate(self.RSRPs):
            #  if rsrp != 0:
            rx_ray_o1 = rx_ray_o[idx].view(-1)  # [3]
            rx_rays_d1 = rx_rays_d[idx].view(-1)  # [n_rays x 3]
            nn_inputs[data_counter] = torch.cat([rx_ray_o1, rx_rays_d1], dim=-1)
            nn_labels[data_counter] = self.RSRPs[idx]  # 40
            nn_APSs[data_counter]=self.APS[idx]
            data_counter += 1

        # print("1",nn_labels.shape )
        # self.logger.debug(f"nn_labels:{nn_labels}")

        # nn_labels = rsrp2amp(nn_labels)
        # print("rsrp2amp:0", nn_labels)
        # nn_labels[nn_labels!=0] = rsrp2amp(nn_labels[nn_labels!=0])
        # print("rsrp2amp:1", nn_labels)
        # print("0", nn_labels.shape)
        # nn_labels[nn_labels!=0] += 50

        # return nn_inputs, 100 + (nn_labels)
        # return nn_inputs, ((nn_labels)+180)/100
        return nn_inputs, nn_labels, nn_APSs
        # return nn_inputs, rsrp2amp(nn_labels)

    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_rx, 1, 3]. The origin of rays
        r_d : tensor. [n_rx, n_rays, 3]. The direction of rays, unit vector
        """
        alphas = (-63.5 + 90 + torch.linspace(90, -265, self.alpha_res)) / 180 * np.pi
        # 36->48
        betas = (-11.5 + torch.linspace(90, -90, self.beta_res)) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)  # [0,10,20,30,....]
        betas = betas.repeat_interleave(self.alpha_res)

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)
        

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.rx_poses.shape[0], self.beta_res * self.alpha_res, 3])  # [ n_rx,9*36, 3]
        r_o = self.rx_poses.unsqueeze(1)  # [n_rx, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d

    def __len__(self):
        # RSRPS = self.rssis[self.dataset_index]
        return len(self.dataset_index)  # *  39#torch.sum(rssis != -100)

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index], self.APSs[index]




    



# dataset_dict = { "ble": BLE_dataset, "mimo": CSI_dataset, "rsrp": RSRP_dataset}
dataset_dict = { "rsrp": RSRP_dataset}