# -*- coding: utf-8 -*-
"""SM_LSCM runner for training and testing
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from scipy.optimize import nnls
import argparse
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import yaml
# from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import scipy.io as scio

# from dataloader import *
from dataloader_aps_new import *
# from dataloader_rotate import *
from SM_LSCM_model import *
from SM_LSCM_renderer import renderer_dict
from utils.data_painter import *
import time
from utils.logger import logger_config
import pdb
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Using Xavier initialization
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SM_LSCM_Runner():

    def __init__(self, mode, dataset_type, **kwargs) -> None:

        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['networks']
        kwargs_train = kwargs['train']
        self.dataset_type = dataset_type

        ## Path settings
        self.expname = kwargs_path['expname']
        self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')
        self.huberloss=nn.HuberLoss(reduction='sum', delta=1)

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, self.expname, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='SM_LSCM')
        self.logger.info("expname:%s, datadir:%s, logdir:%s", self.expname, self.datadir, self.logdir)
        self.writer = SummaryWriter(os.path.join(self.logdir, self.expname, 'tensorboard'))

        # self.Matrix_A=torch.load('/data-test/user1/NeRF_Point/new_data0918/xiaoqu1_CSI_A.pt').view(-1,6480)
        # self.Base_Station=torch.tensor([5.0, 0.26, 0]).float().to(self.devices)
        self.Base_Station=torch.tensor([-204., -59., 22.10000038]).float().to(self.devices)
        # [5.0-0.85, 0.26, 0]
        
        self.Matrix_A_before = torch.tensor(np.load('./data/Matrix_A_before.npy')).float()
        # pdb.set_trace()
        self.Matrix_A_after = torch.tensor(np.load('./data/Matrix_A_after.npy')).float()
        # self.Matrix_A_after = torch.tensor(np.load(os.path.join(self.datadir,'Matrix_A_reverse_20_2.npy'))).float()
        ## Networks
        self.sm_lscm_network = SM_LSCM(**kwargs_network).to(self.devices)
        self.sm_lscm_network.apply(init_weights)


        params = list(self.sm_lscm_network.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=float(kwargs_train['lr']),
                                          weight_decay=float(kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))
        # pdb.set_trace()
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                        T_max=float(kwargs_train['T_max']), eta_min=float(kwargs_train['eta_min']),
                                                                        last_epoch=-1)

        ## Renderer
        renderer = renderer_dict[kwargs_render['mode']]
        self.renderer = renderer(networks_fn=self.sm_lscm_network, **kwargs_render)
        self.scale_worldsize = kwargs_render['scale_worldsize']

        self.sm_lscm_network.init_point_cloud(pc=torch.load('./outdoor_new_point_density.pt'),
                                            world_sclar=self.scale_worldsize)

        ## Print total number of parameters
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        ## Train Settings
        self.current_iteration = 1
        if kwargs_train['load_ckpt'] or mode == 'test':
            self.load_best_checkpoints()
        if mode != 'test':
            self.batch_size = kwargs_train['batch_size']
        else:
            self.batch_size= 1
        self.total_iterations = kwargs_train['total_iterations']
        self.save_freq = kwargs_train['save_freq']

        self.eps=1e-15

        ## Dataset
        dataset = dataset_dict[dataset_type]
#         train_index = os.path.join(self.datadir, "train_index.txt")
        # train_index = os.path.join(self.datadir, "shanghai_train_index.txt")
        # test_index = os.path.join(self.datadir, "shanghai_test_index.txt")

        # train_index = os.path.join(self.datadir, "shanghai_train_index.txt")

        train_index = './data/train_index_ex.txt'
        test_index = './data/test_index_ex.txt'
        # test_index = os.path.join(self.datadir, "shanghai_train_index.txt")

        
        # if not os.path.exists(train_index) or not os.path.exists(test_index):
            # split_dataset(self.datadir, ratio=0.8, dataset_type=dataset_type)
        self.logger.info("Loading training set...")
        self.train_set = dataset(self.datadir, train_index, self.scale_worldsize)
        self.logger.info("Loading test set...")
        self.test_set = dataset(self.datadir, test_index, self.scale_worldsize)

        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.logger.info("Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))


    def load_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        self.logger.info('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.sm_lscm_network.load_state_dict(ckpt['sm_lscm_network_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']
    

    def load_best_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'pth' in f]
        self.logger.info('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.sm_lscm_network.load_state_dict(ckpt['sm_lscm_network_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']



    def save_checkpoint(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_iteration))
        torch.save({
            'current_iteration': self.current_iteration,
            'sm_lscm_network_state_dict': self.sm_lscm_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname
    
    def save_checkpoint_best(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.pth')]
        if len(model_lst) > 2:
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, 'best_model.pth')
        torch.save({
            'current_iteration': self.current_iteration,
            'sm_lscm_network_state_dict': self.sm_lscm_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname
    
    # def evaluation(self):
    #     self.logger.info("Start rsrp_evaluation")
    #     loss_saved=[]
    #     mae_saved=[]
    #     for test_input, test_label in self.test_iter:
    #         tx,rays_d = test_input[:,:3],test_input[:,3:]
    #         rx = self.Base_Station/self.scale_worldsize
    #         Matrix_A_before = self.Matrix_A_before.view(32,-1)
    #         A_max_indice = find_max_A(Matrix_A_before)
    #                     # pdb.set_trace()
    #                     # predict_x ,L1_pc= self.renderer.render_x(tx, rx, rays_d,A_max_indice)
    #         predict_x ,L1_pc= self.renderer.render_x(rx, tx, rays_d,A_max_indice)
    #         predict_rsrp = 10*torch.log10(Matrix_A_before.cuda()@predict_x.T).T
    def eval_network_rsrp(self):
        with torch.no_grad():
            times=[]
            MAE=0
            MSE=0
            ii=0
            with open(os.path.join(self.logdir, self.expname, "result.txt"), 'w') as f:
                for test_input, test_label,aps in tqdm(self.test_iter):
                    ii+=1
                    tx, rays_d = test_input[:, :3].to(self.devices), test_input[:, 3:].to(self.devices)
                    rx = self.Base_Station/self.scale_worldsize
                    # Matrix_A_before = self.Matrix_A_before
                    A_max_indice = find_max_A(self.Matrix_A_before,2000)
                    # A_max_indice = find_max_A(Matrix_A_before)
                    t1=time.time()
                    predict_x = self.renderer.render_x(rx, tx, rays_d,A_max_indice)
                    t2=time.time()
                    t_gap=(t2-t1)*1000
                    times.append(t_gap)
                    RSRP_dB = test_label.detach().cpu().numpy()
                    # predict_rsrp_dB = 10. * torch.log10(predict_x @ self.Matrix_A_after.T.to(predict_x.device)+self.eps).detach().cpu().numpy()
                    predict_rsrp_dB = 10. * torch.log10(predict_x @ self.Matrix_A_before.T.to(predict_x.device)+self.eps).detach().cpu().numpy()
                    # mae = np.mean(abs(recover_rsrp(predict_rsrp_dB) - (recover_rsrp(RSRP_dB))))
                    mae = np.mean(abs((predict_rsrp_dB) - ((RSRP_dB))))
                    mse = np.mean((predict_x.detach().cpu().numpy()- aps.detach().cpu().numpy())**2)
                    # MAE_error = torch.mean(torch.abs(recover_rsrp(predict_rsrp) - recover_rsrp(train_label)))
                    # save_mae[counter] = mae
                    print('======================')
                    print('Evaluation MAE (dB):', mae)
                    print('Evaluation MSE :', mse)
                    print('Time(ms):',t_gap)
                    print('======================')
                    MAE+=mae
                    MSE+=mse
                MAE_mean=MAE/ii
                MSE_mean=MSE/ii
        print('==================================')
        # print(pc_save.shape)
        # print(bb)
        print('Total Test Mean MAE(dB):',MAE_mean)
        print('Total Test Mean MSE:',MSE_mean)
        print('Total Test Mean Time(ms):',np.mean(times))
        # print(pp)
        # np.save('/data-test/user1/radsplatter/log/radsplatter_0819_new/bad_positions_indx.npy',np.array(indx).squeeze())
        # np.save('/data-test/user1/radsplatter/log/radsplatter_0819_new/bad_positions_mae.npy',np.array(mae_bad_positions).squeeze())
        # np.save('/data-test/user1/radsplatter/log/radsplatter_0819_new/predict_rsrp.npy',np.array( predict_rsrp_db).squeeze())
        print('==================================')
        return MAE_mean


  



    def train(self):
        """train the model
        """
        self.logger.info("Start training. Current Iteration:%d", self.current_iteration)
        loss_saved=[]
        mae_saved=[]
        maee=999999
        bestb_loss=1e10
        mae_=1e10
        indictor=0
        while self.current_iteration <= self.total_iterations:
            with tqdm(total=len(self.train_iter), desc=f"Iteration {self.current_iteration}/{self.total_iterations}") as pbar:
                try:
                     loss_saved.append(np.mean(loss_temp))
                     mae_saved.append(np.mean(mae_temp))
                except:
                    print('Start a new recursion')
                loss_temp=[]
                mae_temp=[]
                times=[]
                for train_input, train_label,_ in self.train_iter:
                    if self.current_iteration > self.total_iterations:
                        break

                    train_input, train_label = train_input.to(self.devices), train_label.to(self.devices)
                    if self.dataset_type == "rfid":
                        rays_o, rays_d, tx_o = train_input[:, :3], train_input[:, 3:6], train_input[:, 6:9]
                        predict_spectrum = self.renderer.render_ss(tx_o, rays_o, rays_d)
                        loss = sig2mse(predict_spectrum, train_label.view(-1))
                    elif self.dataset_type == 'ble':
                        tx_o, rays_o, rays_d = train_input[:, :3], train_input[:, 3:6], train_input[:, 6:]
                        predict_rssi = self.renderer.render_rssi(tx_o, rays_o, rays_d)
                        loss = sig2mse(predict_rssi, train_label.view(-1))
                    elif self.dataset_type == 'mimo':
                        uplink, rays_o, rays_d = train_input[:, :52], train_input[:, 52:55], train_input[:, 55:]
                        predict_downlink = self.renderer.render_csi(uplink, rays_o, rays_d)
                        predict_downlink = torch.concat((predict_downlink.real, predict_downlink.imag), dim=-1)
                        loss = sig2mse(predict_downlink, train_label)
                    elif self.dataset_type=='rsrp' :
                        tx,rays_d = train_input[:,:3],train_input[0,3:]
                        rx = self.Base_Station/self.scale_worldsize
                        Matrix_A_before = self.Matrix_A_before.view(32,-1)
                        A_max_indice = find_max_A(Matrix_A_before,2000)
                        # pdb.set_trace()

                        # train_label_=10**(train_label/10)
                        # pdb.set_trace()
                        # pdb.set_trace()
                        # predict_x ,L1_pc= self.renderer.render_x(tx, rx, rays_d,A_max_indice)
                        # predict_x ,L1_pc= self.renderer.render_x(rx, tx, rays_d,A_max_indice)
                        tic=time.time()
                        predict_x = self.renderer.render_x(rx, tx, rays_d,A_max_indice)
                        predict_rsrp_dB = 10*torch.log10(Matrix_A_before.cuda()@predict_x.T+self.eps).T
                        # predict_rsrp = (Matrix_A_before.cuda()@predict_x.T).T
                        # pdb.set_trace()
                        # loss1=sig2mse(10*torch.log10(predict_rsrp),(train_label+180)/100)
                        # loss1=sig2mse(predict_rsrp,train_label_*1e5)
                        train_label_dB=train_label
                        loss1=sig2mse(predict_rsrp_dB,train_label_dB)

                        # loss1=self.huberloss(predict_rsrp,train_label)
                        loss2=0.0002*torch.norm(predict_x[:,A_max_indice],1)
                        # pdb.set_trace()
                        # loss3=0.05*L1_pc
                        loss4=torch.mean(torch.abs(predict_rsrp_dB - train_label_dB))
                        loss = loss1+loss2+loss4
                        # predict_rsrp_dB= 10*torch.log10(predict_rsrp)*100-180
                        # predict_rsrp_dB= 10*torch.log10(predict_rsrp)
                        # train_label_dB=train_label
                
                        # loss=loss1 
                        # loss = loss1+0.1*L1_pc+loss2
                        # MAE_error = torch.mean(torch.abs(recover_rsrp(predict_rsrp_dB) - recover_rsrp(train_label_dB)))
                        MAE_error =torch.mean(torch.abs(predict_rsrp_dB - train_label_dB))
                        print('==============')
                        # print(lo)
                        # print('loss1:',loss1)
                        # print('loss2:',loss2)
                        # print(loss1)
                        # print('loss3:',loss3)
                        # print('loss4',loss4)
                        # print(L1_pc)
                        print('LOSS',loss)
                        # print(loss)
                        # print('MAE:',MAE_error)
                        print('==============')


                    self.optimizer.zero_grad()
                    loss.backward()
                    toc=time.time()
                    gap=(toc-tic)*1000
                    times.append(gap)
                    print('Time(ms):',gap)
                    self.optimizer.step()
                    self.cosine_scheduler.step()
                    self.current_iteration += 1
                    loss_temp.append(loss.detach().cpu().numpy())
                    mae_temp.append(MAE_error.detach().cpu().numpy())

                    self.writer.add_scalar('X', predict_x.max()/predict_x.mean(), self.current_iteration)
                    self.writer.add_scalar('Loss', loss, self.current_iteration)
                    self.writer.add_scalar('MAE', MAE_error, self.current_iteration)
                    pbar.update(1)
                    pbar.set_description(f"Iteration {self.current_iteration}/{self.total_iterations}")
                    pbar.set_postfix_str('loss = {:.6f}, MAE = {:.6f}, lr = {:.6f}'.format(loss.item(), MAE_error.item(), self.optimizer.param_groups[0]['lr']))

       
                print(np.mean(times))
                # pdb.set_trace()
                if  np.mean(mae_temp)<maee:
                    maee=np.mean(mae_temp)
                    ckptname = self.save_checkpoint()
                    ckptname = self.save_checkpoint_best()
                    print('Save at'+ ckptname)
                    pbar.write('Saved checkpoints at {}'.format(ckptname))
                
                # if loss_saved>=bestb_loss:
                #     patient_i+=1
                # #  print('Early Stop')F
                # #  break
                #     if patient_i>=50:
                #      print('Early Stop')
                #      break
                # if self.current_iteration
                # m=self.eval_network_rsrp()
                # if m<=mae_:
                #     if indictor>0:
                #         indictor=0
                #     mae_=m
                #     ckptname = self.save_checkpoint_best()
                #     print('Save at'+ ckptname)
                # else:
                #     print('Continue')
                #     # if self.current_iteration>=6000:
                #     # if self.current_iteration>=3500:
                #     if self.current_iteration>=0:
                #         indictor+=1
                #         if indictor>=50:
                #             print('Overfitting STOP at:',self.current_iteration)
                #             break


















if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./shanghai_SM_LSCM.yml', help='config file path')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_type', type=str, default='rsrp')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()

    ## backup config file
    if args.mode == 'train':
        logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
        os.makedirs(logdir, exist_ok=True)
        copyfile(args.config, os.path.join(logdir,'config.yml'))

    worker = SM_LSCM_Runner(mode=args.mode, dataset_type=args.dataset_type, **kwargs)
    if args.mode == 'train':
        worker.train()
    elif args.mode == 'test':
        if args.dataset_type == 'rsrp':
            worker.eval_network_rsrp()
  