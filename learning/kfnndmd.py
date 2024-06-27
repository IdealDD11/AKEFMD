# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 22:08:27 2022

@author: Ideal
"""
import numpy as np
import torch
import torch.nn as nn
# import gym
# from torch.utils.data import Dataset, DataLoader
import control
import os
# from ReplayBuffer import ReplayBuffer
# import time
import argparse
import sys  
sys.path.append('D:/cpython.p_y/DKP/keedmd-master') 
import scipy
from scipy.integrate import odeint
from .edmd import Edmd
from sklearn import linear_model
from numpy import array, concatenate, zeros, dot, linalg, eye, diag, std, divide, tile, multiply, atleast_2d, ones, zeros_like
from torch import nn, cuda, optim, from_numpy, manual_seed, mean, transpose as t_transpose, mm, bmm, matmul, cat

parser = argparse.ArgumentParser()
# parser.add_argument("--env_name", default='InvertedPendulum-v2')
parser.add_argument("--model_name", default='KF_vdp')  ##2:10,64  11:10,64

parser.add_argument("--max_iter", default=100)
parser.add_argument("--hidden_dim", default=10, type=int)
# parser.add_argument("--mode", default="train")
parser.add_argument("--mode", '-false')
args = parser.parse_args()

## 1 iter 200 epoch 100  random input  stable 16  hidden  4  sample1 cha  64
## 2 iter 100 epoch 50   ones input   stable 16 hidden 4   sample1 cha
## 3 iter 200 epoch 100   ones input  stable 64   hidden   6  sample1   gou
## 4 iter 100 epoch 100   ones input  stable 64   hidden   6  sample2   cha
## 5 iter 200 epoch 100   ones input  stable 64   hidden   8  sample1   cha  64
## 5 iter 100 epoch 100   ones input  stable 64   hidden   6  sample1   128

class KfNNdmd(Edmd):
    def __init__(self, basis, system_dim,model_name = "KF_vdp", l1_pos=0., l1_ratio_pos=0.5, l1_vel=0., l1_ratio_vel=0.5, l1_eig=0., l1_ratio_eig=0.5, acceleration_bounds=None, override_C=True, K_p = None, K_d = None, episodic=False):
        super().__init__(basis, system_dim, l1=l1_vel, l1_ratio=l1_ratio_vel, acceleration_bounds=acceleration_bounds, override_C=override_C)
        self.episodic = episodic
        self.K_p = K_p
        self.K_d = K_d
        self.Z_std = ones((basis.Nlift + basis.n, 1))
        self.l1_pos = l1_pos
        self.l1_ratio_pos = l1_ratio_pos
        self.l1_vel = l1_vel
        self.l1_ratio_vel =  l1_ratio_vel
        self.l1_eig = l1_eig
        self.l1_ratio_eig = l1_ratio_eig

        if self.basis.Lambda is None:
            raise Exception('Basis provided is not an Koopman eigenfunction basis')
            
    # def __init__(self, model_name = "KF_vdp", hidden_dim = 3):
        self.model_name = model_name
        # self.env = gym.make(env_name)
        # self.state_dim = self.env.observation_space.shape[0]+1
        self.state_dim=basis.n
        self.hidden_dim = basis.Nlift
        self.n_lift=basis.n+basis.Nlift
        # self.stable_dim=64
        # self.action_dim = self.env.action_space.shape[0]
        self.action_dim=1

        
        self.propagate = nn.Linear(self.n_lift+self.action_dim, self.state_dim, bias = False)
        self.propagatee = nn.Linear(self.action_dim, self.hidden_dim, bias = False)
        
        self.lambda1 = 1.0
        self.lambda2 = 0.3
        
        self.dt = 0.01
        
        # self.replay_buffer = ReplayBuffer(100000)
    
    def get_system(self):       
        weight = self.propagate.weight.data.numpy()
        AA = weight[:, :self.state_dim+self.hidden_dim]
        B1 = weight[:, self.state_dim+self.hidden_dim:]
        B2 = self.propagatee.weight.data.numpy()
        B=concatenate((B1,B2),axis = 0)
        A = zeros((self.n_lift, self.n_lift))
        A[:self.n, :] = AA  
        A[self.n:, :self.n] =- dot(B[self.n:, :], concatenate((self.K_p, self.K_d), axis=1))
        A[self.n:, self.n:] = diag(self.basis.Lambda)
        return A, B
    
    def forward(self,l,ZZ_batch,UU_batch,U_nomm_batch,Lambda_matrix):
        # gt = self.encoder(xt)
        # xt_ = self.decoder(gt)
        Z,U,U_nom=ZZ_batch[0],UU_batch[0],U_nomm_batch[0]
        gtdot1 = self.propagate(torch.cat((Z, U), axis = -1))
        gtdot2 = self.propagatee((U-U_nom))+mm(Z[:,self.state_dim:],Lambda_matrix)
        Z_dot_hat=torch.cat((gtdot1, gtdot2), axis = -1)
        gt1 = Z + self.dt*Z_dot_hat
        zz=[gt1]
        for i in range(l):
            U=UU_batch[i+1]
            U_nom=U_nomm_batch[i+1]
            gtdot1 = self.propagate(torch.cat((gt1, U), axis = -1))
            gtdot2 = self.propagatee(U-U_nom)+mm(gt1[:,self.state_dim:],Lambda_matrix)
            gtdot=torch.cat((gtdot1, gtdot2), axis = -1)
            gt1 = gt1 + self.dt*gtdot
            zz.append(gt1)
        # Z_kl_hat= gt1    
        # return  Z_dot_hat,Z_kl_hat,zz
        return zz
    
    def save(self):
        if not os.path.exists("weights/"):
            os.mkdir("weights/")
        file_name = "weights/" + self.model_name + ".pt"
        # file_name = "weights/" + 'robot-ae' + ".pt"
        torch.save({"propagate" : self.propagate.state_dict(),
                    "propagatee" : self.propagate.state_dict()}, file_name)
        print("save model to " + file_name)
    
    def load(self):
        try:
            if not os.path.exists("weights/"):
                os.mkdir("weights/")
            file_name = "weights/" + self.model_name + ".pt"
            # file_name = "weights/" + 'robot-ae' + ".pt"
            checkpoint = torch.load(file_name)
            self.propagate.load_state_dict(checkpoint["propagate"])
            self.propagate.load_state_dict(checkpoint["propagatee"])
            print("load model from " + file_name)
        except:
            print("fail to load model!")
            
###add batch size
    def sample(self,l,Z):
        ##parameters
        tr=100
        lent=int(Z.shape[1]/tr)
        lent_new=lent-l
        zz=[]
        for j in range(l+1):
            z_new=zeros((Z.shape[0],int(Z.shape[1]-tr*(l-1))))
            for i in range(tr):
                z_new[:,int(i*lent_new):int((i+1)*lent_new)]=Z[:,int(i*lent)+j:int(i*lent+lent_new)+j]
            zz.append(z_new)                
        return zz                                                                                                                                               

    def addbatch(self,l,batch_size,ZZ, UU, U_nomm):
        rr=array(ZZ[0]).shape[1]-batch_size
        rrr=np.random.randint(0, rr, size=1)
        r=rrr[0]
        zz,uu,uunom=[],[],[]
        for i in range(l+1):   
            a=array(ZZ[i])
            zz.append(array(a[:,r:r+batch_size]).reshape(batch_size, -1))
            a=array(UU[i])
            uu.append(array(a[:,r:r+batch_size]).reshape(batch_size, -1))
            a=array(U_nomm[i])
            uunom.append(array(a[:,r:r+batch_size]).reshape(batch_size, -1))
        # print(Z1.shape,U1.shape,U_nom1.shape,Z_kl1.shape,U_kl1.shape)
               
        return zz,uu,uunom                                                                                                                                         

    
    def train(self, Z, Z_dot, U, U_nom, max_iter=50,lr =0.001):
        mseloss = nn.MSELoss()
        # l1loss = nn.L1Loss()
        l=10
        batchsize=64
        ZZ, UU, U_nomm = self.sample(l,Z),self.sample(l,U),self.sample(l,U_nom)
        
        propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr = lr)
        propagatee_optimizer = torch.optim.Adam(self.propagatee.parameters(), lr = lr)

        for it in range(max_iter):
            loss_hist = []
            for _ in range(100):
    
                ZZ_batch,UU_batch,U_nomm_batch = self.addbatch(l,batchsize,ZZ, UU, U_nomm)
                Lambda_matrix=diag(self.basis.Lambda)

                ZZ_batch = torch.FloatTensor(ZZ_batch)
                UU_batch = torch.FloatTensor(UU_batch)
                U_nomm_batch = torch.FloatTensor(U_nomm_batch)
                Lambda_matrix = torch.FloatTensor(Lambda_matrix)                

                zz = self.forward(l,ZZ_batch,UU_batch,U_nomm_batch,Lambda_matrix)
                total_loss=0
                for i in range(l):
                    total_loss=total_loss+mseloss(ZZ_batch[i+1],zz[i])
                                       
                # one_step_loss = mseloss(Z_dot_batch, Z_dot_hat)
                # multi_step_loss = mseloss(Z_kl_batch, Z_kl_hat)
                # # metric_loss = l1loss(torch.norm(gt1-gt, dim=1), torch.norm(xt1-xt, dim=1))
                # #reg_loss = torch.norm(self.propagate.weight.data[:, self.hidden_dim:])
                # total_loss =  self.lambda1*one_step_loss + self.lambda2*multi_step_loss
                                
                propagate_optimizer.zero_grad()
                propagatee_optimizer.zero_grad()
                
                total_loss.backward()
                
                propagate_optimizer.step()
                propagatee_optimizer.step()
                loss_hist.append(total_loss.detach().numpy())
            print("epoch: %d, loss: %2.5f" % (it, np.mean(loss_hist)))
            # for i in range(5):
            #     self.policy_rollout()
            # for i in range(5):
            #     self.random_rollout()

# if __name__ == "__main__":
#     model = KfNNdmd(args.model_name, args.hidden_dim)

#     if args.mode == "train":
#         model.train(args.max_iter, 0.001)
#         model.save()
#     else:
#         model.load()
#         A, B = model.get_system()

        
























