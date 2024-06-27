# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:38:55 2023

@author: Ideal
"""


from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, cuda, optim, from_numpy, manual_seed, mean, transpose as t_transpose, mm, bmm, matmul, cat
from .coupling_block import CouplingBlock
from .permute_data import Permute_data


class CouplingDiffeomorphismNetImp(nn.Module):

    def __init__(self, n, A_cl, jacobian_penalty = 1e-2, n_hidden_layers = 2, layer_width=50, batch_size = 64, dropout_prob=0.1, traj_input=False):
        super(CouplingDiffeomorphismNetImp, self).__init__()
        self.n = n
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.jacobian_penalty = jacobian_penalty
        self.traj_input = traj_input

        N, H, d_h_out = batch_size, layer_width, self.n
        if self.traj_input:
            self.d_h_in = 2 * self.n
        else:
            self.d_h_in = self.n
            
        self.channel_part_1 = self.d_h_in // 2
        self.channel_part_2 = self.d_h_in - self.d_h_in// 2

        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.A_cl = A_cl.to(self.device)
    
        self.fc_in = nn.Linear(self.channel_part_1, H).double().to(self.device)
        self.fc_hidden1=nn.Linear(H, H).double().to(self.device)
        self.fc_hidden2=nn.Linear(H, H).double().to(self.device)
        self.fc_hidden3=nn.Linear(H, H).double().to(self.device)
        self.fc_hidden = []
        self.fc_hidden.append(self.fc_hidden1)
        self.fc_hidden.append(self.fc_hidden2)
        self.fc_hidden.append(self.fc_hidden3)
        # self.fc_hidden = []
        # for _ in range(self.n_hidden_layers):
        #     self.fc_hidden.append(nn.Linear(H, H).double().to(self.device))
        self.fc_out = nn.Linear(H, self.channel_part_2).double().to(self.device)
        
        ####======
        self.d_h_c=9
        self.NLa1_out = nn.Linear(self.d_h_in, self.d_h_in,False).double().to(self.device)
        self.NLa2_out = nn.Linear(3, 3,False).double().to(self.device)
        self.NLa3_out = nn.Linear(4, 4,False).double().to(self.device)
        self.NLc_out= nn.Linear(self.d_h_c, self.d_h_in,False).double().to(self.device)
        ###====
            # Define diffeomorphism model:
    def diffJaco(self, xt, input_dimension,output_dimension): 
        cur_batch_size = xt.shape[0]           
        h = []
        h.append(self.fc_in(xt))
        for ii in range(self.n_hidden_layers):
            h.append(F.relu(self.fc_hidden[ii](h[-1])))
        h_out = self.fc_out(h[-1])

        # Define diffeomorphism Jacobian model:
        h_grad = self.fc_in.weight
        h_grad = mm(self.fc_hidden[0].weight, h_grad)
        h_grad = h_grad.unsqueeze(0).expand(cur_batch_size, self.layer_width, input_dimension)
        delta = F.relu(self.fc_hidden[0](h[1])).sign().unsqueeze_(-1).expand(cur_batch_size, self.layer_width,input_dimension)
        h_grad = delta * h_grad
        for ii in range(1, self.n_hidden_layers):
            h_grad = bmm(self.fc_hidden[ii].weight.unsqueeze(0).expand(cur_batch_size, self.layer_width, self.layer_width), h_grad)
            delta = F.relu(self.fc_hidden[ii](h[ii+1])).sign().unsqueeze_(-1).expand(cur_batch_size,self.layer_width,input_dimension)
            h_grad = delta * h_grad

        h_grad = bmm(self.fc_out.weight.unsqueeze(0).expand(cur_batch_size,output_dimension,self.layer_width), h_grad)
        return h_grad,h_out
    
    def calcu_h(self, x, sample_the_data=False):
        x1 = x.narrow(1, 0, self.channel_part_1)
        x2 = x.narrow(1, self.channel_part_1, self.channel_part_2)
        cur_batch_size = x.shape[0] 

        if sample_the_data == False:
            # x1_c = torch.cat([x1, c], 1) 
            x1_c=x1
            s_grad,s_out = self.diffJaco(x1_c,self.channel_part_1,self.channel_part_2)
            t_grad,t_out = self.diffJaco(x1_c,self.channel_part_1,self.channel_part_2)
            # y2 = (torch.exp(0.636 *2* torch.atan(self.s_network))) * x2 + self.t_network
            y2 = torch.exp(s_out) * x2 + t_out
            # print(s_grad.shape,s_out.shape,y2.shape)
            output = torch.cat((x1, y2), 1)
            jacobian1=torch.eye(self.channel_part_1).unsqueeze(0).expand(cur_batch_size,self.channel_part_1,self.channel_part_1)
            jacobian2=torch.zeros(self.channel_part_1,self.channel_part_2).unsqueeze(0).expand(cur_batch_size,self.channel_part_1,self.channel_part_2)
            jacobian3 = (s_grad*torch.exp(s_out.unsqueeze(-1)) * x2.unsqueeze(-1)+t_grad)
            jacobian4 = (torch.eye(self.channel_part_2) * torch.exp(s_out).unsqueeze(2)).reshape(cur_batch_size,self.channel_part_2,self.channel_part_2)
            jacobian_output = torch.cat( (torch.cat((jacobian1, jacobian2), 2),torch.cat((jacobian3, jacobian4), 2) ), 1)
            return jacobian_output, output
        else:
            # x1_c = torch.cat([x1, c], 1) 
            x1_c=x1
            self.s_network = self.s_net(x1_c)
            self.t_network = self.t_net(x1_c)
            temp = (x2 - self.t_network) / (torch.exp(0.636 *2* torch.atan(self.s_network)))
            output = torch.cat((x1, temp), 1)
            jacobian1 = torch.sum((0.636 *2* torch.atan(self.s_network )), dim=tuple(range(1, self.input_len+1)))
            self.jacobian_output = (- jacobian1)
            return output

    def forward(self, x):
        xt = x[:, :self.d_h_in]  # [x] 
        xtdot = x[:, self.d_h_in:2 * self.d_h_in]  # [x_dot]
        x_zero = x[:, 2 * self.d_h_in:]
        cur_batch_size = x.shape[0]

        ########## d(x)=d1d2d3...dn(x)
        h_grad,h_out = self.calcu_h(xt)
        
        # h_grad2,h_out2 = self.calcu_h(h_out)
        # h_grad3,h_out3 = self.calcu_h(h_out2)
        # h_grad4,h_out4 = self.calcu_h(h_out3)
        
        # h_grad,h_out =  bmm(bmm(bmm(h_grad, h_grad2),h_grad3) ,h_grad4),   h_out4  
        
        h_dot = bmm(h_grad, xtdot.unsqueeze(-1))
        h_dot = h_dot.squeeze(-1)

        # Calculate zero Jacobian:

        h_zerograd,h_zero_out = self.calcu_h(x_zero)
        
        # h_zerograd2,h_zero_out2 = self.calcu_h(h_zero_out)
        # h_zerograd3,h_zero_out3 = self.calcu_h(h_zero_out2)
        # h_zerograd4,h_zero_out4 = self.calcu_h(h_zero_out3)
        
        # h_zerograd,h_zero_out =  bmm(bmm(bmm(h_zerograd, h_zerograd2),h_zerograd3) ,h_zerograd4),   h_zero_out4 
        
        h_zerograd = h_zerograd[:, :, :self.n]

        y_pred = cat([h_out, h_dot, h_zerograd.norm(p=2,dim=2)], 1)  # [h, h_dot, norm(zerograd)]
        ###########==========begin
        yp1=h_out
        a1_out =self.NLa1_out(yp1)
        yp21= torch.mul(h_out[:,0].unsqueeze(1),h_out[:,1].unsqueeze(1))
        yp2=cat([(h_out[:,0]**2).unsqueeze(1),(h_out[:,1]**2).unsqueeze(1),yp21],1)
        a2_out =self.NLa2_out(yp2)
        yp31= torch.mul(h_out[:,0].unsqueeze(1),(h_out[:,1]**2).unsqueeze(1))
        yp32= torch.mul((h_out[:,0]**2).unsqueeze(1),h_out[:,1].unsqueeze(1))
        yp3=cat([(h_out[:,0]**3).unsqueeze(1),(h_out[:,0]**3).unsqueeze(1),yp31,yp32],1)
        a3_out =self.NLa3_out(yp3)
        c_out =cat([self.NLc_out(cat([a1_out, a2_out, a3_out],1)),self.NLc_out(cat([yp1, yp2, yp3],1))],1)
        c_out_dot=c_out-x
        
        a1=self.NLa1_out.weight.unsqueeze(0).expand(cur_batch_size,2,2)
        # a2=self.NLa2_out.weight.unsqueeze(0).expand(cur_batch_size,3,3)
        # a3=self.NLa3_out.weight.unsqueeze(0).expand(cur_batch_size,4,4)
        # c=self.NLc_out.weight.unsqueeze(0).expand(cur_batch_size,2,9)
        
        y_pred=cat([y_pred,c_out_dot],1)
        y_predd=[y_pred,a1]
        ###########==========end       

        return y_predd

    def diffeomorphism_loss(self, y_true, y_predd,is_training):
        y_pred=y_predd[0]
        h = y_pred[:,:self.n]
        h_dot = y_pred[:,self.n:2*self.n]
        zerograd = y_pred[:,2*self.n:-2*self.n]
        c_out_dot= y_pred[:,-2*self.n:]
        # cur_batch_size = y_pred.shape[0]
        # A_cl_batch = self.A_cl.unsqueeze(0).expand(cur_batch_size, self.n, self.n)
        A_cl_batch = y_predd[1]

        if is_training:
            #return mean((y_true - (h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze()))**2) + self.jacobian_penalty*mean(zerograd**2)
            return mean(
                (y_true + h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze()) ** 2) + self.jacobian_penalty * mean(
                zerograd ** 2)+mean(c_out_dot**2)
        else:
            #return mean((y_true - (h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze())) ** 2)
            return mean(
                (y_true + h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze()) ** 2)

    def predict(self, x):
        x.to(self.device)
        # h = self.fc_in(x)
        # for ii in range(self.n_hidden_layers):
        #     h = F.relu(self.fc_hidden[ii](h))
        # h = self.fc_out(h)
        
        h1,h = self.calcu_h(x)

        return h.detach().numpy()
    

# class CouplingDiffeomorphismNet(nn.Module):

#     def __init__(self, n, A_cl, jacobian_penalty = 1e-2, n_hidden_layers = 2, layer_width=50, batch_size = 64, dropout_prob=0.1, traj_input=False):
#         super(CouplingDiffeomorphismNet, self).__init__()
#         self.n = n
#         self.n_hidden_layers = n_hidden_layers
#         self.layer_width = layer_width
#         self.batch_size = batch_size
#         self.dropout_prob = dropout_prob
#         self.jacobian_penalty = jacobian_penalty
#         self.traj_input = traj_input

#         N, H, d_h_out = batch_size, layer_width, self.n
#         if self.traj_input:
#             self.d_h_in = 2 * self.n
#         else:
#             self.d_h_in = self.n

#         self.device = 'cuda' if cuda.is_available() else 'cpu'
#         self.A_cl = A_cl.to(self.device)
#         self.fc_in = nn.Linear(self.d_h_in, H).double().to(self.device)
#         self.fc_hidden = []
#         for _ in range(self.n_hidden_layers):
#             self.fc_hidden.append(nn.Linear(H, H).double().to(self.device))
#         self.fc_out = nn.Linear(H, d_h_out).double().to(self.device)

#         self.channel_part_1 = self.d_h_in // 2
#         self.channel_part_2 = self.d_h_in - self.d_h_in // 2
#         self.network_s_t = self.fully_connected(self.channel_part_1, self.channel_part_1)
#         # self.t_net = self.network_s_t2(self.channel_part_2, self.channel_part_2)
#         input_dimension1,input_dimension12=self.n,self.n-1
        
#         self.coupling1 = CouplingBlock(self.network_s_t, input_dimension1,input_dimension12)
#         self.coupling2 = CouplingBlock(self.network_s_t, input_dimension1,input_dimension12)
#         self.coupling3 = CouplingBlock(self.network_s_t, input_dimension1,input_dimension12)
#         self.coupling4 = CouplingBlock(self.network_s_t, input_dimension1,input_dimension12)
#         self.coupling5 = CouplingBlock(self.network_s_t, input_dimension1,input_dimension12)
        

#     def convolution_network(self,input_channel, output_channel):
#         return lambda input_channel, output_channel: nn.Sequential(
#                                     nn.Conv2d(input_channel, self.layer_width, 3, padding=1),
#                                     nn.ReLU(),
#                                     nn.Conv2d(self.layer_width, output_channel, 3, padding=1))

#     def fully_connected(self,input_channel, output_channel):
#         return lambda input_channel, output_channel: nn.Sequential(
#                                         nn.Linear(input_channel, self.layer_width).double().to(self.device),
#                                         nn.ReLU(),
#                                         nn.Linear(self.layer_width, output_channel).double().to(self.device))

#     def Couplingblock(self, x, sample_the_data=False):
#         # x1 = x.narrow(1, 0, self.channel_part_1)
#         # x2 = x.narrow(1, self.channel_part_1, self.channel_part_2)

#         # if sample_the_data == False:
#         #     # x1_c = torch.cat([x1, c], 1) 
#         #     x1_c=x1
#         #     self.s_network = self.s_net(x1_c)
#         #     self.t_network = self.t_net(x1_c)
#         #     y2 = (torch.exp(0.636 *2* torch.atan(self.s_network))) * x2 + self.t_network
#         #     output = torch.cat((x1, y2), 1)
#         #     jacobian2 = torch.sum((0.636 *2* torch.atan(self.s_network)), tuple(range(1, self.input_len+1)))
#         #     # self.jacobian_output = jacobian2
#         #     return output,jacobian2
#         # else:
#         #     # x1_c = torch.cat([x1, c], 1) 
#         #     x1_c=x1
#         #     self.s_network = self.s_net(x1_c)
#         #     self.t_network = self.t_net(x1_c)
#         #     temp = (x2 - self.t_network) / (torch.exp(0.636 *2* torch.atan(self.s_network)))
#         #     output = torch.cat((x1, temp), 1)
#         #     jacobian1 = torch.sum((0.636 *2* torch.atan(self.s_network )), dim=tuple(range(1, self.input_len+1)))
#         #     # self.jacobian_output = (- jacobian1)
#         #     return output,- jacobian1
#              #2
#             out12 = self.coupling1(x)
#             jac1 = self.coupling1.jacobian()
#             # out13 = self.permute(out12)

#             out14 = self.coupling2(out12)
#             jac1_c1 = self.coupling2.jacobian()
#             # out15 = self.permute_c1(out14)

#             out16 = self.coupling3(out14)
#             jac1_c2 = self.coupling3.jacobian()
#             # out17 = self.permute_c2(out16)

#             out18 = self.coupling4(out16)
#             jac1_c3 = self.coupling4.jacobian()
#             # out19 = self.permute_c3(out18)

#             out20 = self.coupling5(out18)
#             jac1_c4 = self.coupling5.jacobian()
#             # out21 = self.permute_c4(out20)


#             # out22 = self.split(out21)
#             # out1s = out22[0] 
#             # out2s = out22[1] 


#             # flat_output1 = self.flat2(out2s)
            
#             # final_out=flat_output1
#             final_out=out20
#             jac =(jac1+jac1_c1+jac1_c2+jac1_c3+jac1_c4)
#             return final_out, jac


#     def forward(self, x):
#         xt = x[:, :self.d_h_in]  # [x]
#         xtdot = x[:, self.d_h_in:2 * self.d_h_in]  # [x_dot]
#         x_zero = x[:, 2 * self.d_h_in:]
#         cur_batch_size = x.shape[0]

#         # Define diffeomorphism model:
#         h_out,h_grad = self.Couplingblock(xt)
#         # h_grad=self.Couplingblock.jacobian()
#         h_grad = h_grad.unsqueeze(0).expand(cur_batch_size, self.n, self.d_h_in)
#         h_dot = bmm(h_grad, xtdot.unsqueeze(-1))
#         h_dot = h_dot.squeeze(-1)
        
#         h_zero,h_zerograd = self.Couplingblock(x_zero)
#         # h_zerograd=self.Couplingblock.jacobian()
#         # h_zerograd.unsqueeze(-1)
#         h_zerograd = h_zerograd.unsqueeze(0).expand(cur_batch_size, self.n, self.d_h_in)
#         h_zerograd = h_zerograd[:, :, :self.n]

#         y_pred = cat([h_out, h_dot, h_zerograd.norm(p=2,dim=2)], 1)  # [h, h_dot, norm(zerograd)]

#         return y_pred

#     def diffeomorphism_loss(self, y_true, y_pred, is_training):
#         h = y_pred[:,:self.n]
#         h_dot = y_pred[:,self.n:2*self.n]
#         zerograd = y_pred[:,2*self.n:]
#         cur_batch_size = y_pred.shape[0]
#         A_cl_batch = self.A_cl.unsqueeze(0).expand(cur_batch_size, self.n, self.n)

#         if is_training:
#             #return mean((y_true - (h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze()))**2) + self.jacobian_penalty*mean(zerograd**2)
#             return mean(
#                 (y_true + h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze()) ** 2) + self.jacobian_penalty * mean(
#                 zerograd ** 2)
#         else:
#             #return mean((y_true - (h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze())) ** 2)
#             return mean(
#                 (y_true + h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze()) ** 2)

#     def predict(self, x):
#         x.to(self.device)
#         # h = self.fc_in(x)
#         # for ii in range(self.n_hidden_layers):
#         #     h = F.relu(self.fc_hidden[ii](h))
#         # h = self.fc_out(h)
#         h_out,h_grad = self.Couplingblock(x)

#         return h_out.detach().numpy()      
        
        
        
        
        
        
        
        
        
        
        