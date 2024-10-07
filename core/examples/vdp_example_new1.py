
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:10:39 2024

@author: Ideal
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.model_selection import KFold
from numpy import arange, array, linalg,transpose,iscomplex,concatenate,diag,zeros, pi, random, interp, dot, multiply, asarray
import control
import torch.nn.functional as F
import warnings 
warnings.filterwarnings('ignore')
# Van der Pol 系统函数
A_nom =  array([[0., 1.], [-1., 2.]])
B_nom=array([[0],[1]])
Q = np.eye(2)
R = np.array([[1]])
K, _, _ = control.lqr(A_nom, B_nom, Q, R)
A_cl = A_nom - dot(B_nom,K)
KK=K
def van_der_pol(t, y, mu,u_func):
    unorm=np.dot(KK,y)
    return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]+unorm[0]+u_func(t)],dtype=float)

# 生成训练数据，包括 x 和 x_dot
def generate_data(num_points=100,num_samples=100, t_span=(0,10),mu=2.0,u_func = lambda t:0 ):#+np.random.rand(1)#0.1 * np.sin(0.1 * t)#(-1)**(int(t*100/2+1))
    t_eval = np.linspace(*t_span, num_samples)
    xx,uu,xx_dot=[],[],[]
    for i in range(num_points):
        # y0=np.random.randn(2)
        y0 = np.random.uniform(-2, 2, size=2)
        # u_func = lambda t: generate_random_control_input(t)
        sol = solve_ivp(van_der_pol, t_span, y0, t_eval=t_eval, args=(mu,u_func))
        x = sol.y.T
        u = np.array([u_func(t) for t in t_eval]) 
        x_dot = np.array([van_der_pol(t, x[i], mu,u_func) for i, t in enumerate(t_eval)])
        xx.append(x)
        uu.append(u)
        xx_dot.append(x_dot)
    x,u,x_dot=np.concatenate(xx),np.concatenate(uu),np.concatenate(xx_dot)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(x_dot, dtype=torch.float32),u

data_x, data_x_dot,u = generate_data()
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AttentionBlock_s(nn.Module):
    def __init__(self, input_output_dim, input_dim):
        super(AttentionBlock_s, self).__init__()
        self.query = nn.Linear(input_output_dim, input_output_dim)
        self.key = nn.Linear(input_output_dim, input_output_dim)
        self.output_linear = nn.Linear(input_output_dim, input_dim)  # 加一个输出线性层，将hidden_dim映射回input_dim
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.nnout = nn.Linear(input_output_dim, input_output_dim)

    def forward(self, x):
        nn_output=x
        Q = F.relu(self.query(nn_output))
        K = F.relu(self.key(nn_output))
        V = nn_output
        attention_scores = Q+K
        attention_weights = self.sigmoid(attention_scores)
        out = torch.mul(attention_weights, V)
        attention_weights2 = self.sigmoid(F.relu(self.nnout(out)))
        F2=nn_output+torch.mul(attention_weights2, out)
        return self.output_linear(F2)  # 映射回input_dim
    
class AttentionBlock_t(nn.Module):
    def __init__(self, input_output_dim, input_dim):
        super(AttentionBlock_t, self).__init__()
        self.query = nn.Linear(input_output_dim, input_output_dim)
        self.key = nn.Linear(input_output_dim, input_output_dim)
        self.output_linear = nn.Linear(input_output_dim, input_dim)  # 加一个输出线性层，将hidden_dim映射回input_dim
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.nnout = nn.Linear(input_output_dim, input_output_dim)

    def forward(self, x):
        nn_output=x
        Q = F.relu(self.query(nn_output))
        K = F.relu(self.key(nn_output))
        V = nn_output
        attention_scores = Q+K
        attention_weights = self.sigmoid(attention_scores)
        out = torch.mul(attention_weights, V)
        attention_weights2 = self.sigmoid(F.relu(self.nnout(out)))
        F2=nn_output+torch.mul(attention_weights2, out)
        return self.output_linear(F2)  # 映射回input_dim


class InvertibleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim,encoder_hidden_dim):
        super(InvertibleNN, self).__init__()
        input_dim=input_dim-1
        input_output_dim= input_dim+ encoder_hidden_dim
        self.encoder = Encoder(input_dim,hidden_dim, encoder_hidden_dim)
        self.nn = SimpleNN(input_output_dim,hidden_dim,input_output_dim)
        self.s = AttentionBlock_s(input_output_dim, input_dim)
        self.t = AttentionBlock_t(input_output_dim, input_dim)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        # y1 = x1 * torch.exp(self.s(x2)) + self.t(x2)
        # y1 = x1
        encoded = self.encoder(x1)
        nn_output =self.nn(torch.cat((x1, encoded), dim=-1))  
        y2 = x2 * torch.exp(self.s(nn_output)) + self.t(nn_output)
        return torch.cat((x1, y2), dim=1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        x2 = (y2 - self.t(y1)) / torch.exp(self.s(y1))
        x1 = (y1 - self.t(x2)) / torch.exp(self.s(x2))
        return torch.cat((x1, x2), dim=1)

# def jacobian(y, x): 
#     jac = torch.zeros(y.size(0), y.size(1), x.size(1))
#     for i in range(y.size(1)):
#         jac[:, i, :] = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True, create_graph=True)[0]
#     return jac
def jacobian(y, x):
    jac = []
    for i in range(y.size(1)):
        grad_output = torch.zeros_like(y)
        grad_output[:, i] = 1
        grad = torch.autograd.grad(y, x, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0]
        jac.append(grad.unsqueeze(1))
    return torch.cat(jac, dim=1)


def loss_fn(model, x, x_dot, A_bar):
    x.requires_grad_(True)
    z = model(x)
    J = jacobian(z, x)
    # print(J.size())
    I = torch.eye(x.size(1)).to(x.device)
    x0=torch.zeros((x.size(0),x.size(1)))
    x0.requires_grad_(True)
    z0=model(x0)
    J0 = jacobian(z0, x0)
    
    # term1 = torch.norm(torch.matmul(J, x_dot.T) - torch.matmul(A_bar, z.T))
    term1 = torch.norm(torch.matmul(J, x_dot.unsqueeze(2)).squeeze(2)-torch.matmul(A_bar, z.T).T)
    term2 = torch.norm(J0 - I)
    term3  =torch.norm(z0)
    # term2 = torch.norm(J[0] - I)#(term1 + term2)/x.size(0)
    # term1=F.mse_loss(torch.matmul(J, x_dot.T),torch.matmul(A_bar, z.T), reduction='mean')
    # term2=F.mse_loss(J[0], I, reduction='mean')#term1 + term2
    # maee=mean_absolute_error(torch.matmul(J, x_dot.unsqueeze(2)).squeeze(2), torch.matmul(A_bar, z.T).T)
    # r22=r2_score(torch.matmul(J, x_dot.unsqueeze(2)).squeeze(2), torch.matmul(A_bar, z.T).T)
    # mseee=F.mse_loss(torch.matmul(J, x_dot.unsqueeze(2)).squeeze(2), torch.matmul(A_bar, z.T).T)
    
    # return (term1 + term2+term3)/x.size(0), maee, r22,mseee
    
    return (term1 + term2+term3)/x.size(0)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
input_dim = data_x.size(1)
hidden_dim = 64
encoder_hidden_dim=8

BK = dot(B_nom,K)
A_bar=torch.tensor(A_cl, dtype=torch.float32) 
# A_bar = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)  # Van der Pol 系统的局部线性化矩阵

all_train_losses = []
all_val_losses = []

for train_idx, val_idx in kf.split(data_x):
    train_x, train_x_dot = data_x[train_idx], data_x_dot[train_idx]
    val_x, val_x_dot = data_x[val_idx], data_x_dot[val_idx]

    model = InvertibleNN(input_dim, hidden_dim,encoder_hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_x.requires_grad_(True)
        loss = loss_fn(model, train_x, train_x_dot, A_bar)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            with torch.set_grad_enabled(True):  
                val_x.requires_grad_(True)  
                val_loss = loss_fn(model, val_x, val_x_dot, A_bar)
        val_losses.append(val_loss.item())
        
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    # 保存模型
    torch.save(model.state_dict(), f'invertible_nn_fold_{len(all_train_losses)}.pth')

plt.figure(figsize=(12, 6))
for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
    plt.plot(train_losses, label=f'Train Loss Fold {i+1}')
    plt.plot(val_losses, label=f'Val Loss Fold {i+1}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.show()

# 加载一个已训练好的模型
model = InvertibleNN(input_dim, hidden_dim,encoder_hidden_dim)
model.load_state_dict(torch.load('invertible_nn_fold_1.pth'))
model.eval()

# 使用模型进行预测
with torch.no_grad():
    predictions = model(data_x)

# 可视化预测结果
plt.figure(figsize=(12, 6))
T=1000
# plt.plot(data_x.numpy()[:T, 0], label='True x1')
# plt.plot(predictions.numpy()[:T, 0], '--', label='Predicted x1')
# plt.plot(data_x.numpy()[:T, 1], label='True x2')
# plt.plot(predictions.numpy()[:T, 1], '--', label='Predicted x2')
plt.plot(data_x.numpy()[:T, 0],data_x.numpy()[:T, 1], label='True x2')
plt.plot(predictions.numpy()[:T, 0],predictions.numpy()[:T, 1], '--', label='Predicted x2')
plt.xlabel('Time Steps')
plt.ylabel('States')
plt.legend()
plt.title('True vs Predicted States')
plt.show()

# #==========================construct eigenfunctions and linear Koopman model
from numpy import array, linalg, transpose, diag, dot, ones, zeros, unique, power, prod, exp, log, divide, real, iscomplex, any, ones_like
from itertools import combinations_with_replacement, permutations
'''*******baiss*************'''
lambd, v = linalg.eig(A_cl)
_, w = linalg.eig(transpose(A_cl))

if any(iscomplex(lambd)) or any(iscomplex(w)):
    Warning("Complex eigenvalues and/or eigenvalues. Complex part supressed.")
    lambd = real(lambd)
    w = real(w)

# Scale up w to get kronecker delta
w_scaling = diag(dot(v.T, w))
w = divide(w, w_scaling.reshape(1,w.shape[0]))

max_power=2
nn=2
p = array([ii for ii in range(max_power+1)])
combinations = array(list(combinations_with_replacement(p,nn)))
powers = array([list(permutations(c,nn)) for c in combinations]) # Find all permutations of powers
powers = unique(powers.reshape((powers.shape[0] * powers.shape[1], powers.shape[2])),axis=0)  # Remove duplicates
# powerss = powers[:50,:]
powerss = powers
linfunc = lambda q: dot(transpose(w), q)  # Define principal eigenfunctions of the linearized system
eigfunc_lin = lambda q: prod(power(linfunc(q), transpose(powerss)), axis=0)  # Create desired number of eigenfunctions
Nlift = eigfunc_lin(ones((nn,1))).shape[0]
Lambda = log(prod(power(exp(lambd).reshape((nn,1)), transpose(powerss)), axis=0))  # Calculate corresponding eigenvalues

# scale_factor = 10**(0)*np.ones((nn,1))
# scale_func = lambda q: divide(q, scale_factor)
# # def scale_func(x):
# #     min_vals = np.min(x, axis=0)
# #     max_vals = np.max(x, axis=0)
# #     return 2 * (x - min_vals) / (max_vals - min_vals) - 1
# basis = lambda q: eigfunc_lin(scale_func(q))

'''**************baiss end****************'''
'''********************\phi**********'''
# data_x, data_x_dot,uu = generate_data()
# with torch.no_grad():
#     predictions = model(data_x)
# diffm=predictions.numpy()
# aa=np.array([basis(diffm[i,:].reshape(2,1)) for i in range(diffm.shape[0])])
# bb=np.array([linfunc(diffm[i,:].reshape(2,1)) for i in range(diffm.shape[0])])
'''******************\dot phi*************'''
# num_samples=1000 
# t_span=(0, 20)
# t_eval = np.linspace(*t_span, num_samples)
# num_traj=5
# dz=[]
# for i in range(num_traj):
#     dz_dt = np.diff(aa[num_samples*i:num_samples*(i+1),:], axis=0) / np.diff(t_eval)[:, None]
#     dz_dt = np.vstack([dz_dt, dz_dt[-1, :]])
#     dz.append(dz_dt) 
# dz_dt=np.vstack(dz)
# # dz_dt = np.vstack([dz_dt, dz_dt[-1, :]])
'''**********************model phi*********'''
# 生成训练数据，包括 x 和 x_dot
def van_der_pol(t, y, mu,u_func):
    # unorm=np.dot(KK,y)
    return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]+u_func(t)],dtype=float)
def add_noise(data, std, noise_level):
    return data + np.random.normal(0, std * noise_level, data.shape)
def generate_dataa(num_points=10,num_samples=1500, t_span=(0,15),mu=2.0,u_func = lambda t:(-1)**(int(t*100/150+1))):#+0.5* np.sin(10*np.pi * t)#+1.8*np.random.rand(1)-0.9#+0.1 * np.sin(0.1 * t)#+np.random.rand(1)#0.1 * np.sin(0.1 * t)#(-1)**(int(t*100/2+1))
    t_eval = np.linspace(*t_span, num_samples)
    xx,uu,xx_dot=[],[],[]
    for i in range(num_points):
        # y0=np.random.randn(2)
        y0 = np.random.uniform(-2, 2, size=2)
        # u_func = lambda t: generate_random_control_input(t)
        sol = solve_ivp(van_der_pol, t_span, y0, t_eval=t_eval, args=(mu,u_func))
        # x = sol.y.T
        if i<5:
            x = sol.y.T
        else:
            x = sol.y[0]
            y = sol.y[1]
            x_std = np.std(x)
            y_std = np.std(y)
            # x_noisy_1 = add_noise(x, x_std, 0.01)
            # y_noisy_1 = add_noise(y, y_std, 0.01)
            # data_noisy_1 = np.column_stack((x_noisy_1, y_noisy_1))
            x_noisy_5 = add_noise(x, x_std, 0.05)
            y_noisy_5 = add_noise(y, y_std, 0.05)
            data_noisy_5 = np.column_stack((x_noisy_5, y_noisy_5))
            x=data_noisy_5
            # x_noisy_10 = add_noise(x, x_std, 0.10)
            # y_noisy_10 = add_noise(y, y_std, 0.10)
            # data_noisy_10 = np.column_stack((x_noisy_10, y_noisy_10))
            # x=data_noisy_10
        u = np.array([u_func(t) for t in t_eval]) 
        x_dot = np.array([van_der_pol(t, x[i], mu,u_func) for i, t in enumerate(t_eval)])
        xx.append(x)
        uu.append(u)
        xx_dot.append(x_dot)
    x,u,x_dot=np.concatenate(xx),np.concatenate(uu),np.concatenate(xx_dot)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(x_dot, dtype=torch.float32),u
data_x, data_x_dot,u = generate_dataa()

import torch
import torch.nn as nn
import torch.optim as optim

# 定义 phi 函数的神经网络模型
class PhiModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=9, hidden_dim=64):
        super(PhiModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义学习参数 A, A2, B, B2
class SystemModel(nn.Module):
    def __init__(self):
        super(SystemModel, self).__init__()
        self.A = nn.Parameter(torch.randn(2, 2))
        self.A2 = nn.Parameter(torch.randn(2, 9))
        self.B = nn.Parameter(torch.randn(2, 1))
        self.B2 = nn.Parameter(torch.randn(9, 1))
    
    def forward(self, x, phi_x, u):
        # 计算 \dot{x}
        x_dot_pred = torch.matmul(x, self.A.T) + torch.matmul(phi_x, self.A2.T) + torch.matmul(u, self.B.T)
        
        return x_dot_pred

# 初始化模型
phi_model = PhiModel(input_dim=2, output_dim=9)
system_model = SystemModel()

# 已知矩阵 K 和 Lambda
K = torch.tensor(K, dtype=torch.float32)  # 1x2
Lambda2 = torch.diag(torch.tensor(Lambda, dtype=torch.float32))           # 9x9

# 优化器
optimizer = optim.Adam(list(phi_model.parameters()) + list(system_model.parameters()), lr=0.01)

##======
x = data_x
x_dot = data_x_dot
u = torch.tensor(u.reshape(data_x.size(0),1), dtype=torch.float32)
# dif =predictions
ww = torch.tensor(w, dtype=torch.float32)
pws = torch.tensor(powerss, dtype=torch.float32)
# 训练循环
num_epochs = 2000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 计算 phi(x)
    with torch.no_grad():
        dif = model(x)
    dif.requires_grad_(True)
    phi_x = phi_model(dif)
    # phi_xx=[torch.prod(torch.pow(torch.matmul(ww.T, phi_x[i,:].T.reshape(2,1)), pws.T), axis=0) for i in range(phi_x.size(0))]
    # phi_x = torch.stack(phi_xx)
    # ph=data_x[0,:].T
    # pp=torch.prod(torch.pow(torch.matmul(ww.T, ph.reshape(2,1)), pws.T), axis=0) 
    # pp.detach().numpy()
    # a=phi_x.detach().numpy()
    
    # 计算 \dot{x}
    x_dot_pred = system_model(x, phi_x, u)
    
    # 计算 \dot{phi}(x)   
    phi_dot_pred = -torch.matmul(x, K.T) @ system_model.B2.T + torch.matmul(phi_x, Lambda2.T) + torch.matmul(u, system_model.B2.T)
    
    # 计算损失函数
    loss_x = torch.mean((x_dot_pred - x_dot) ** 2)
    loss_phi = torch.mean((torch.autograd.grad(outputs=phi_x.sum(), inputs=phi_x, create_graph=True)[0] - phi_dot_pred) ** 2)
    loss = loss_x + loss_phi
    
    # 反向传播与优化
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 保存模型s
torch.save({
    'phi_model_state_dict': phi_model.state_dict(),
    'system_model_state_dict': system_model.state_dict(),
}, 'model.pth')

# 评估模型
# checkpoint = torch.load('model.pth')
# phi_model = PhiModel(input_dim=2, output_dim=9)
# system_model = SystemModel()

# phi_model.load_state_dict(checkpoint['phi_model_state_dict'])
# system_model.load_state_dict(checkpoint['system_model_state_dict'])
# with torch.no_grad():
#     phi_x_val = phi_model(dif)
#     x_dot_pred = system_model()

# 动态系统的状态更新函数
def dynamic_system_step(x,phi_model, system_model, u):
    with torch.no_grad():
        dif = model(x)
    with torch.no_grad():
        phi_x = phi_model(dif)
        # phi_xx=[torch.prod(torch.pow(torch.matmul(ww.T, phi_x[i,:].T.reshape(2,1)), pws.T), axis=0) for i in range(phi_x.size(0))]
        # phi_x = torch.stack(phi_xx)
        x_dot = torch.matmul(x, system_model.A.T) + torch.matmul(phi_x, system_model.A2.T) + torch.matmul(u, system_model.B.T)
        phi_dot = -torch.matmul(x, K.T) @ system_model.B2.T + torch.matmul(phi_x, Lambda2.T) + torch.matmul(u, system_model.B2.T)
    return x_dot, phi_dot,phi_x

# 生成随机初始点
# initial_states = data_x[0,:] # 10 个初始点，每个点是 2 维
# initial_states1 = dif[0,:] 
#### keep initial states consistent
path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre/'
cfl=1
ed=9
file=path+"kfold5%d"%(cfl)+"/"
data_x=np.load(file+'data_x%d'%(ed)+'.npy')
data_x=torch.tensor(data_x, dtype=torch.float32) 
u = u  
# 预测 1000 步的状态s
traj=20
num_steps = 1500
x_trajectory = np.zeros((traj, num_steps, 2))
phi_trajectory = np.zeros((traj, num_steps, 9))

# 对每个初始点进行预测
for i in range(traj):
    initial_states = data_x[i*num_steps,:]
    x = initial_states.unsqueeze(0)
    # phi_x = initial_states1.unsqueeze(0)
    
    for t in range(num_steps):
        x_dot, phi_dot,phi_x = dynamic_system_step(x,phi_model, system_model, u[t,:])
        x = x + x_dot * 0.01  # 使用简单的欧拉法，步长为 0.01
        # phi_x = phi_x + phi_dot * 0.02
        
        x_trajectory[i, t] = x.detach().numpy()
        # phi_trajectory[i, t] = phi_x.detach().numpy()

# 可视化 x 的预测轨迹
plt.figure(figsize=(14, 6))
for i in range(1):
    plt.plot(x_trajectory[i, :, 0], x_trajectory[i, :, 1], label=f'Trajectory {i+1}')
plt.title('Predicted Trajectories of x')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
T=1000
plt.plot(data_x.numpy()[:T, 0], label='True x1')
plt.plot(x_trajectory[i, :, 0], '--', label='Predicted x1')
plt.plot(data_x.numpy()[:T, 1], label='True x2')
plt.plot(x_trajectory[i, :, 1], '--', label='Predicted x2')
# plt.plot(u[:T], '--', label='Predicted x2')
plt.xlabel('Time Steps')
plt.ylabel('States')
plt.legend()
plt.title('True vs Predicted States')
plt.show()
plt.figure(figsize=(6, 3))
T=1000
plt.plot(data_x.numpy()[:T, 0], data_x.numpy()[:T, 1], label='True x')
plt.plot(x_trajectory[i, :, 0],x_trajectory[i, :, 1], '--', label='Predicted x')
# plt.plot(data_x.numpy()[:T, 1], label='True x2')
# plt.plot(x_trajectory[i, :, 1], '--', label='Predicted x2')
plt.xlabel('Time Steps')
plt.ylabel('States')
plt.legend()
plt.title('True vs Predicted States')
plt.show()










