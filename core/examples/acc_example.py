# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:01:47 2022

@author: Ideal
"""


#%%
"""ACC system Example"""
import matplotlib
from matplotlib.pyplot import ylim, xlabel, ylabel, fill_between
import os
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, title, savefig, tight_layout
from matplotlib.ticker import MaxNLocator
from numpy import arange, array, concatenate,diag
from numpy import zeros, pi, random, interp, dot, multiply, asarray
import numpy as np
from scipy.io import savemat
import sys  
sys.path.append('D:/cpython.p_y/DKP/keedmd-master/core') 
# sys.path.append('F:/keedmd-master/core') 
from systems import CartPole
from dynamics import LinearSystemDynamics
from controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, KFKoopmanEigenfunctions, Kfdmd,KfNNdmd,KACFKoopmanEigenfunctions,KHLACFKoopmanEigenfunctions,KRHLACFKoopmanEigenfunctions,KSdmd,KMdmd,K2Mdmd
import dill
import control
from datetime import datetime
import random as rand
import scipy.sparse as sparse

    
"""##############################"""
### acc's system
class CollectData():
    
    def __init__(self,Flag=0):
        ### parameters
        self.steps=0.01
        self.nx=2
        self.nu=1
        self.time=5
        self.nsim=int(self.time/self.steps)
        self.ntraj=200    
        self.flag=Flag
        
    def acc(self, P, steps, sets):   ### actual the acc paper's system
        a=-1.3
        b=-2
        c=1.5
        if self.flag==0:      ### original data
            x, y= P
            # miu, lamda = sets
            u =sets            
            dy=b*y+u        
            y=y + dy * steps
            dx=(a+c*(np.sin(y))**2)*x            
        elif self.flag==1:    ### noise data
            x, y= P
            # miu, lamda = sets
            u =sets
            dy=b*y+u +rand.gauss(0,0.02)      
            y=y + dy * steps
            dx=(a+c*(np.sin(y))**2)*x
            # dy=b*y+u+rand.gauss(0,0.02)
            # dx=(a+c*(np.sin(y))**2)*x+u+rand.gauss(0,0.02)
            # y=y + dy * steps
        elif self.flag==2:    ### linear data
            x, y= P
            # miu, lamda = sets
            u =sets
            dy=b*y+u
            dx=a*x
            y=y + dy * steps
            # dx = y 
        elif self.flag==3:    ### uncontrolled data
            x, y= P
            # miu, lamda = sets
            u =sets
            dy=b*y
            y=y + dy * steps
            dx=(a+c*(np.sin(y))**2)*x            
            # dx = y 
        return [x + dx * steps, y]
    ### train data   
    def random_rollout(self):
        ##parameters
        setss=4*np.random.rand(self.ntraj,self.nsim,self.nu)-2
        xd = zeros((self.ntraj,self.nsim+1,self.nx))
        t_eval = self.steps * arange(self.nsim + 1)
        ##parameters
        
        ##position iteration
        P0 = 4*np.random.rand(self.ntraj,self.nx)-2
        d = []
        tt=[]
        for i in range(self.ntraj):
            p0=P0[i]
            ### other system
            P1=[p0]
            for j in range(self.nsim):
                sets=setss[i,j,:]
                p1 = self.acc(p0, self.steps, sets)
                P1.append(p1)
                p0=p1
            ### pendulum system
            # P1=self.pend(p0,time,steps)
            ### pend endx
            d.append(P1)
            tt.append(t_eval)
        dnp = np.asarray(d,'float64')
        tt=array(tt)
        x=dnp
        u=setss
        return x, u,xd,tt
    ###add batch size
    def sample(self,batch_size):
        ##paraneters
        x, u= self.random_rollout()
        ############################################################## old samples
        ind=np.random.randint(0, self.nsim, size = batch_size)
        i=np.random.randint(0, self.ntraj, size=1)
        X,U,Y=[],[],[]
        for j in ind:
            x0=x[i,j,:]
            x1=x[i,j+1,:]
            u0=u[i,j,:]
            X.append(x0)
            U.append(u0)
            Y.append(x1)
        return np.array(X).reshape(batch_size, -1), np.array(U).reshape(batch_size, -1), np.array(Y).reshape(batch_size, -1) 
    
# ### test
# CD=CollectData()
# xx,uu,xxd,tt=CD.random_rollout()

class LinearPrediction():
    def __init__(self,A,B):
        ### parameters
        self.A=A
        self.B=B
        self.steps=0.01
        self.nx=self.A.shape[0]
        self.nu=self.B.shape[1]
        self.Ad,self.Bd=self.descretize()        
    def vdp(self, P, sets):   ### actual acc paper's system
        a=-1.3
        b=-2
        c=1.5
        x, y= P
        # miu, lamda = sets
        u =sets
        # dy =-0.5*y-x*(4*x**2-1) +0.5*u
        dy=b*y+u        
        y=y + dy * self.steps
        dx=(a+c*(np.sin(y))**2)*x
        # dx = y       
        return [x + dx * self.steps, y]
    def predict(self,x0,u,nsim):
        x1=[x0]
        for i in range(nsim):
            x1.append(dot(self.Ad,x0)+self.Bd*u[0,i])
            x0=x1[-1]
        return array(x1).squeeze()    
    def predict_true(self,x0,u,nsim):
        x1_true=[x0]
        for i in range(nsim):
            x1_true.append(self.vdp(x0,u[0,i]))
            x0=x1_true[-1]
        return array(x1_true).squeeze()
    def descretize(self):     
        Ad=np.eye(self.nx)+self.A*self.steps
        Bd=self.B*self.steps
        return Ad,Bd

    
# ### test
# A =  array([[0., 1.], [-1., 2.]])
# B=array([[0],[1]])
# LP=LinearPrediction(A,B)
# x0=array([[-0.1],[0.1]])
# nsim=200
# u=np.random.rand(1,nsim)
# x1_nom=LP.predict(x0,u,nsim)
"""##############################"""
#%% 
#! ===============================================   SET PARAMETERS    ===============================================
###########################collect data
CD=CollectData()
xx,uu,xxd,tt=CD.random_rollout()
# Define true system
 ### for vdp
n, m = 2, 1
# upper_bounds = array([2.0, 2])              
# lower_bounds = -upper_bounds 
upper_bounds = None              
lower_bounds = None
A_nom =  array([[-1.3, 0], [0, -2]])
B_nom=array([[0],[1]])
Q = np.eye(n)
R = np.array([[1]])
K, _, _ = control.lqr(A_nom, B_nom, Q, R)
K_p,K_d=array([[K[0,0]]]),array([[K[0,1]]])
uu_nom=dot(xx,K.T)
a=zeros((xx.shape[0],xx.shape[1],1))
a[:,:,0]=uu_nom
uu_nom1=a[:,:-1,:]
# uu_nom1=uu_nom[:,:-1,np.newaxis]
# uu_nom2=uu_nom1.reshape((xx.shape[0],xx.shape[1]-1,1))
# .reshape((uu_nom.shape[0],uu_nom.shape[1]-1,1))
# Koopman eigenfunction parameters
eigenfunction_max_power = 2                             # Max power of variables in eigenfunction products
l2_diffeomorphism = 0.0                                 # l2 regularization strength
jacobian_penalty_diffeomorphism = 1e1                   # Estimator jacobian regularization strength
diff_n_epochs = 50                                   # Number of epochs   500
diff_train_frac = 0.9                                   # Fraction of data to be used for training
diff_n_hidden_layers = 3                                # Number of hidden layers
diff_layer_width = 64                                   # Number of units in each layer
diff_batch_size = 64                                    # Batch size
diff_learn_rate = 0.001                                  # Learning rate
diff_learn_rate_decay = 0.99                            # Learning rate decay
diff_dropout_prob = 0.25                                # Dropout rate

# KEEDMD parameters
l1_pos_keedmd = 0.001979592839755224                    # l1 regularization strength for position states
l1_pos_ratio_keedmd = 0.1                               # l1-l2 ratio for position states
l1_vel_keedmd = 0.024029630466870816                    # l1 regularization strength for velocity states
l1_vel_ratio_keedmd = 1.0                               # l1-l2 ratio for velocity states
l1_eig_keedmd = 6.819171287059534                       # l1 regularization strength for eigenfunction states
l1_eig_ratio_keedmd = 0.1                               # l1-l2 ratio for eigenfunction states

# EDMD parameters (benchmark to compare against)
n_lift_edmd = (eigenfunction_max_power+1)**n-1          # Lifting dimension EDMD (same number as for KEEDMD)
l1_edmd = 0.00687693796                                 # l1 regularization strength
l1_ratio_edmd = 1.00                                    # l1-l2 ratio
l1_ratio_vals = array([0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])


"""##############################"""
#%% 
# #! ===============================================   SET PARAMETERS for two-steps Approximation    ===============================================
# ###########################collect uncontrolled data
# CD1=CollectData(Flag=3)
# xx,uu,xxd,tt=CD1.random_rollout()
# # Define true system
#  ### for vdp
# n, m = 2, 1
# # upper_bounds = array([2.0, 2])              
# # lower_bounds = -upper_bounds 
# upper_bounds = None              
# lower_bounds = None
# A_nom =  array([[0., 1.], [1., -0.5]])
# B_nom=array([[0],[0]])
# K=array([[0,0]])
# K_p,K_d=array([[K[0,0]]]),array([[K[0,1]]])
# uu_nom=dot(xx,K.T).reshape(xx.shape[0],xx.shape[1])
# a=zeros((xx.shape[0],xx.shape[1],1))
# a[:,:,0]=uu_nom
# uu_nom1=a[:,:-1,:]
# ###########################collect controlled data
# CD2=CollectData()
# xx2,uu2,xxd2,tt2=CD2.random_rollout()
# # Define true system
# uu_nom2=dot(xx2,K.T).reshape(xx.shape[0],xx.shape[1])
# a=zeros((xx2.shape[0],xx2.shape[1],1))
# a[:,:,0]=uu_nom2
# uu_nom12=a[:,:-1,:]
# # uu_nom1=uu_nom[:,:-1,np.newaxis]
# # uu_nom2=uu_nom1.reshape((xx.shape[0],xx.shape[1]-1,1))
# # .reshape((uu_nom.shape[0],uu_nom.shape[1]-1,1))
# # Koopman eigenfunction parameters
# eigenfunction_max_power = 2                             # Max power of variables in eigenfunction products
# l2_diffeomorphism = 0.0                                 # l2 regularization strength
# jacobian_penalty_diffeomorphism = 1e1                   # Estimator jacobian regularization strength
# diff_n_epochs = 20                                   # Number of epochs   500
# diff_train_frac = 0.9                                   # Fraction of data to be used for training
# diff_n_hidden_layers = 3                                # Number of hidden layers
# diff_layer_width = 64                                   # Number of units in each layer
# diff_batch_size = 64                                    # Batch size
# diff_learn_rate = 0.001                                  # Learning rate
# diff_learn_rate_decay = 0.99                            # Learning rate decay
# diff_dropout_prob = 0.25                                # Dropout rate

# # KEEDMD parameters
# l1_pos_keedmd = 0.001979592839755224                    # l1 regularization strength for position states
# l1_pos_ratio_keedmd = 0.1                               # l1-l2 ratio for position states
# l1_vel_keedmd = 0.024029630466870816                    # l1 regularization strength for velocity states
# l1_vel_ratio_keedmd = 1.0                               # l1-l2 ratio for velocity states
# l1_eig_keedmd = 6.819171287059534                       # l1 regularization strength for eigenfunction states
# l1_eig_ratio_keedmd = 0.1                               # l1-l2 ratio for eigenfunction states

# # EDMD parameters (benchmark to compare against)
# n_lift_edmd = (eigenfunction_max_power+1)**n-1          # Lifting dimension EDMD (same number as for KEEDMD)
# l1_edmd = 0.00687693796                                 # l1 regularization strength
# l1_ratio_edmd = 1.00                                    # l1-l2 ratio
# l1_ratio_vals = array([0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])

#%%
#!  ===============================================     FIT MODELS      ===============================================

# Construct basis of Koopman eigenfunctions for KEEDMD:
#################################### vdp start
"""KEEDMD"""
###############KEEDMD
A_cl = A_nom - dot(B_nom,K)
BK = dot(B_nom,K)
# lambd=np.linalg.eig(A_cl)
eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
## xs ts collect data  q_d initial state  desired state 
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
###################### 1. Fit KEEDMD model:
keedmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(xx, xxd, uu, uu_nom1, tt)
# keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
keedmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
#===============
####################### save NN data
#### save
path='D:/cpython.p_y/DKP/keedmd-master/acc_pre/'
# path='F:/keedmd-master/acc_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'acc_keedmd' + ".pt"
eigenfunction_basis.save_diffeomorphism_model(file_name)
### load
# eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
# eigenfunction_basis.load_diffeomorphism_model(file_name)
# wb=eigenfunction_basis.diffeomorphism_model.state_dict()
####
### save A,B
A=keedmd_model.A 
B=keedmd_model.B
np.save(path+"weights/"+'/A_keedmd.npy',A) 
np.save(path+"weights/"+'/B_keedmd.npy',B)
#===================
# ######################### 2. Fit KSDMD model:
# ksdmd_model = KSdmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
# X, X_d, Z, Z_dot, U, U_nom, t = ksdmd_model.process(xx, xxd, uu, uu_nom1, tt)
# ksdmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
# # ksdmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
# ######################### 3. Fit KMDMD model:
# kmdmd_model = KMdmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
# X, X_d, Z, Z_dot, U, U_nom, t = kmdmd_model.process(xx, xxd, uu, uu_nom1, tt)
# kmdmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
# # kmdmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, A_nom, B_nom, l1_ratio=l1_ratio_vals)
# ######################### 4. two-steps KMDMD model:
#     ### the first step using the uncontrolled dataset ###
# A_cl = A_nom
# BK = dot(B_nom,K)
# eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
# eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
# ## xs ts collect data  q_d initial state  desired state 
# eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
#                                              learning_decay=diff_learn_rate_decay, n_epochs=diff_n_e    pochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
# eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
# L_matrix=eigenfunction_basis.Lambda
#     ### the second step using the controlled dataset ###
# k2mdmd_model = K2Mdmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
# X_uc, X_d_uc, Z_uc, Z_dot_uc, U_uc, U_nom_uc, t_uc = k2mdmd_model.process(xx, xxd, uu, uu_nom1, tt)
# X, X_d, Z, Z_dot, U, U_nom, t = k2mdmd_model.process(xx2, xxd2, uu2, uu_nom12, tt2)
# k2mdmd_model.fit(Z_uc, Z_dot_uc,X, X_d, Z, Z_dot, U, U_nom)
# # kmdmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, A_nom, B_nom, l1_ratio=l1_ratio_vals)
###############
"""EDMD"""
##############EDMD
# Construct basis of RBFs for EDMD:
# upper_bounds = array([2.0, 2])              
# lower_bounds = -upper_bounds 
# rbf_centers = multiply(random.rand(n,n_lift_edmd),(upper_bounds-lower_bounds).reshape((upper_bounds.shape[0],1)))+lower_bounds.reshape((upper_bounds.shape[0],1))
rbf_centers = random.rand(n,n_lift_edmd)
rbf_basis = RBF(rbf_centers, n, gamma=1.)
rbf_basis.construct_basis()
# Fit EDMD model
edmd_model = Edmd(rbf_basis, n, l1=l1_edmd, l1_ratio=l1_ratio_edmd)
X, X_d, Z, Z_dot, U, U_nom, t = edmd_model.process(xx, xxd, uu, uu_nom1, tt)
edmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
### save A,B
A=edmd_model.A 
B=edmd_model.B
np.save(path+"weights/"+'/A_edmd.npy',A) 
np.save(path+"weights/"+'/B_edmd.npy',B)
################
"""KFDMD"""
###############KFDMD
A_cl = A_nom - dot(B_nom,K)
BK = dot(B_nom,K)
eigenfunction_basis = KFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
## xs ts collect data  q_d initial state  desired state 
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
### Fit KFDMD model:
kfdmd_model = Kfdmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
X, X_d, Z, Z_dot, U, U_nom, t = kfdmd_model.process(xx, xxd, uu, uu_nom1, tt)
#keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
kfdmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
#===============
####################### save NN data
#### save
# path='F:/keedmd-master/acc_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'acc_kfdmd' + ".pt"
eigenfunction_basis.save_diffeomorphism_model(file_name)
### load
# eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
# eigenfunction_basis.load_diffeomorphism_model(file_name)
# wb=eigenfunction_basis.diffeomorphism_model.state_dict()
####
### save A,B
A=kfdmd_model.A 
B=kfdmd_model.B
np.save(path+"weights/"+'/A_kfdmd.npy',A) 
np.save(path+"weights/"+'/B_kfdmd.npy',B)
#===================
###############
"""KENNDMD"""
# ###############KFNNDMD
# A_cl = A_nom - dot(B_nom,K)
# BK = dot(B_nom,K)
# eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
# eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
# ## xs ts collect data  q_d initial state  desired state 
# eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
#                                              learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
# eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
# ### Fit KFDMD model:
# keedmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
# X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(xx, xxd, uu, uu_nom1, tt)
# model = KfNNdmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
# model.train(Z, Z_dot, U, U_nom)
# model.save()
# model.load()
# A_kfnn, B_kfnn = model.get_system()
# # A_kfnn, B_kfnn=A, B
# ###############
"""KACFDMD"""
###############KACFDMD
A_cl = A_nom - dot(B_nom,K)
BK = dot(B_nom,K)
eigenfunction_basis = KACFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
## xs ts collect data  q_d initial state  desired state 
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
### Fit KEEDMD model:
kacfdmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
X, X_d, Z, Z_dot, U, U_nom, t = kacfdmd_model.process(xx, xxd, uu, uu_nom1, tt)
#keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
kacfdmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
#===============
####################### save NN data
#### save
# path='F:/keedmd-master/acc_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'acc_kacfdmd' + ".pt"
eigenfunction_basis.save_diffeomorphism_model(file_name)
# KACFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK).save_diffeomorphism_model(file_name)
### load
# eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
# eigenfunction_basis.load_diffeomorphism_model(file_name)
# wb=eigenfunction_basis.diffeomorphism_model.state_dict()
####
### save A,B
A=kacfdmd_model.A 
B=kacfdmd_model.B
np.save(path+"weights/"+'/A_kacfdmd.npy',A) 
np.save(path+"weights/"+'/B_kacfdmd.npy',B)
#===================
###############
"""KHLACFDMD"""
###############KHLACFDMD
# A_cl = A_nom - dot(B_nom,K)
# BK = dot(B_nom,K)
# eigenfunction_basis = KHLACFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
# eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
# ## xs ts collect data  q_d initial state  desired state 
# eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
#                                               learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
# eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
# ### Fit KEEDMD model:
# khlacfdmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
# X, X_d, Z, Z_dot, U, U_nom, t = khlacfdmd_model.process(xx, xxd, uu, uu_nom1, tt)
# #keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
# khlacfdmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
"""KRHLACFDMD"""
###############KRHLACFDMD
############### high quality data
A_cl = A_nom - dot(B_nom,K)
BK = dot(B_nom,K)
eigenfunction_basis = KHLACFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
## xs ts collect data  q_d initial state  desired state 
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                              learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
################ low quality data
CD2=CollectData(Flag=1)
x,u,xd,t=CD2.random_rollout()
xx2,uu2,xxd2,tt2=concatenate((xx,x)),concatenate((uu,u)),concatenate((xxd,xd)),concatenate((tt,t))
CD3=CollectData(Flag=2)
x,u,xd,t=CD2.random_rollout()
xx1,uu1,xxd1,tt1=concatenate((xx2,x)),concatenate((uu2,u)),concatenate((xxd2,xd)),concatenate((tt2,t))
uu_nom=dot(xx1,K.T)
a=zeros((xx1.shape[0],xx1.shape[1],1))
a[:,:,0]=uu_nom
uu_nom1=a[:,:-1,:]
#################
eigenfunction_basiss = KRHLACFKoopmanEigenfunctions(eigenfunction_basis,n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
eigenfunction_basiss.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
## xs ts collect data  q_d initial state  desired state 
eigenfunction_basiss.fit_diffeomorphism_model(X=xx1, t=tt1, X_d=xxd1, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                              learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basiss.construct_basis(ub=upper_bounds, lb=lower_bounds)
### Fit KEEDMD model:
krhlacfdmd_model = Keedmd(eigenfunction_basiss, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
X, X_d, Z, Z_dot, U, U_nom, t = krhlacfdmd_model.process(xx1, xxd1, uu1, uu_nom1, tt1)
#keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
krhlacfdmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
#===============
####################### save NN data
#### save
# path='F:/keedmd-master/acc_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'acc_krhlacfdmd' + ".pt"
eigenfunction_basis.save_diffeomorphism_model(file_name)
### load
# eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
# eigenfunction_basis.load_diffeomorphism_model(file_name)
# wb=eigenfunction_basis.diffeomorphism_model.state_dict()
####
### save A,B
A=krhlacfdmd_model.A 
B=krhlacfdmd_model.B
np.save(path+"weights/"+'/A_krhlacfdmd.npy',A) 
np.save(path+"weights/"+'/B_krhlacfdmd.npy',B)
#===================
###################################### vdp end  
def sample(l,Z):
    ##parameters
    tr=300
    lent=int(Z.shape[1]/tr)
    lent_new=lent-l
    zz=[]
    for j in range(l+1):
        z_new=zeros((Z.shape[0],int(Z.shape[1]-tr*(l-1))))
        for i in range(tr):
            z_new[:,int(i*lent_new):int((i+1)*lent_new)]=Z[:,int(i*lent)+j:int(i*lent+lent_new)+j]
        zz.append(z_new)                
    return zz  
ZZ=sample(2,Z)
#%%
#!  ===============================================      MODELS PREDICTION      ===============================================
"""PREDICTION"""
###################################### prediction start
# A =  array([[0., 1.], [-1., 2.]])
# B=array([[0],[1]])
LP1=LinearPrediction(A_nom,B_nom)
LP2=LinearPrediction(A=keedmd_model.A, B=keedmd_model.B)
LP3=LinearPrediction(A=edmd_model.A, B=edmd_model.B)
LP4=LinearPrediction(A=kfdmd_model.A, B=kfdmd_model.B)
# LP5=LinearPrediction(A=A_kfnn, B=B_kfnn)
LP6=LinearPrediction(A=kacfdmd_model.A, B=kacfdmd_model.B)
# LP7=LinearPrediction(A=khlacfdmd_model.A, B=khlacfdmd_model.B)
LP8=LinearPrediction(A=krhlacfdmd_model.A, B=krhlacfdmd_model.B)
# LP9=LinearPrediction(A=ksdmd_model.A, B=ksdmd_model.B)
# LP10=LinearPrediction(A=k2mdmd_model.A, B=k2mdmd_model.B)
# A=k2mdmd_model.A
# B=k2mdmd_model.B
# C=k2mdmd_model.C
# AA=ksdmd_model.A
# BB=ksdmd_model.B
A=krhlacfdmd_model.A
B=krhlacfdmd_model.B
import scipy.io as io
path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre'
io.savemat(path+'/A_krhlacfdmd.mat',{'A_matrix':A}) 
io.savemat(path+'/B_krhlacfdmd.mat',{'B_matrix':B}) 
##########################
x0=array([[0.2],[0.2]])
xd=zeros((2,1))
nsim=400
u=4*np.random.rand(1,nsim)-2
# u=np.ones((1,nsim))*0.2
###################  true  1 Nomial
x1_true=LP1.predict_true(x0,u,nsim)
x1_nom=LP1.predict(x0,u,nsim)
#######################
# z0=concatenate((x0,eigenfunction_basis.lift(x0,xd).T))
##################### 2 KEEDMD
z0=keedmd_model.lift(x0,xd).T 
x1_keedmd=LP2.predict(z0,u,nsim)
####################  3 EDMD
z0=edmd_model.lift(x0,xd).T    
x1_edmd=LP3.predict(z0,u,nsim)
####################  4 KFDMD
z0=kfdmd_model.lift(x0,xd).T 
x1_kfdmd=LP4.predict(z0,u,nsim)
####################  5 KfNNDMD
# z0=keedmd_model.lift(x0,xd).T 
# x1_kfnndmd=LP5.predict(z0,u,nsim)
# LP1.plotPre(x1_nom,x1_keedmd,x1_edmd)
####################  6 KACFDMD
z0=kacfdmd_model.lift(x0,xd).T 
x1_kacfdmd=LP6.predict(z0,u,nsim)
####################  7 KHLACFDMD
# z0=khlacfdmd_model.lift(x0,xd).T 
# x1_khlacfdmd=LP7.predict(z0,u,nsim)
####################  8 KRHLACFDMD
z0=krhlacfdmd_model.lift(x0,xd).T 
x1_krhlacfdmd=LP8.predict(z0,u,nsim)
####################  9 KSDMD
# z0=ksdmd_model.lift(x0,xd).T 
# x1_ksdmd=LP9.predict(z0,u,nsim)
####################  10 KMDMD
# z0=k2mdmd_model.lift(x0,xd).T 
# x1_k2mdmd=LP10.predict(z0,u,nsim)
####################  11 K2MDMD
# z0=k2mdmd_model.lift(x0,xd).T 
# z0=z0[n:,:]
# x1_k2mdmdd=LP10.predict(z0,u,nsim)
# x1_k2mdmd=dot(k2mdmd_model.C,x1_k2mdmdd.T).T
####################################### prediction end

for ii in range(2):
    subplot(2, 1, ii+1)
    plot(x1_true[:,ii],  linewidth=2, label='Nominal')  #, color='tab:gray'
    plot(x1_nom[:,ii], linewidth=2, label='Nominal')    #, color='tab:blue'
    plot(x1_edmd[:,ii], linewidth=2, label='KEEDMD')     #,color='tab:orange'
    plot(x1_keedmd[:,ii], linewidth=2, label='KEEDMD')    #, color='tab:green'       
    plot(x1_kfdmd[:,ii], linewidth=2, label='KFDMD')   #,color='tab:red'
    # plot(x1_kfnndmd[:,ii], linewidth=2, label='KFDMD',color='tab:red')
    plot(x1_kacfdmd[:,ii], linewidth=2, label='KACFDMD')  #,color='tab:red'
    # plot(x1_khlacfdmd[:,ii], linewidth=2, label='KACFDMD') 
    # plot(x1_ksdmd[:,ii], linewidth=2, label='KSDMD')
    # plot(x1_kmdmd[:,ii], linewidth=2, label='KMDMD')
    # plot(x1_k2mdmd[:,ii], linewidth=2, label='KMDMD')
legend(['True','Nominal','EDMD','KEEDMD','KEFMD','AKEFMD','AKEFMD*'],fontsize=12,loc=1)
# legend(['True','Nominal','KACFDMD'],fontsize=12,loc=1)
e1=np.mean((x1_nom[:,0]-x1_true[:,0])**2)+np.mean((x1_nom[:,1]-x1_true[:,1])**2)
e2=np.mean((x1_edmd[:,0]-x1_true[:,0])**2)+np.mean((x1_edmd[:,1]-x1_true[:,1])**2)
e3=np.mean((x1_keedmd[:,0]-x1_true[:,0])**2)+np.mean((x1_keedmd[:,1]-x1_true[:,1])**2)
e4=np.mean((x1_kfdmd[:,0]-x1_true[:,0])**2)+np.mean((x1_kfdmd[:,1]-x1_true[:,1])**2)
e5=np.mean((x1_kacfdmd[:,0]-x1_true[:,0])**2)+np.mean((x1_kacfdmd[:,1]-x1_true[:,1])**2)
# e6=np.mean((x1_khlacfdmd[:,0]-x1_true[:,0])**2)+np.mean((x1_khlacfdmd[:,1]-x1_true[:,1])**2)
e7=np.mean((x1_krhlacfdmd[:,0]-x1_true[:,0])**2)+np.mean((x1_krhlacfdmd[:,1]-x1_true[:,1])**2)

# Calculate error statistics
dt = 1.0e-2                                             
N = int(2./dt)                                          
t_eval = dt * arange(N+1 )
Ntraj_pred,xs_pred,t_pred=nsim,x1_true,t_eval.squeeze()
x2_nom,xs2_pred,Ntraj_pred=[x1_nom,x1_nom],[xs_pred,xs_pred],2
mse_nom = array([(x1_nom[ii,:] - xs_pred[ii,:])**2 for ii in range(Ntraj_pred)])
e_nom = array(np.abs([x1_nom[ii,:] - xs_pred[ii,:] for ii in range(Ntraj_pred)]))
mse_nom = np.mean(np.mean(np.mean(mse_nom)))
e_mean_nom = np.mean(e_nom, axis=0)
e_std_nom = np.std(e_nom,axis=0).reshape(2,1)



mse_nom = array([(x2_nom[ii] - xs2_pred[ii])**2 for ii in range(Ntraj_pred)])
e_nom = array(np.abs([x2_nom[ii] - xs2_pred[ii] for ii in range(Ntraj_pred)]))
mse_nom = np.mean(np.mean(np.mean(mse_nom)))
e_mean_nom = np.mean(e_nom, axis=0).T
e_std_nom = np.std(e_nom,axis=0).T
for ii in range(2):
    subplot(2, 1, ii+1)
    plot(t_pred,e_mean_nom[ii,:], linewidth=2, label='Nominal', color='tab:gray')
    fill_between(t_pred,e_mean_nom[ii,:]-e_std_nom[ii,:], e_mean_nom[ii,:]+e_std_nom[ii,:], alpha=0.2, color='tab:gray')

    # plot(e_mean_edmd[ii,:], linewidth=2, label='EDMD', color='tab:green')
    # fill_between( e_mean_edmd[ii,:] - e_std_edmd[ii, :], e_mean_edmd[ii,:] + e_std_edmd[ii, :], alpha=0.2, color='tab:green')

    # plot( e_mean_keedmd[ii,:], linewidth=2, label='KEEDMD',color='tab:orange')

    # fill_between( e_mean_keedmd[ii,:]- e_std_keedmd[ii, :], e_mean_keedmd[ii,:] + e_std_keedmd[ii, :], alpha=0.2,color='tab:orange')


#%%
#!  ===============================================      MODELS PREDICTION WITH MORE INITIAL CONDISTIONS     ===============================================
"""PREDICTION"""
##########################
x0=array([[0.8],[0.8]])
xd=zeros((2,1))
nsim=200
Ntraj_pred=50
x2_nom=[]
x2_keedmd=[]
x2_edmd=[]
x2_kfdmd=[]
x2_kacfdmd=[]
# x2_khlacfdmd=[]
x2_krhlacfdmd=[]
x2_ksdmd=[]
x2_kmdmd=[]
xs2_pred=[]
######## different inputs
for i in range(Ntraj_pred):    
    u=4*np.random.rand(1,nsim)-2
    # u=np.ones((1,nsim))*0.2
    ###################  true  1 Nomial
    x1_true=LP1.predict_true(x0,u,nsim)
    x1_nom=LP1.predict(x0,u,nsim)
    x2_nom.append(x1_nom)
    xs2_pred.append(x1_true)
    #######################
    # z0=concatenate((x0,eigenfunction_basis.lift(x0,xd).T))
    ##################### 2 KEEDMD
    z0=keedmd_model.lift(x0,xd).T 
    x1_keedmd=LP2.predict(z0,u,nsim)
    x2_keedmd.append(x1_keedmd)
    # ####################  3 EDMD
    z0=edmd_model.lift(x0,xd).T    
    x1_edmd=LP3.predict(z0,u,nsim)
    x2_edmd.append(x1_edmd)
    # ####################  4 KFDMD
    z0=kfdmd_model.lift(x0,xd).T 
    x1_kfdmd=LP4.predict(z0,u,nsim)
    x2_kfdmd.append(x1_kfdmd)
    # ####################  5 KfNNDMD
    # z0=keedmd_model.lift(x0,xd).T 
    # x1_kfnndmd=LP5.predict(z0,u,nsim)
    # x2_krhlacfdmd.append(x1_kfnndmd)
    # # # LP1.plotPre(x1_nom,x1_keedmd,x1_edmd)
    # # ####################  6 KACFDMD
    z0=kacfdmd_model.lift(x0,xd).T 
    x1_kacfdmd=LP6.predict(z0,u,nsim)
    x2_kacfdmd.append(x1_kacfdmd)
    # # ####################  7 KHLACFDMD
    # z0=khlacfdmd_model.lift(x0,xd).T 
    # x1_khlacfdmd=LP7.predict(z0,u,nsim)
    # x2_khlacfdmd.append(x1_khlacfdmd)
    # ####################  8 KRHLACFDMD
    z0=krhlacfdmd_model.lift(x0,xd).T 
    x1_krhlacfdmd=LP8.predict(z0,u,nsim)
    x2_krhlacfdmd.append(x1_krhlacfdmd)
    # ####################  9 KSDMD
    # z0=ksdmd_model.lift(x0,xd).T 
    # x1_ksdmd=LP9.predict(z0,u,nsim)
    # x2_ksdmd.append(x1_ksdmd)
    # ####################  10 KSDMD
    # z0=kmdmd_model.lift(x0,xd).T 
    # x1_kmdmd=LP10.predict(z0,u,nsim)
    # x2_kmdmd.append(x1_kmdmd)
######## different initial conditions
u=4*np.random.rand(1,nsim)-2  
x2_nom=[]
x2_keedmd=[]
x2_edmd=[]
x2_kfdmd=[]
x2_kacfdmd=[]
# x2_khlacfdmd=[]
x2_krhlacfdmd=[]
xs2_pred=[]  
for i in range(Ntraj_pred):    
    x0=4*np.random.rand(2,1)-1
    # u=np.ones((1,nsim))*0.2
    ###################  true  1 Nomial
    x1_true=LP1.predict_true(x0,u,nsim)
    x1_nom=LP1.predict(x0,u,nsim)
    x2_nom.append(x1_nom)
    xs2_pred.append(x1_true)
    #######################
    # z0=concatenate((x0,eigenfunction_basis.lift(x0,xd).T))
    ##################### 2 KEEDMD
    z0=keedmd_model.lift(x0,xd).T 
    x1_keedmd=LP2.predict(z0,u,nsim)
    x2_keedmd.append(x1_keedmd)
    # ####################  3 EDMD
    z0=edmd_model.lift(x0,xd).T    
    x1_edmd=LP3.predict(z0,u,nsim)
    x2_edmd.append(x1_edmd)
    # ####################  4 KFDMD
    z0=kfdmd_model.lift(x0,xd).T 
    x1_kfdmd=LP4.predict(z0,u,nsim)
    x2_kfdmd.append(x1_kfdmd)
    # ####################  5 KfNNDMD
    # # z0=keedmd_model.lift(x0,xd).T 
    # # x1_kfnndmd=LP5.predict(z0,u,nsim)
    # # LP1.plotPre(x1_nom,x1_keedmd,x1_edmd)
    # ####################  6 KACFDMD
    z0=kacfdmd_model.lift(x0,xd).T 
    x1_kacfdmd=LP6.predict(z0,u,nsim)
    x2_kacfdmd.append(x1_kacfdmd)
    # ####################  7 KHLACFDMD
    # z0=khlacfdmd_model.lift(x0,xd).T 
    # x1_khlacfdmd=LP7.predict(z0,u,nsim)
    # x2_khlacfdmd.append(x1_khlacfdmd)
    ####################  8 KRHLACFDMD
    z0=krhlacfdmd_model.lift(x0,xd).T 
    x1_krhlacfdmd=LP8.predict(z0,u,nsim)
    x2_krhlacfdmd.append(x1_krhlacfdmd)
####################################### prediction end
# Calculate error statistics                                          
def mean_std_cal(x2_nom,xs2_pred,Ntraj_pred):
    if x2_nom[0].shape[1]>xs2_pred[0].shape[1]:
        x2_nom=[x[:,:xs2_pred[0].shape[1]] for x in x2_nom]
    mse_nom = array([(x2_nom[ii] - xs2_pred[ii])**2 for ii in range(Ntraj_pred)])
    e_nom = array(np.abs([x2_nom[ii] - xs2_pred[ii] for ii in range(Ntraj_pred)]))
    mse_nom = np.mean(np.mean(np.mean(mse_nom)))
    e_mean_nom = np.mean(e_nom, axis=0).T
    e_std_nom = np.std(e_nom,axis=0).T
    return e_mean_nom,e_std_nom
e_mean_nom,e_std_nom=mean_std_cal(x2_nom,xs2_pred,Ntraj_pred)
e_mean_edmd,e_std_edmd=mean_std_cal(x2_edmd,xs2_pred,Ntraj_pred)
e_mean_keedmd,e_std_keedmd=mean_std_cal(x2_keedmd,xs2_pred,Ntraj_pred)
e_mean_kfdmd,e_std_kfdmd=mean_std_cal(x2_kfdmd,xs2_pred,Ntraj_pred)
e_mean_kacfdmd,e_std_kacfdmd=mean_std_cal(x2_kacfdmd,xs2_pred,Ntraj_pred)
# e_mean_khlacfdmd,e_std_khlacfdmd=mean_std_cal(x2_khlacfdmd,xs2_pred,Ntraj_pred)
e_mean_krhlacfdmd,e_std_krhlacfdmd=mean_std_cal(x2_krhlacfdmd,xs2_pred,Ntraj_pred)
# e_mean_edmd,e_std_edmd=mean_std_cal(x2_ksdmd,xs2_pred,Ntraj_pred)
###
e1=np.mean(np.mean(e_mean_nom))
e2=np.mean(np.mean(e_mean_edmd))
e3=np.mean(np.mean(e_mean_keedmd))
e4=np.mean(np.mean(e_mean_kfdmd))
e5=np.mean(np.mean(e_mean_kacfdmd))
# e6=np.mean(np.mean(e_mean_khlacfdmd))
e6=np.mean(np.mean(e_mean_krhlacfdmd))
###
###
e_mean_nom,e_std_nom=e_mean_nom1,e_std_nom1
e_mean_edmd,e_std_edmd=e_mean_edmd1,e_std_edmd1
e_mean_keedmd,e_std_keedmd=e_mean_keedmd1,e_std_keedmd1
e_mean_kfdmd,e_std_kfdmd=e_mean_kfdmd1,e_std_kfdmd1
e_mean_kacfdmd,e_std_kacfdmd=e_mean_kacfdmd1,e_std_kacfdmd1
e_mean_krhlacfdmd,e_std_krhlacfdmd=e_mean_krhlacfdmd1,e_std_krhlacfdmd1
e_mean_nom,e_std_nom=e_mean_nom2,e_std_nom2
e_mean_edmd,e_std_edmd=e_mean_edmd2,e_std_edmd2
e_mean_keedmd,e_std_keedmd=e_mean_keedmd2,e_std_keedmd2
e_mean_kfdmd,e_std_kfdmd=e_mean_kfdmd2,e_std_kfdmd2
e_mean_kacfdmd,e_std_kacfdmd=e_mean_kacfdmd2,e_std_kacfdmd2
e_mean_krhlacfdmd,e_std_krhlacfdmd=e_mean_krhlacfdmd2,e_std_krhlacfdmd2
###
dt=0.01
t_eval = dt * arange(nsim+1 )
t_pred=t_eval.squeeze()
import matplotlib.pyplot as plt
fig,axes = plt.subplots(figsize=(8,10))
# fig, ax = plt.subplots()
for ii in range(2):
    plt.subplot(2, 1, ii+1)
    plt.plot(t_pred,e_mean_nom[ii,:], linewidth=2, label='Nominal', color='tab:gray')
    fill_between(t_pred,e_mean_nom[ii,:]-e_std_nom[ii,:], e_mean_nom[ii,:]+e_std_nom[ii,:], alpha=0.2, color='tab:gray')

    plt.plot(t_pred,e_mean_edmd[ii,:], linewidth=2, label='Nominal', color='tab:green')
    fill_between(t_pred,e_mean_edmd[ii,:]-e_std_edmd[ii,:], e_mean_edmd[ii,:]+e_std_nom[ii,:], alpha=0.2, color='tab:green')

    plt.plot( t_pred,e_mean_keedmd[ii,:], linewidth=2, label='KEEDMD',color='tab:orange')
    fill_between( t_pred,e_mean_keedmd[ii,:]- e_std_keedmd[ii, :], e_mean_keedmd[ii,:] + e_std_keedmd[ii, :], alpha=0.2,color='tab:orange')

    plt.plot( t_pred,e_mean_kfdmd[ii,:], linewidth=2, label='KEEDMD',color='tab:red')
    fill_between(t_pred, e_mean_kfdmd[ii,:]- e_std_kfdmd[ii, :], e_mean_kfdmd[ii,:] + e_std_kfdmd[ii, :], alpha=0.2,color='tab:red')

    plot(t_pred, e_mean_kacfdmd[ii,:], linewidth=2, label='KEEDMD',color='tab:blue')
    fill_between(t_pred, e_mean_kacfdmd[ii,:]- e_std_kacfdmd[ii, :], e_mean_kacfdmd[ii,:] + e_std_kacfdmd[ii, :], alpha=0.2,color='tab:blue')

    # plot(t_pred, e_mean_khlacfdmd[ii,:], linewidth=2, label='KEEDMD',color='tab:purple')
    # fill_between(t_pred, e_mean_khlacfdmd[ii,:]- e_std_khlacfdmd[ii, :], e_mean_khlacfdmd[ii,:] + e_std_khlacfdmd[ii, :], alpha=0.2,color='tab:purple')

    plt.plot(t_pred, e_mean_krhlacfdmd[ii,:], linewidth=2, label='KEEDMD',color='#DA70D6')
    fill_between(t_pred, e_mean_krhlacfdmd[ii,:]- e_std_krhlacfdmd[ii, :], e_mean_krhlacfdmd[ii,:] + e_std_krhlacfdmd[ii, :], alpha=0.2,color='#DA70D6')
    
    plt.xlabel("$T$",fontsize=12)
    plt.ylabel("$e_%d$"%(ii+1),fontsize=12)
    plt.grid()
    plt.xlim(0, 2)
    if ii==0:
        plt.title("Mean prediction error (+/-1 std)",fontsize=12)
        legend(['Nominal','EDMD','KEEDMD','KEFMD','AKEFMD','AKEFMD*'],fontsize=12)
    # if ii==0:
    #     plt.xlabel("T")
    #     plt.ylabel("$x_1$")
    # if ii==1:
    #     plt.xlabel("T")
    #     plt.ylabel("$x_2$")
path='F:/keedmd-master/acc_pre/'
if not os.path.exists(path+"Figure/"):
            os.mkdir(path+"Figure/")
plt.savefig(path+'/Figure/Initial.pdf',dpi=600,bbox_inches = 'tight')   
# plt.title("Mean prediction error (+/-1 std)",fontsize=12)
# legend(['Nominal','EDMD','KEEDMD','KEFMD','AKEFMD','AKEFMD*'],fontsize=12,loc=1)
#### box plot
import pandas as pd
import numpy as np
import seaborn as sns
# from scipy.io import loadmat
def er(x):
    er=(x[0,:]+x[1,:])/2
    return er
er1=er(e_mean_nom)
er2=er(e_mean_edmd)
er3=er(e_mean_keedmd)
er4=er(e_mean_kfdmd)
er5=er(e_mean_kacfdmd)
er6=er(e_mean_krhlacfdmd)
data={'Nominal':list(er1),'EDMD':list(er2),'KEEDMD':list(er3),'KEFDMD':list(er4),'AKEFMD':list(er5),'AKEFMD*':list(er6)}
data=pd.DataFrame(data)
fig,axes = plt.subplots()
# sns.boxplot(x='Method',y='MSE and MAD',hue='Error',palette=sns.color_palette('Dark2'), 
#             data=data,orient='v',ax=axes,width = 0.4 , showfliers=False)  #orient='h',
# axes.text(0.45,-0.07,'(e) Method vs Error',fontsize=13)
sns.boxplot(data=data,orient='v',width = 0.3 )  ##palette=sns.color_palette('Accent'),
axes.text(2.4,-4.,'(b)',fontsize=13)
# axes.text(2.4,-2.7,'(a)',fontsize=13)
plt.ylabel("Error",fontsize=12)
plt.savefig(path+'/Figure/Initial_box.pdf',dpi=600,bbox_inches = 'tight')
############################################## save data for paper
#### inputs
# x2_nom=np.array(x2_nom)
# x2_edmd=np.array(x2_edmd)
# x2_keedmd=np.array(x2_keedmd)
# x2_kfdmd=np.array(x2_kfdmd)
# x2_kacfdmd=np.array(x2_kacfdmd)
# x2_krhlacfdmd=np.array(x2_krhlacfdmd)
# xs2_pred=np.array(xs2_pred)
path='F:/keedmd-master/acc_pre/'
np.save(path+'/x2_nom1.npy',np.array(x2_nom))
np.save(path+'/x2_edmd1.npy',np.array(x2_edmd))  
np.save(path+'/x2_keedmd1.npy',np.array(x2_keedmd))
np.save(path+'/x2_kfdmd1.npy',np.array(x2_kfdmd)) 
np.save(path+'/x2_kacfdmd1.npy',np.array(x2_kacfdmd))
np.save(path+'/x2_krhlacfdmd1.npy',np.array(x2_krhlacfdmd)) 
np.save(path+'/xs2_pred1.npy',np.array(xs2_pred))
#### initial
np.save(path+'/x2_nom2.npy',np.array(x2_nom))
np.save(path+'/x2_edmd2.npy',np.array(x2_edmd))  
np.save(path+'/x2_keedmd2.npy',np.array(x2_keedmd))
np.save(path+'/x2_kfdmd2.npy',np.array(x2_kfdmd)) 
np.save(path+'/x2_kacfdmd2.npy',np.array(x2_kacfdmd))
np.save(path+'/x2_krhlacfdmd2.npy',np.array(x2_krhlacfdmd)) 
np.save(path+'/xs2_pred2.npy',np.array(xs2_pred))
###
### save data
#### inputs
path='F:/keedmd-master/acc_pre/'
np.save(path+'/e_mean_nom1.npy',e_mean_nom) 
np.save(path+'/e_std_nom1.npy',e_std_nom)
np.save(path+'/e_mean_edmd1.npy',e_mean_edmd) 
np.save(path+'/e_std_edmd1.npy',e_std_edmd)
np.save(path+'/e_mean_keedmd1.npy',e_mean_keedmd) 
np.save(path+'/e_std_keedmd1.npy',e_std_keedmd)
np.save(path+'/e_mean_kfdmd1.npy',e_mean_kfdmd) 
np.save(path+'/e_std_kfdmd1.npy',e_std_kfdmd)
np.save(path+'/e_mean_kacfdmd1.npy',e_mean_kacfdmd) 
np.save(path+'/e_std_kacfdmd1.npy',e_std_kacfdmd)
np.save(path+'/e_mean_krhlacfdmd1.npy',e_mean_krhlacfdmd) 
np.save(path+'/e_std_krhlacfdmd1.npy',e_std_krhlacfdmd)
#### initial conditions
np.save(path+'/e_mean_nom2.npy',e_mean_nom) 
np.save(path+'/e_std_nom2.npy',e_std_nom)
np.save(path+'/e_mean_edmd2.npy',e_mean_edmd) 
np.save(path+'/e_std_edmd2.npy',e_std_edmd)
np.save(path+'/e_mean_keedmd2.npy',e_mean_keedmd) 
np.save(path+'/e_std_keedmd2.npy',e_std_keedmd)
np.save(path+'/e_mean_kfdmd2.npy',e_mean_kfdmd) 
np.save(path+'/e_std_kfdmd2.npy',e_std_kfdmd)
np.save(path+'/e_mean_kacfdmd2.npy',e_mean_kacfdmd) 
np.save(path+'/e_std_kacfdmd2.npy',e_std_kacfdmd)
np.save(path+'/e_mean_krhlacfdmd2.npy',e_mean_krhlacfdmd) 
np.save(path+'/e_std_krhlacfdmd2.npy',e_std_krhlacfdmd)
###
#%%
#!  ===============================================      KMPC CONTROL      ===============================================


import numpy as np
import scipy.sparse as sparse
import time
import sys  
# sys.path.append('D:/cpython.p_y/DKP/keedmd-master/pyMPC-master') 
sys.path.append('F:/keedmd-master/pyMPC-master') 
import matplotlib.pyplot as plt
from pyMPC.mpc import MPCController
from scipy.integrate import ode

# Constants #
Ts = 0.01 # sampling time (s)
# Continuous-time matrices (just for reference)
Ac=kacfdmd_model.A
Bc=kacfdmd_model.B
# Ac=krhlacfdmd_model.A
# Bc=krhlacfdmd_model.B
def f_ODE(t,x,u):
    # der = Ac @ x + Bc @ u
    der =np.zeros(len(x))
    der[0]=x[1]
    # der[1]=2*(1-x[0]**2)*x[1]-x[0] +u  
    der[1]=-0.5*x[1]-x[0]*(4*x[0]**2-1) +0.5*u
    return der

[nx, nu] = Bc.shape  # number of states and number or inputs

# Simple forward euler discretization
Ad = np.eye(nx) + Ac*Ts
Bd = Bc*Ts


# Reference input and states
xref = np.zeros(nx) # reference state
uref = np.zeros(nu)   # reference input
uc = np.ones(nu)
uminus1 =np.zeros(nu)     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

# Constraints
xmin = -2*np.ones(nx)
xmax = -xmin

umin = -2*np.ones(nu)
umax = -umin

Dumin = -2e-1*np.ones(nu)
Dumax = -Dumin

#####====Objective function start
# Objective function
# Qx = 1*sparse.eye(nx)   # Quadratic cost for states x0, x1, ..., x_N-1
# QxN =1*sparse.eye(nx)  # Quadratic cost for xN  sparse.diags([1, 1])
# Qu = 0.01 * sparse.eye(nu)        # Quadratic cost for u0, u1, ...., u_N-1
# QDu = 0.01 * sparse.eye(nu)       # Quadratic cost for Du0, Du1, ...., Du_N-1
####=====
# row=np.arange(nx)
# col=np.arange(nx)
# data=np.ones(nx)
# data[0]=0.1 
# coo = sparse.coo_matrix((data, (row, col)), shape=(nx, nx))
# Qx = coo
# QxN=coo
# Qu = 0.01 * sparse.eye(nu)        # Quadratic cost for u0, u1, ...., u_N-1
# QDu = 0.01 * sparse.eye(nu)
#####====
data=np.ones(nx)
data[0]=260
data[1]=8
offsets=np.array([0])
Qx=sparse.dia_matrix((data, offsets), shape=(nx, nx))
QxN=Qx
Qu = 0.01 * sparse.eye(nu)        # Quadratic cost for u0, u1, ...., u_N-1
QDu =0.01 * sparse.eye(nu)
####====Objective function end
# Initial state
# x0 = np.array([0.1, 0.2]) # initial state
x0=array([[0.5],[0.5]])
xd=zeros((2,1))
z0=kacfdmd_model.lift(x0,xd).T
# z0=krhlacfdmd_model.lift(x0,xd).T
z0=z0.squeeze()
x0=x0.squeeze()

system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
system_dyn.set_initial_value(x0, 0)
system_dyn.set_f_params(0.0)

# Prediction horizon
Np =10

K = MPCController(Ad,Bd,Np=Np, x0=z0,xref=xref,uminus1=uminus1,
                  Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                  xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax)
K.setup()

# Simulate in closed loop
[nx, nu] = Bd.shape # number of states and number or inputs
len_sim = 12 # simulation length (s)
nsim = int(len_sim/Ts) # simulation length(timesteps)
xsim = np.zeros((nsim,n))
usim = np.zeros((nsim,nu))
tcalc = np.zeros((nsim,1))
tsim = np.arange(0,nsim)*Ts

xstep = x0
uMPC = uminus1

# time_start = time.time()
for i in range(nsim):
    xsim[i,:] = xstep
    
    x0=xstep.reshape(len(xstep),1)
    # z0=krhlacfdmd_model.lift(x0,xd).T
    z0=kacfdmd_model.lift(x0,xd).T
    z0=z0.squeeze()

    # MPC update and step. Could be in just one function call
    # time_start = time.time()
    K.update(z0, uMPC) # update with measurement
    uMPC = K.output() # MPC step (u_k value)
    # tcalc[i,:] = time.time() - time_start
    usim[i,:] = uMPC

    #xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # Real system step (x_k+1 value)
    system_dyn.set_f_params(uMPC) # set current input value to uMPC
    system_dyn.integrate(system_dyn.t + Ts)
    xstep = system_dyn.y


# time_sim = time.time() - time_start
fig,axes = plt.subplots(3,1, figsize=(8,10))
axes[0].plot(tsim, xsim[:nsim,0], "k", label='State')
axes[0].plot(tsim, xref[0]*np.ones(np.shape(tsim)), "r--", label="Reference")
axes[0].set_xlabel("$T$",fontsize=12)
axes[0].set_ylabel("$x_1$",fontsize=12)
# axes[0].legend('$x_1$',"$x_r$",fontsize=12)
# axes[0].set_title("Position (m)")

axes[1].plot(tsim, xsim[:nsim,1], "k",label="State")
axes[1].plot(tsim, xref[1]*np.ones(np.shape(tsim)), "r--", label="Reference")
axes[1].set_xlabel("$T$",fontsize=12)
axes[1].set_ylabel("$x_2$",fontsize=12)
# axes[1].set_title("Velocity (m/s)")

axes[2].plot(tsim, usim[:nsim,0],"k", label="Input")
axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="Reference")
axes[2].plot(tsim, -uc*np.ones(np.shape(tsim))*2, ":",color='blue')
axes[2].plot(tsim, uc*np.ones(np.shape(tsim))*2, ":", label="Constraints",color='blue')
axes[2].set_xlabel("$T$",fontsize=12)
axes[2].set_ylabel("$u$",fontsize=12)
# axes[2].set_title("Force (N)")

for ax in axes:
    ax.grid(True)
    ax.legend()
plt.savefig(path+'/Figure/control_kac.pdf',dpi=600,bbox_inches = 'tight')
# plt.figure()
# plt.hist(tcalc*1000)
# plt.grid(True)


















