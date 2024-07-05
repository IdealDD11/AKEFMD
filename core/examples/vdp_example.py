# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:17:24 2022

@author: Ideal
"""

#%%
"""Van der Pol Example"""
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
from systems import CartPole
from dynamics import LinearSystemDynamics
from controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, KFKoopmanEigenfunctions, Kfdmd,KfNNdmd,KACFKoopmanEigenfunctions,KHLACFKoopmanEigenfunctions,KRHLACFKoopmanEigenfunctions,KSdmd,KMdmd,K2Mdmd,KFIMPKoopmanEigenfunctions
import dill
import control
from datetime import datetime
import random as rand
import scipy.sparse as sparse
import CollectData
import LinearPrediction


    
"""##############################"""
### vdp system    
# ### test
# CD=CollectData()
# xx,uu,xxd,tt=CD.random_rollout()
    
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
A_nom =  array([[0., 1.], [-1., 2.]])
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
diff_n_epochs = 20                                   # Number of epochs   500
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
# #%% 
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
# A_nom =  array([[0., 1.], [-1., 2.]])
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
path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'vdp_keedmd' + ".pt"
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
#                                              learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
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
path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'vdp_kfdmd' + ".pt"
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
################
"""KFIMPDMD"""
###############KFDMD
A_cl = A_nom - dot(B_nom,K)
BK = dot(B_nom,K)
eigenfunction_basis = KFIMPKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  ##given
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
## xs ts collect data  q_d initial state  desired state 
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
### Fit KFDMD model:
keedmd_model = Kfdmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(xx, xxd, uu, uu_nom1, tt)
#keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
keedmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
#===============
####################### save NN data
#### save
path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'vdp_kfdmd' + ".pt"
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
path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'vdp_kacfdmd' + ".pt"
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
path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre/'
if not os.path.exists(path+"weights/"):
            os.mkdir(path+"weights/")
file_name = path+"weights/" + 'vdp_krhlacfdmd' + ".pt"
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
# LP2=LinearPrediction(A=A_keedmd, B=B_keedmd)
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
# A=krhlacfdmd_model.A
# B=krhlacfdmd_model.B
# import scipy.io as io
# path='D:/cpython.p_y/DKP/keedmd-master/vdp_pre'
# io.savemat(path+'/A_krhlacfdmd.mat',{'A_matrix':A}) 
# io.savemat(path+'/B_krhlacfdmd.mat',{'B_matrix':B}) 
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
    plot(x1_true[:,ii],  linewidth=2, label='True')  #, color='tab:gray'
    plot(x1_nom[:,ii], linewidth=2, label='Nominal')    #, color='tab:blue'
    plot(x1_edmd[:,ii], linewidth=2, label='EDMD')     #,color='tab:orange'
    plot(x1_keedmd[:,ii], linewidth=2, label='KEEDMD')    #, color='tab:green'       
    plot(x1_kfdmd[:,ii], linewidth=2, label='KFDMD')   #,color='tab:red'
    # plot(x1_kfnndmd[:,ii], linewidth=2, label='KFDMD',color='tab:red')
    plot(x1_kacfdmd[:,ii], linewidth=2, label='KACFDMD')  #,color='tab:red'
    # plot(x1_khlacfdmd[:,ii], linewidth=2, label='KACFDMD') 
    # plot(x1_ksdmd[:,ii], linewidth=2, label='KSDMD')
    # plot(x1_kmdmd[:,ii], linewidth=2, label='KMDMD')
    # plot(x1_k2mdmd[:,ii], linewidth=2, label='KMDMD')
legend(['True','Nominal','EDMD','KEEDMD','KEFMD','AKEFMD','AKEFMD*'],fontsize=12,loc=1)






