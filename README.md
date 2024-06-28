# Attention-based Koopman EigenFlow Mode Decomposition (AKEFMD)
A Python library for simulating dynamics using Koopman operator theory and attention-based invertible neural network techniques.

The code in this repository has been developed to implement the methodologies described in 

M. Wang, X. Lou, B. Cui, J. Suykens, "An attention-based approach for Koopman modeling and predictive control of nonlinear systems", submited, 2024 

# Koopman Eigenfunction Extended Dynamic Mode Decomposition (KEEDMD)
The simulation framework of this method can be accessed via [KEEDMD](https://github.com/Cafolkes/keedmd).

## Running the code
Import a custom python library
```
from dynamics import LinearSystemDynamics
from controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, KFKoopmanEigenfunctions, Kfdmd,KfNNdmd,KACFKoopmanEigenfunctions,KHLACFKoopmanEigenfunctions,KRHLACFKoopmanEigenfunctions,KSdmd,KMdmd,K2Mdmd,KFIMPKoopmanEigenfunctions
```
Collect data and set parameters
```
from dynamics import LinearSystemDynamics
from controllers import PDController, OpenLoopController, MPCController, MPCControllerDense
from learning import KoopmanEigenfunctions, RBF, Edmd, Keedmd, KFKoopmanEigenfunctions, Kfdmd,KfNNdmd,KACFKoopmanEigenfunctions,KHLACFKoopmanEigenfunctions,KRHLACFKoopmanEigenfunctions,KSdmd,KMdmd,K2Mdmd,KFIMPKoopmanEigenfunctions
CD=CollectData()
xx,uu,xxd,tt=CD.random_rollout()
```
Methods:
- EDMD
- KEEDMD
- KEFMD
- KFMD
- AKEFMD (ours)
- AKEFMD* (ours)

EDMD
```
rbf_centers = random.rand(n,n_lift_edmd)
rbf_basis = RBF(rbf_centers, n, gamma=1.)
rbf_basis.construct_basis()
# Fit EDMD model
edmd_model = Edmd(rbf_basis, n, l1=l1_edmd, l1_ratio=l1_ratio_edmd)
X, X_d, Z, Z_dot, U, U_nom, t = edmd_model.process(xx, xxd, uu, uu_nom1, tt)
edmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
```

KEEDMD
```
A_cl = A_nom - dot(B_nom,K)
BK = dot(B_nom,K)
eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
keedmd_model = Keedmd(eigenfunction_basis, n, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd, l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd, l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d)
X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(xx, xxd, uu, uu_nom1, tt)
# keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
keedmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom, l1_ratio=l1_ratio_vals)
```

KEFMD
```
eigenfunction_basis = KFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
```

KFMD
```
eigenfunction_basis = KFIMPKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
```

AKEFMD
```
eigenfunction_basis = KACFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
eigenfunction_basis.fit_diffeomorphism_model(X=xx, t=tt, X_d=xxd, l2=l2_diffeomorphism, learning_rate=diff_learn_rate,
                                             learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac, batch_size=diff_batch_size)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
```

AKEFMD*
```
eigenfunction_basis = KHLACFKoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)  
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)
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
### Fit AKEFDMD model:
```

Prediction model establishment
```
LP1=LinearPrediction(A_nom,B_nom)
LP2=LinearPrediction(A=keedmd_model.A, B=keedmd_model.B)
LP3=LinearPrediction(A=edmd_model.A, B=edmd_model.B)
LP4=LinearPrediction(A=kfdmd_model.A, B=kfdmd_model.B)
LP5=LinearPrediction(A=kfimpdmd_model.A, B=kfimpdmd_model.B)
LP6=LinearPrediction(A=kacfdmd_model.A, B=kacfdmd_model.B)
LP7=LinearPrediction(A=krhlacfdmd_model.A, B=krhlacfdmd_model.B)
```
