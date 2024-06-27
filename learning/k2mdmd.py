# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:14:48 2022

@author: Ideal
"""


from .edmd import Edmd
from sklearn import linear_model
from numpy import kron,array, concatenate, zeros, dot, linalg, eye, diag, std, divide, tile, multiply, atleast_2d, ones, zeros_like


class K2Mdmd(Edmd):
    def __init__(self, basis, system_dim, l1_pos=0., l1_ratio_pos=0.5, l1_vel=0., l1_ratio_vel=0.5, l1_eig=0., l1_ratio_eig=0.5, acceleration_bounds=None, override_C=True, K_p = None, K_d = None, episodic=False):
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

    def fit(self, z_uc,z_dot_uc,X, X_d, Z, Z_dot, U, U_nom):
        self.n_lift = Z.shape[0]

        if self.l1 == 0.:
            # Solve least squares problem to find A and B for velocity terms:
            
            if self.episodic:
                input_vel = concatenate((Z, U-U_nom),axis=0).T
            else:
                input_vel = concatenate((Z, U), axis=0).T
            output_vel = Z_dot[:self.n,:].T
            sol_vel = atleast_2d(dot(linalg.pinv(input_vel),output_vel).transpose())
            A_vel = sol_vel[:,:self.n_lift]
            B_vel = sol_vel[:,self.n_lift:]

            # Construct A matrix
            self.A = zeros((self.n_lift, self.n_lift))
            # self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
            self.A[:self.n,:] = A_vel
            self.A[self.n:,self.n:] = diag(self.basis.Lambda)

            # Solve least squares problem to find B for position terms:
            ### new_one
            U_state_fb = dot(concatenate((self.K_p, self.K_d), axis=1),X)
            U_nom =  U_state_fb
            
            l=2
            phi=Z[self.n:, :]
            phi_,phi0 = self.sample(l,phi,0)
            UU=self.sample(l,U,1)
            U_nomm=self.sample(l,U_nom,1)
            A_nom=diag(self.basis.Lambda)
            AA=A_nom
            AAA=[eye(A_nom.shape[0]),A_nom]
            # for i in range(2,l+1):
            #     AA=concatenate(( AA,dot(AAA[-1],A_nom)), axis=0)
            #     AAA.append(dot(AAA[-1],A_nom))
            for i in range(2,l+1):
                AA=concatenate(( AA,A_nom**i), axis=0)
                AAA.append(A_nom**i)
            BB= kron((UU[0] - U_nomm[0]).T,eye(A_nom.shape[0]))
            for i in range(1,l):
                a=kron((UU[0] - U_nomm[0]).T,AAA[i])
                for j in range(1,i+1):
                    a=a+kron((UU[j] - U_nomm[j]).T,AAA[i-j])
                BB=concatenate(( BB,a), axis=0)
            input_eig = BB
            output_eig =( (phi_ - dot(AA, phi0)).T).flatten()
            output_eig=output_eig.reshape(len(output_eig),1)
            # B_eig = atleast_2d(dot(linalg.pinv(BB),output_eig).reshape(A_nom.shape[0],-1))
            B_eig = atleast_2d(dot(linalg.pinv(input_eig), output_eig))
            # B_eig = atleast_2d(dot(linalg.pinv(A_matrix),B_eig_cof).transpose())
    
            # Construct B matrix:
            self.B = concatenate(( B_vel, B_eig), axis=0)

            # Solve least squares problem to find B for eigenfunction terms:
            # U_state_fb = dot(concatenate((self.K_p, self.K_d), axis=1), X)
            # input_eig = (U - U_state_fb).T
            # output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
            # B_eig = atleast_2d(dot(linalg.pinv(input_eig), output_eig).transpose())

            # Construct B matrix:
            # self.B = concatenate(( B_vel, B_eig), axis=0)

            if self.override_C:
                self.C = zeros((self.n,self.n_lift))
                self.C[:self.n,:self.n] = eye(self.n)
                self.C = multiply(self.C, self.Z_std.transpose())
                raise Exception('Warning: Learning of C not implemented for structured regression.')

        # else:
        #     # reg_model = linear_model.Lasso(alpha=self.l1,  fit_intercept=False,
        #     #                                              normalize=False, selection='random', max_iter=1e5)

        #     reg_model = linear_model.ElasticNet(alpha=self.l1, l1_ratio=self.l1_ratio, fit_intercept=False,
        #                                                   normalize=False, selection='random', max_iter=1e5)

        #     # Solve least squares problem to find A and B for velocity terms:
            
        #     input_vel = z_uc[self.n:,:].T
        #     output_vel = z_uc[:self.n,:].T


        #     reg_model.fit(input_vel, output_vel)

        #     sol_vel = atleast_2d(reg_model.coef_)
        #     self.C = sol_vel
            
        #     # Construct A matrix
        #     self.A = diag(self.basis.Lambda)


        #     # # Construct B matrix:
        #     # self.B = concatenate(( B_vel, B_eig), axis=0)
            
        #     l=3
        #     phi=Z[self.n:, :]
        #     phi_,phi0 = self.sample(l,phi,0)
        #     UU=self.sample(l,U,1)            
        #     A_nom=diag(self.basis.Lambda)
        #     AA=A_nom
        #     AAA=[eye(A_nom.shape[0]),A_nom]
        #     # for i in range(2,l+1):
        #     #     AA=concatenate(( AA,dot(AAA[-1],A_nom)), axis=0)
        #     #     AAA.append(dot(AAA[-1],A_nom))
        #     for i in range(2,l+1):
        #         AA=concatenate(( AA,A_nom**i), axis=0)
        #         AAA.append(A_nom**i)
        #     BB= kron((UU[0]).T,eye(A_nom.shape[0]))
        #     for i in range(1,l):
        #         a=kron((UU[0]).T,AAA[i])
        #         for j in range(1,i+1):
        #             a=a+kron((UU[j]).T,AAA[i-j])
        #         BB=concatenate(( BB,a), axis=0)
        #     input_eig = BB
        #     output_eig =( (phi_ - dot(AA, phi0)).T).flatten()
        #     output_eig=output_eig.reshape(len(output_eig),1)
        #     # B_eig = atleast_2d(dot(linalg.pinv(BB),output_eig).reshape(A_nom.shape[0],-1))
        #     # B_eig = atleast_2d(dot(linalg.pinv(input_eig), output_eig))
        #     reg_model.fit(input_eig, output_eig)
        #     B_eig_cof = atleast_2d(reg_model.coef_)
        #     B_eig=B_eig_cof.T
        #     # self.B = concatenate(( B_vel, B_eig), axis=0)
        #     self.B=B_eig

        #     if self.override_C:
        #         # self.C = zeros((self.n, self.n_lift))
        #         # self.C[:self.n, :self.n] = eye(self.n)
        #         a=self.Z_std.transpose()
        #         self.C = multiply(self.C, a[:,self.n])
        #         # self.C=self.C
        #     else:
        #         raise Exception('Warning: Learning of C not implemented for structured regression.')

        # if not self.episodic:
        #     if self.K_p is None or self.K_p is None:
        #         raise Exception('Nominal controller gains not defined.')
        #     # Take nominal controller into account:
        #     # self.A[self.n:,:self.n] -= dot(self.B[self.n:,:],concatenate((self.K_p, self.K_d), axis=1))
        #     #B_apnd = zeros_like(self.B)   #TODO: Revert to run modified controller adjustment
        #     #B_apnd[self.n:,:] = -self.B[self.n:, :]
        #     #self.B = concatenate((self.B,B_apnd), axis=1)
            
        else:
                reg_model = linear_model.Lasso(alpha=self.l1,  fit_intercept=False,
                                                             normalize=False, selection='random', max_iter=1e5)
    
                # reg_model = linear_model.ElasticNet(alpha=self.l1, l1_ratio=self.l1_ratio, fit_intercept=False,
                #                                              normalize=False, selection='random', max_iter=1e5)
    
                # Solve least squares problem to find A and B for velocity terms:
                if self.episodic:
                    input_vel = concatenate((Z, U-U_nom), axis=0).T
                else:
                    input_vel = concatenate((Z, U), axis=0).T
                output_vel = z_dot_uc[:self.n,:].T
                input_vel = z_uc.T
    
    
                reg_model.fit(input_vel, output_vel)
    
                sol_vel = atleast_2d(reg_model.coef_)
                A_vel = sol_vel
    
                # Construct A matrix
                self.A = zeros((self.n_lift, self.n_lift))
                # self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
                self.A[:self.n, :] = A_vel
                self.A[self.n:, self.n:] = diag(self.basis.Lambda)
    
                # Solve least squares problem to find B for position terms:
                # if self.episodic:
                #     input_pos = (U-U_nom).T
                # else:
                #     input_pos = U.T
                # output_pos = (Z_dot[:int(self.n / 2), :] - dot(self.A[:int(self.n / 2), :], Z)).T
                # reg_model.fit(input_pos, output_pos)
                # B_pos = atleast_2d(reg_model.coef_)
    
    
                # Solve least squares problem to find B for eigenfunction terms:
                #input_eig = (U - U_nom).T
                # U_state_fb = dot(concatenate((self.K_p, self.K_d), axis=1),X)
                # input_eig = (U - U_state_fb).T
                # output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
                # reg_model.fit(input_eig, output_eig)
                # B_eig = atleast_2d(reg_model.coef_)
    
                # # Construct B matrix:
                # self.B = concatenate(( B_vel, B_eig), axis=0)
                
                l=3
                # phi=Z[self.n:, :]
                phi=Z
                phi_,phi0 = self.sample(l,phi,0)
                UU=self.sample(l,U,1)
                
                # A_nom=diag(self.basis.Lambda)
                A_nom=self.A
                AA=A_nom
                AAA=[eye(A_nom.shape[0]),A_nom]
                # for i in range(2,l+1):
                #     AA=concatenate(( AA,dot(AAA[-1],A_nom)), axis=0)
                #     AAA.append(dot(AAA[-1],A_nom))
                for i in range(2,l+1):
                    AA=concatenate(( AA,A_nom**i), axis=0)
                    AAA.append(A_nom**i)
                BB= kron((UU[0]).T,eye(A_nom.shape[0]))
                for i in range(1,l):
                    a=kron((UU[0]).T,AAA[i])
                    for j in range(1,i+1):
                        a=a+kron((UU[j]).T,AAA[i-j])
                    BB=concatenate(( BB,a), axis=0)
                input_eig = BB
                output_eig =( (phi_ - dot(AA, phi0)).T).flatten()
                output_eig=output_eig.reshape(len(output_eig),1)
                # B_eig = atleast_2d(dot(linalg.pinv(BB),output_eig).reshape(A_nom.shape[0],-1))
                # B_eig = atleast_2d(dot(linalg.pinv(input_eig), output_eig))
                reg_model.fit(input_eig, output_eig)
                B_eig_cof = atleast_2d(reg_model.coef_)
                B_eig=B_eig_cof.T
                # self.B = concatenate(( B_vel, B_eig), axis=0)
                self.B=B_eig
    
                if self.override_C:
                    self.C = zeros((self.n, self.n_lift))
                    self.C[:self.n, :self.n] = eye(self.n)
                    self.C = multiply(self.C, self.Z_std.transpose())
                else:
                    raise Exception('Warning: Learning of C not implemented for structured regression.')
    
        if not self.episodic:
            if self.K_p is None or self.K_p is None:
                raise Exception('Nominal controller gains not defined.')
            # Take nominal controller into account:
            # self.A[self.n:,:self.n] -= dot(self.B[self.n:,:],concatenate((self.K_p, self.K_d), axis=1))
    
    # def sample(self,l,Z,flag):
    #     ##parameters
    #     tr=100
    #     lent=int(Z.shape[1]/tr)
    #     lent_new=lent-l
    #     zz=[]
    #     for j in range(l+1):
    #         z_new=zeros((Z.shape[0],int(Z.shape[1]-tr*(l-1))))
    #         for i in range(tr):
    #             z_new[:,int(i*lent_new):int((i+1)*lent_new)]=Z[:,int(i*lent)+j:int(i*lent+lent_new)+j]
    #         zz.append(z_new) 
    #     if flag==0:
    #         zzz=zz[1]
    #         for i in range(1,len(zz)):
    #             zzz=concatenate((zzz, zz[i]),axis=0) 
    #     elif flag==1:
    #         zzz=zz[0]
    #         for i in range(1,len(zz)-1):
    #             zzz=concatenate((zzz, zz[i]),axis=0) 
    #     return zzz  
    def sample(self, l,Z,flag):
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
        if flag==0:
            zzz=zz[1]
            for i in range(2,len(zz)):
                zzz=concatenate((zzz, zz[i]),axis=0) 
            return zzz,zz[0]
        elif flag==1:
            # zzz=zz[0]
            # for i in range(1,len(zz)-1):
            #     zzz=concatenate((zzz, zz[i]),axis=0) 
            return zz
        #################################### example test start
        # l=3
        # phi=Z[2:, :]
        # phi_, phi0, UU, U_nomm =sample(l,phi,0),sample(l,U,1),sample(l,U_nom,1)
        # UU=sample(l,U,1)
        # U_nomm=sample(l,U_nom,1)
        # AA=A_nom
        # AAA=[np.eye(A_nom.shape[0],A_nom.shape[1]),A_nom]
        # for i in range(2,l+1):
        #     AA=concatenate(( AA,dot(AAA[-1],A_nom)), axis=0)
        #     AAA.append(dot(AAA[-1],A_nom))
        # BB= np.kron((UU[0] - U_nomm[0]).T,np.eye(n))
        # for i in range(1,l):
        #     a=np.kron((UU[0] - U_nomm[0]).T,AAA[i])
        #     for j in range(1,i+1):
        #         a=a+np.kron((UU[j] - U_nomm[j]).T,AAA[i-j])
        #     BB=concatenate(( BB,a), axis=0)
            
        # output_eig =( (phi_ [:6,:]- dot(AA, phi0[:2,:])).T).flatten()
        # output_eig=output_eig.reshape(len(output_eig),1)
        # B_eig = atleast_2d(dot(linalg.pinv(BB),output_eig).reshape(A_nom.shape[0],-1))
        
        ###################################   example test end


    def lift(self, X, X_d):
        Z = self.basis.lift(X, X_d)
        output_norm = divide(concatenate((X.T, Z),axis=1),self.Z_std.transpose())
        # output_norm = divide(Z,self.Z_std.transpose())
        return output_norm





# class KMdmd(Edmd):
#     def __init__(self, basis, system_dim, l1_pos=0., l1_ratio_pos=0.5, l1_vel=0., l1_ratio_vel=0.5, l1_eig=0., l1_ratio_eig=0.5, acceleration_bounds=None, override_C=False, K_p = None, K_d = None, episodic=False):
#         super().__init__(basis, system_dim, l1=l1_vel, l1_ratio=l1_ratio_vel, acceleration_bounds=acceleration_bounds, override_C=override_C)
#         self.episodic = episodic
#         self.K_p = K_p
#         self.K_d = K_d
#         self.Z_std = ones((basis.Nlift + basis.n, 1))
#         self.l1_pos = l1_pos
#         self.l1_ratio_pos = l1_ratio_pos
#         self.l1_vel = l1_vel
#         self.l1_ratio_vel =  l1_ratio_vel
#         self.l1_eig = l1_eig
#         self.l1_ratio_eig = l1_ratio_eig
#         self.l1 = 0.

#         if self.basis.Lambda is None:
#             raise Exception('Basis provided is not an Koopman eigenfunction basis')

#     def fit(self, X, X_d, Z, Z_dot, U, U_nom,A_nomi,B_nomi):
#         self.n_lift = Z.shape[0]

#         if self.l1 == 0.:
#             # Solve least squares problem to find A and B for velocity terms:
#             output_vel = (Z_dot[:self.n, :]-dot(A_nomi,Z[:self.n, :])-dot(B_nomi,U)).T
#             input_vel = Z[self.n:, :].T
#             sol_vel = atleast_2d(dot(linalg.pinv(input_vel),output_vel).transpose())
#             A_vel = sol_vel
#             B_vel=B_nomi
            
#             if self.episodic:
#                 input_vel = concatenate((Z, U-U_nom),axis=0).T
#             else:
#                 input_vel = concatenate((Z, U), axis=0).T
#             output_vel = Z_dot[:self.n,:].T
#             sol_vel = atleast_2d(dot(linalg.pinv(input_vel),output_vel).transpose())
#             A_vel = sol_vel[:,:self.n_lift]
#             B_vel = sol_vel[:,self.n_lift:]

#             # Construct A matrix
#             self.A = zeros((self.n_lift, self.n_lift))
#             # self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
#             self.A[:self.n,:] = A_vel
#             self.A[self.n:,self.n:] = diag(self.basis.Lambda)

#             # Construct A matrix
#             self.A = zeros((self.n_lift, self.n_lift))
#             # self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
#             self.A[:self.n, :self.n] = A_nomi
#             self.A[:self.n, self.n:] = A_vel
#             self.A[self.n:, self.n:] = diag(self.basis.Lambda)

#             # Solve least squares problem to find B for position terms:
#             ### new_one
#             l=3
#             phi=Z[self.n:, :]
#             phi_,phi0 = self.sample(l,phi,0)
#             UU=self.sample(l,U,1)
#             U_nomm=self.sample(l,U_nom,1)
#             A_nom=diag(self.basis.Lambda)
#             AA=A_nom
#             AAA=[eye(A_nom.shape[0]),A_nom]
#             # for i in range(2,l+1):
#             #     AA=concatenate(( AA,dot(AAA[-1],A_nom)), axis=0)
#             #     AAA.append(dot(AAA[-1],A_nom))
#             for i in range(2,l+1):
#                 AA=concatenate(( AA,A_nom**i), axis=0)
#                 AAA.append(A_nom**i)
#             BB= kron((UU[0] - U_nomm[0]).T,eye(A_nom.shape[0]))
#             for i in range(1,l):
#                 a=kron((UU[0] - U_nomm[0]).T,AAA[i])
#                 for j in range(1,i+1):
#                     a=a+kron((UU[j] - U_nomm[j]).T,AAA[i-j])
#                 BB=concatenate(( BB,a), axis=0)
#             input_eig = BB
#             output_eig =( (phi_ - dot(AA, phi0)).T).flatten()
#             output_eig=output_eig.reshape(len(output_eig),1)
#             # B_eig = atleast_2d(dot(linalg.pinv(BB),output_eig).reshape(A_nom.shape[0],-1))
#             B_eig = atleast_2d(dot(linalg.pinv(input_eig), output_eig))
#             # B_eig = atleast_2d(dot(linalg.pinv(A_matrix),B_eig_cof).transpose())
    
#             # Construct B matrix:
#             self.B = concatenate(( B_vel, B_eig), axis=0)

#             # Solve least squares problem to find B for eigenfunction terms:
#             # U_state_fb = dot(concatenate((self.K_p, self.K_d), axis=1), X)
#             # input_eig = (U - U_state_fb).T
#             # output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
#             # B_eig = atleast_2d(dot(linalg.pinv(input_eig), output_eig).transpose())

#             # Construct B matrix:
#             # self.B = concatenate(( B_vel, B_eig), axis=0)

#             if self.override_C:
#                 self.C = zeros((self.n,self.n_lift))
#                 self.C[:self.n,:self.n] = eye(self.n)
#                 self.C = multiply(self.C, self.Z_std.transpose())
#                 raise Exception('Warning: Learning of C not implemented for structured regression.')

#         else:
#             reg_model = linear_model.ElasticNet(alpha=self.l1, l1_ratio=self.l1_ratio, fit_intercept=False,
#                                                          normalize=False, selection='random', max_iter=1e5)

#             # Solve least squares problem to find A and B for velocity terms:
#             if self.episodic:
#                 input_vel = concatenate((Z, U-U_nom), axis=0).T
#             else:
#                 input_vel = concatenate((Z, U), axis=0).T
#             output_vel = (Z_dot[:self.n, :]-dot(A_nomi,Z[:self.n, :])-dot(B_nomi,U)).T


#             reg_model.fit(input_vel, output_vel)

#             sol_vel = atleast_2d(reg_model.coef_)
#             A_vel = sol_vel[:, :self.n_lift]
#             B_vel = sol_vel[:, self.n_lift:]

#             # Construct A matrix
#             self.A = zeros((self.n_lift, self.n_lift))
#             # self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
#             self.A[:self.n, :] = A_vel
#             self.A[self.n:, self.n:] = diag(self.basis.Lambda)

#             # Solve least squares problem to find B for position terms:
#             # if self.episodic:
#             #     input_pos = (U-U_nom).T
#             # else:
#             #     input_pos = U.T
#             # output_pos = (Z_dot[:int(self.n / 2), :] - dot(self.A[:int(self.n / 2), :], Z)).T
#             # reg_model.fit(input_pos, output_pos)
#             # B_pos = atleast_2d(reg_model.coef_)


#             # Solve least squares problem to find B for eigenfunction terms:
#             #input_eig = (U - U_nom).T
#             U_state_fb = dot(concatenate((self.K_p, self.K_d), axis=1),X)
#             input_eig = (U - U_state_fb).T
#             output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
#             reg_model.fit(input_eig, output_eig)
#             B_eig = atleast_2d(reg_model.coef_)

#             # Construct B matrix:
#             self.B = concatenate(( B_vel, B_eig), axis=0)

#             if self.override_C:
#                 self.C = zeros((self.n, self.n_lift))
#                 self.C[:self.n, :self.n] = eye(self.n)
#                 self.C = multiply(self.C, self.Z_std.transpose())
#             else:
#                 raise Exception('Warning: Learning of C not implemented for structured regression.')

#         if not self.episodic:
#             if self.K_p is None or self.K_p is None:
#                 raise Exception('Nominal controller gains not defined.')
#             # Take nominal controller into account:
#             self.A[self.n:,:self.n] -= dot(self.B[self.n:,:],concatenate((self.K_p, self.K_d), axis=1))
#             #B_apnd = zeros_like(self.B)   #TODO: Revert to run modified controller adjustment
#             #B_apnd[self.n:,:] = -self.B[self.n:, :]
#             #self.B = concatenate((self.B,B_apnd), axis=1)
    
#     # def sample(self,l,Z,flag):
#     #     ##parameters
#     #     tr=100
#     #     lent=int(Z.shape[1]/tr)
#     #     lent_new=lent-l
#     #     zz=[]
#     #     for j in range(l+1):
#     #         z_new=zeros((Z.shape[0],int(Z.shape[1]-tr*(l-1))))
#     #         for i in range(tr):
#     #             z_new[:,int(i*lent_new):int((i+1)*lent_new)]=Z[:,int(i*lent)+j:int(i*lent+lent_new)+j]
#     #         zz.append(z_new) 
#     #     if flag==0:
#     #         zzz=zz[1]
#     #         for i in range(1,len(zz)):
#     #             zzz=concatenate((zzz, zz[i]),axis=0) 
#     #     elif flag==1:
#     #         zzz=zz[0]
#     #         for i in range(1,len(zz)-1):
#     #             zzz=concatenate((zzz, zz[i]),axis=0) 
#     #     return zzz  
#     def sample(self, l,Z,flag):
#         ##parameters
#         tr=100
#         lent=int(Z.shape[1]/tr)
#         lent_new=lent-l
#         zz=[]
#         for j in range(l+1):
#             z_new=zeros((Z.shape[0],int(Z.shape[1]-tr*(l-1))))
#             for i in range(tr):
#                 z_new[:,int(i*lent_new):int((i+1)*lent_new)]=Z[:,int(i*lent)+j:int(i*lent+lent_new)+j]
#             zz.append(z_new) 
#         if flag==0:
#             zzz=zz[1]
#             for i in range(2,len(zz)):
#                 zzz=concatenate((zzz, zz[i]),axis=0) 
#             return zzz,zz[0]
#         elif flag==1:
#             # zzz=zz[0]
#             # for i in range(1,len(zz)-1):
#             #     zzz=concatenate((zzz, zz[i]),axis=0) 
#             return zz
#         #################################### example test start
#         # l=3
#         # phi=Z[2:, :]
#         # phi_, phi0, UU, U_nomm =sample(l,phi,0),sample(l,U,1),sample(l,U_nom,1)
#         # UU=sample(l,U,1)
#         # U_nomm=sample(l,U_nom,1)
#         # AA=A_nom
#         # AAA=[np.eye(A_nom.shape[0],A_nom.shape[1]),A_nom]
#         # for i in range(2,l+1):
#         #     AA=concatenate(( AA,dot(AAA[-1],A_nom)), axis=0)
#         #     AAA.append(dot(AAA[-1],A_nom))
#         # BB= np.kron((UU[0] - U_nomm[0]).T,np.eye(n))
#         # for i in range(1,l):
#         #     a=np.kron((UU[0] - U_nomm[0]).T,AAA[i])
#         #     for j in range(1,i+1):
#         #         a=a+np.kron((UU[j] - U_nomm[j]).T,AAA[i-j])
#         #     BB=concatenate(( BB,a), axis=0)
            
#         # output_eig =( (phi_ [:6,:]- dot(AA, phi0[:2,:])).T).flatten()
#         # output_eig=output_eig.reshape(len(output_eig),1)
#         # B_eig = atleast_2d(dot(linalg.pinv(BB),output_eig).reshape(A_nom.shape[0],-1))
        
#         ###################################   example test end

#     def tune_fit(self, X, X_d, Z, Z_dot, U, U_nom, A_nomi,B_nomi, l1_ratio=array([1])):

#         reg_model_cv = linear_model.MultiTaskElasticNetCV(l1_ratio=l1_ratio, fit_intercept=False,
#                                             normalize=False, cv=5, n_jobs=-1, selection='random', max_iter=1e5)

#         # Solve least squares problem to find A and B for velocity terms:
#         output_vel = (Z_dot[:self.n, :]-dot(A_nomi,Z[:self.n, :])-dot(B_nomi,U)).T
#         input_vel = Z[self.n:, :].T

#         reg_model_cv.fit(input_vel, output_vel)

#         sol_vel = atleast_2d(reg_model_cv.coef_)
#         A_vel = sol_vel
#         # B_vel = sol_vel[:, self.n_lift:]
#         B_vel=B_nomi
#         self.l1_vel = reg_model_cv.alpha_
#         self.l1_ratio_vel = reg_model_cv.l1_ratio_

#         # Construct A matrix
#         self.A = zeros((self.n_lift, self.n_lift))
#         # self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
#         self.A[:self.n, :self.n] = A_nomi
#         self.A[:self.n, self.n:] = A_vel
#         self.A[self.n:, self.n:] = diag(self.basis.Lambda)

#         # Solve least squares problem to find B for position terms:
#         # if self.episodic:
#         #     input_pos = (U - U_nom).T
#         # else:
#         #     input_pos = U.T
#         # output_pos = (Z_dot[:int(self.n / 2), :] - dot(self.A[:int(self.n / 2), :], Z)).T
#         # reg_model_cv.fit(input_pos, output_pos)
#         # B_pos = atleast_2d(reg_model_cv.coef_)
#         self.l1_pos = reg_model_cv.alpha_
#         self.l1_ratio_pos = reg_model_cv.l1_ratio_

#         # Solve least squares problem to find B for eigenfunction terms:
#         # input_eig = (U - U_nom).T
#         # output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
#         # reg_model_cv.fit(input_eig, output_eig)
#         # B_eig = atleast_2d(reg_model_cv.coef_)
#         # self.l1_eig = reg_model_cv.alpha_
#         # self.l1_ratio_eig = reg_model_cv.l1_ratio_    

#         ### new_one
#         l=3
#         phi=Z[self.n:, :]
#         phi_,phi0 = self.sample(l,phi,0)
#         UU=self.sample(l,U,1)
#         U_nomm=self.sample(l,U_nom,1)
#         A_nom=diag(self.basis.Lambda)
#         AA=A_nom
#         AAA=[eye(A_nom.shape[0]),A_nom]
#         # for i in range(2,l+1):
#         #     AA=concatenate(( AA,dot(AAA[-1],A_nom)), axis=0)
#         #     AAA.append(dot(AAA[-1],A_nom))
#         for i in range(2,l+1):
#             AA=concatenate(( AA,A_nom**i), axis=0)
#             AAA.append(A_nom**i)
#         BB= kron((UU[0] - U_nomm[0]).T,eye(A_nom.shape[0]))
#         for i in range(1,l):
#             a=kron((UU[0] - U_nomm[0]).T,AAA[i])
#             for j in range(1,i+1):
#                 a=a+kron((UU[j] - U_nomm[j]).T,AAA[i-j])
#             BB=concatenate(( BB,a), axis=0)
#         input_eig = BB
#         output_eig =( (phi_ - dot(AA, phi0)).T).flatten()
#         output_eig=output_eig.reshape(len(output_eig),1)
#         # B_eig = atleast_2d(dot(linalg.pinv(BB),output_eig).reshape(A_nom.shape[0],-1))
#         reg_model_cv.fit(input_eig, output_eig)
#         B_eig_cof = atleast_2d(reg_model_cv.coef_)
#         B_eig=B_eig_cof.T
#         # B_eig = atleast_2d(dot(linalg.pinv(A_matrix),B_eig_cof).transpose())
#         self.l1_eig = reg_model_cv.alpha_
#         self.l1_ratio_eig = reg_model_cv.l1_ratio_

#         # Construct B matrix:
#         self.B = concatenate(( B_vel, B_eig), axis=0)

#         if self.override_C:
#             self.C = zeros((self.n, self.n_lift))
#             self.C[:self.n, :self.n] = eye(self.n)
#             self.C = multiply(self.C, self.Z_std.transpose())
#         else:
#             raise Exception('Warning: Learning of C not implemented for structured regression.')

#         if not self.episodic:
#             if self.K_p is None or self.K_p is None:
#                 raise Exception('Nominal controller gains not defined.')
#             self.A[self.n:, :self.n] -= dot(self.B[self.n:, :], concatenate((self.K_p, self.K_d), axis=1))

#         print('KEEDMD l1 (pos, vel, eig): ', self.l1_pos, self.l1_vel, self.l1_eig)
#         print('KEEDMD l1 ratio (pos, vel, eig): ', self.l1_ratio_pos, self.l1_ratio_vel, self.l1_ratio_eig)

#     def lift(self, X, X_d):
#         Z = self.basis.lift(X, X_d)
#         output_norm = divide(concatenate((X.T, Z),axis=1),self.Z_std.transpose())
#         return output_norm
