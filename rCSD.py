# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:25:45 2021

@author: xinqi
"""

import copy
from matplotlib.pyplot import grid
import numpy as np
from forward_models import b_fwd_1d
import multiprocessing
import os
from tqdm import tqdm
from gpcsd.utility_functions import normalize
PROCESSES = os.cpu_count()-2

def cross_validation_single(y, z,x, intervals_csd, para_name, para_range, my_dicts):
    Cv_Rep_Trial = 24
    nx = len(x)
    delete_electrode_list = np.random.choice(range(nx), Cv_Rep_Trial, replace=False)
    cv_error = np.zeros((len(para_range),Cv_Rep_Trial))
    frames = np.arange(0,10)
    
    for i, para in enumerate(para_range):
        my_dicts[para_name] = para
        A = b_fwd_1d(z.T-x, my_dicts['R'])
        print(para_name,para)
        for i_delete, delete_electrode in enumerate(delete_electrode_list):

            csd_temp = region_conservation(np.delete(y[:,frames], delete_electrode, axis=0), 
                                           np.delete(A, delete_electrode, axis=0), intervals_csd, 
                                           my_dicts['lam_smooth'], my_dicts['lam_region'], my_dicts['lam_lasso'])
            y_temp = A@csd_temp
            cv_error[i,i_delete] =  np.linalg.norm(y_temp[delete_electrode,:]-y[delete_electrode,frames])**2
    return cv_error


def cross_validation_single_parallel(y, z,x, intervals_csd, para_name, para_range, my_dicts):
    Cv_Rep_Trial = 24
    nx = len(x)
    delete_electrode_list = np.random.choice(range(nx), Cv_Rep_Trial, replace=False)
    cv_error = np.zeros((len(para_range),Cv_Rep_Trial))
    frames = np.arange(0,y.shape[1])
    
    for i, para in enumerate(para_range):
        my_dicts[para_name] = para
        A = b_fwd_1d(z.T-x, my_dicts['R'])
        print(para_name,para)
        Y_pool = np.zeros((y.shape[0]-1,len(frames),Cv_Rep_Trial))
        A_pool = np.zeros((A.shape[0]-1,A.shape[1],Cv_Rep_Trial))
        for i_delete, delete_electrode in enumerate(delete_electrode_list):
            Y_pool[:,:,i_delete] = np.delete(y[:,frames], delete_electrode, axis=0)
            A_pool[:,:,i_delete] = np.delete(A, delete_electrode, axis=0)

        csd_pool = region_conservation_multiple_trial(Y_pool, A_pool, intervals_csd, 
                                           my_dicts['lam_smooth'], my_dicts['lam_region'], my_dicts['lam_lasso'])
        for i_delete, delete_electrode in enumerate(delete_electrode_list):
            y_temp = A@csd_pool[:,:,i_delete]
            cv_error[i,i_delete] =  np.linalg.norm(y_temp[delete_electrode,:]-y[delete_electrode,frames])**2
    return cv_error

def region_conservation_multiple_trial(Y_multi, A, intervals_csd, lam_smooth, lam_region, lam_lasso):
    import multiprocessing
    from rCSD import region_conservation
    import os
    import numpy as np
    PROCESSES = os.cpu_count()-2
    with multiprocessing.Pool(processes = PROCESSES) as pool:
        nz = A.shape[1]
        nx, nt, ntrial = Y_multi.shape
        csd = np.zeros((nz,nt,ntrial))
        ntrial = np.arange(Y_multi.shape[2]).tolist()
        if np.ndim(A) == 3:
            results = [pool.apply_async(region_conservation, (Y_multi[:,:,itrial], A[:,:,itrial], intervals_csd, lam_smooth, 
                                                          lam_region, lam_lasso)) for itrial in ntrial]
        else:
            results = [pool.apply_async(region_conservation, (Y_multi[:,:,itrial], A, intervals_csd, lam_smooth, 
                                                          lam_region, lam_lasso)) for itrial in ntrial]

        [result.wait() for result in results]
        i = 0
        for result in results:
            csd[:,:,i] = result.get()
            i += 1
        pool.close()
    return csd


def region_conservation(Y, A, intervals_csd, lam_smooth, lam_region, lam_lasso):
    nx, nz = A.shape
    nt = Y.shape[1]
    csd_lasso = np.zeros((nz, nt))
    csd_ini = np.zeros((nz))
    nregion = intervals_csd.shape[0]-1
    mat_region = np.zeros((nz, nregion))
    # mat_region[0,0] = 1
    for iregion in range(nregion):
        mat_region[intervals_csd[iregion]:intervals_csd[iregion+1], iregion].fill(1)
    zeros_vector = np.zeros((nz-1, 1))
    iden_mat_smooth = np.eye(nz-1)
    mat_diff = np.block([iden_mat_smooth, zeros_vector]) - \
        np.block([zeros_vector, iden_mat_smooth])

    for it in range(nt):
        csd_lasso[:, it] = region_conservation_frame(Y[:, it], A, mat_region, mat_diff,
                                                     csd_ini, lam_smooth, lam_region, lam_lasso)
        # print('frame being proccessing:', it)
        csd_ini = csd_lasso[:, it]
    return csd_lasso

def region_conservation_frame(y, A, mat_region, mat_diff, csd_ini, lam_smooth, lam_region, lam_lasso):
    # A few key point in the optimization algorithm:
    
    # First part update all coefficients simultaneously. 
    # nloop: maximum iteration; 
    # nDichotomy: search steps = 1/2**iDichotomy (nDichotomy is maximum of iDichotomy)
    
    # Second part update all coefficients immediately (greedy)
    # non_update: if the nonzero coefficients doesn't change for some iteration, it stops
    
    nx, nz = A.shape
    nloop = 10
    # y = y[:,None]
    csd_best = copy.deepcopy(csd_ini)
    # csd_best = np.zeros((nz,1))
    csd_temp = copy.deepcopy(csd_ini)
    csd_best_value = np.inf
    csd_temp_value = np.inf
    csd_record = np.zeros((nz, nloop))
    csd_record[:, 0] = csd_ini.squeeze()
    non_update = 0
    for iloop in range(1, nloop):
        update_weight = np.zeros((nz, 1))
        csd_temp = csd_record[:, iloop-1]
        for iz in range(nz):
            rho = 0
            iregion = np.where(mat_region[iz, :])[0][0]
            csd_mask_j = copy.deepcopy(csd_temp)
            csd_mask_j[iz] = 0

            if iz == 0:
                rho += lam_smooth*(csd_temp[1])
            elif iz == nz-1:
                rho += lam_smooth*(csd_temp[nz-2])
            else:
                rho += lam_smooth*(csd_temp[iz-1] + csd_temp[iz+1])
            rho += -lam_region*(mat_region[:, iregion]*csd_mask_j).sum()
            rho += (A[:, iz]*(y-A@csd_mask_j)).sum()
            k = 2*lam_smooth + lam_region + np.linalg.norm(A[:, iz])**2

            if rho < -lam_lasso:
                new_value = (rho+lam_lasso)/k
            elif rho > lam_lasso:
                new_value = (rho-lam_lasso)/k
            else:
                new_value = 0

            update_weight[iz] = new_value - csd_temp[iz]
        
        # search for the best value in current direction (1/2 step, 1/4 step, 1/8 step ...)
        nDichotomy = 10
        csd_search_value = np.nan*np.zeros(nDichotomy)
        csd_temp = csd_record[:, iloop-1]
        csd_search_value[0] = np.inf
        for iDichotomy in range(1,nDichotomy):
            csd_temp = csd_record[:, iloop-1] + 1/2**iDichotomy*update_weight.squeeze()
            csd_search_value[iDichotomy] = 1/2*np.linalg.norm(y-A@csd_temp)**2 \
                + lam_smooth/2*np.linalg.norm(mat_diff@csd_temp)**2 \
                + lam_region/2*np.linalg.norm(mat_region.T@csd_temp)**2 \
                + lam_lasso*np.linalg.norm(csd_temp, 1)
            if csd_search_value[iDichotomy] > csd_search_value[iDichotomy-1]:
                csd_temp = csd_record[:, iloop-1] + 1/2**(iDichotomy-1)*update_weight.squeeze()
                break
        
        csd_record[:, iloop] = csd_temp.squeeze()
        if csd_search_value[iDichotomy] <= csd_best_value:
            csd_best_value = csd_search_value[iDichotomy-1]
            csd_best = copy.deepcopy(csd_temp)
            non_update = 0
        else:
            non_update += 1

        if non_update >= 3:
            break

    nloop = 1000
    csd_record = np.zeros((nz, nloop))
    # update immediately
    for iloop in range(1, nloop):
        update_weight = np.zeros((nz, 1))
        csd_temp = copy.deepcopy(csd_best)
        for iz in range(nz):
            rho = 0
            iregion = np.where(mat_region[iz, :])[0][0]
            csd_mask_j = copy.deepcopy(csd_temp)
            csd_mask_j[iz] = 0

            if iz == 0:
                rho += lam_smooth*(csd_temp[1])
            elif iz == nz-1:
                rho += lam_smooth*(csd_temp[nz-2])
            else:
                rho += lam_smooth*(csd_temp[iz-1] + csd_temp[iz+1])
            rho += -lam_region*(mat_region[:, iregion]*csd_mask_j).sum()
            rho += (A[:, iz]*(y-A@csd_mask_j)).sum()
            k = 2*lam_smooth + lam_region + np.linalg.norm(A[:, iz])**2

            if rho < -lam_lasso:
                new_value = (rho+lam_lasso)/k
            elif rho > lam_lasso:
                new_value = (rho-lam_lasso)/k
            else:
                new_value = 0

            csd_temp[iz] = new_value

        csd_record[:, iloop] = csd_temp.squeeze()

        csd_temp_value = 1/2*np.linalg.norm(y-A@csd_temp)**2 \
            + lam_smooth/2*np.linalg.norm(mat_diff@csd_temp)**2 \
            + lam_region/2*np.linalg.norm(mat_region.T@csd_temp)**2 \
            + lam_lasso*np.linalg.norm(csd_temp, 1)
        
        if csd_temp_value <= csd_best_value:
            csd_best = copy.deepcopy(csd_temp)
            csd_best_value = csd_temp_value
            
        if all((csd_record[:, iloop]==0)==(csd_record[:, iloop-1]==0)):
            non_update += 1
        else:
            non_update = 0
            
    # if update_step<=1e-5:
        if non_update >= 5:
            break

    return csd_best.squeeze()




class rCSD:
    def __init__(
            self, 
            Y, 
            z, 
            x, 
            R=None, 
            boundary_depth=[], 
            lam_smooth=None, 
            lam_region=None, 
            lam_lasso=None
        ):
        self.Y = Y
        if self.Y.ndim == 2:
            self.Y = self.Y[:, :, None]
        self.z = z
        self.x = x
        self.nz = z.shape[0]
        self.nx = self.x.shape[0]
        self.nt = Y.shape[1]
        self.ntrial = Y.shape[2]

        self.R = R
        self.boundary_depth = boundary_depth
        self.lam_smooth = lam_smooth
        self.lam_region = lam_region
        self.lam_lasso = lam_lasso
        if self.R is not None:
            # shape of A: (ny, nz)
            self.A = b_fwd_1d(self.z.T-self.x, self.R)
        else:
            self.A = None
        self.boundary = [0] + [
            np.nonzero(self.z > self.boundary_depth[i])[0][0] 
            for i in range(len(self.boundary_depth))
        ] + [self.nz]
        self.nregion = len(self.boundary) - 1

        # region constraint matrix for charge conservation
        self.mat_region = np.zeros((self.nregion, self.nz))
        for iregion in range(self.nregion):
            self.mat_region[iregion, self.boundary[iregion]:self.boundary[iregion+1]] = 1.0

        # Create a first-order difference matrix for smoothness constraint
        zeros_col = np.zeros((self.nz-1, 1))
        identity = np.eye(self.nz-1)
        self.mat_diff = np.block([identity, zeros_col]) - np.block([zeros_col, identity])

    def cv_electrode(self, rep=10, del_electrode_list=None, return_var=False, return_pred=False, 
            verbose=True):
        A_ori, Y_ori = self.A, self.Y
        errors = []
        lfp_pred_rcd_rcsd = np.zeros_like(Y_ori)
        if del_electrode_list is None:
            del_electrode_list = np.random.choice(range(1, self.nx-1), rep, replace=False)
        for del_electrode in tqdm(del_electrode_list, disable=not verbose):
            self.A = np.delete(A_ori, del_electrode, axis=0)
            self.Y = np.delete(Y_ori, del_electrode, axis=0)
            csd_temp = self.predict(verbose=False)
            lfp_pred = normalize(np.einsum('xz,ztm->xtm', self.A, csd_temp))
            lfp_pred_rcd_rcsd[del_electrode, :, :] = lfp_pred[del_electrode, :, :]
            errors.append(np.mean((Y_ori[del_electrode, :, :] - lfp_pred[del_electrode, :, :] )**2))
        self.A, self.Y = A_ori, Y_ori
        to_return = [np.mean(errors)]
        if return_var:
            to_return.append(np.var(errors)/np.sqrt(len(errors)))
        if return_pred:
            to_return.append(lfp_pred_rcd_rcsd)
        return to_return
    
    def update_hp(self, dic):
        self.R, self.lam_lasso, self.lam_region, self.lam_smooth = (
            dic["R"], dic["lam_lasso"], dic["lam_region"], dic["lam_smooth"]
        )
        self.A = b_fwd_1d(self.z.T-self.x, self.R)
    
    def get_hp_dic(self):
        dic = {}
        dic["R"], dic["lam_lasso"], dic["lam_region"], dic["lam_smooth"] = (
            self.R, self.lam_lasso, self.lam_region, self.lam_smooth
        )
        return dic

    def cv_para(self, para_name, para_range, dic, rep=10):
        err_list, var_list = [], []
        hp_dic_ori = self.get_hp_dic()
        for para in tqdm(para_range):
            dic[para_name] = para
            self.update_hp(dic)
            err, var = self.cv_electrode(rep=rep, return_var=True, verbose=False)
            err_list.append(err)
            var_list.append(var)
        self.update_hp(hp_dic_ori)
        return err_list, var_list

    def predict(self, lr=1e-3, mu=1e-3, verbose=False, parallel=False):
        pred = np.zeros((self.nz, self.nt, self.ntrial))
        if parallel:
            with multiprocessing.Pool(processes=PROCESSES) as pool:
                results = [
                    pool.apply_async(
                        self.pred_rcsd_per_trial,
                        (itrial, )
                    ) for itrial in range(self.ntrial)
                ]

            [result.wait() for result in results]
            for itrial, result in enumerate(results):
                pred[:,:,itrial] = result.get()
        else:
            for trial in tqdm(range(self.ntrial), desc="Predicting rCSD", disable=not verbose):
                pred[:, :, trial] = self.pred_rcsd_per_trial(
                    trial=trial, lr=lr, mu=mu,
                )
        self.pred = pred
        return pred

    def pred_rcsd_per_trial(self, trial=0, lr=1e-4, mu=1e-4, verbose=False):
        pred = np.zeros((self.nz, self.nt))
        prev_csd = np.zeros((self.nz))
        for t in range(self.nt):
            # pred[:, t] = region_conservation_frame(
            #     self.Y[:, t, trial], self.A, self.mat_region.T, self.mat_diff, 
            #     prev_csd, self.lam_smooth, self.lam_region, self.lam_lasso,
            # )
            pred[:, t] = pred_rcsd_per_time(
                self.Y[:, t, trial], self.A, self.mat_region, self.mat_diff,
                prev_csd, self.lam_smooth, self.lam_region, self.lam_lasso,
                lr=lr, mu=mu,
            )
            prev_csd = pred[:, t]
        return pred

    def predict_old_implementation(self,):
        Y_smo = self.Y
        R = self.R
        A = b_fwd_1d(self.z.T-self.x, R)
        self.old_rcsd_pred = region_conservation_multiple_trial(
            Y_smo, A, np.array(self.boundary), self.lam_smooth, self.lam_region, self.lam_lasso
        )
        return self.old_rcsd_pred
    
    def get_difference(self):
        diff = np.linalg.norm(self.pred - self.predict_old_implementation())
        rel_diff = diff / np.linalg.norm(self.pred)
        print(f"Difference between old and new rCSD: {rel_diff}")

def get_approx_loss_and_grad(
        z, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso, 
        mu=1e-3, require_grad=True,
    ):
    # Smooth lasso: phi_mu(x) = sqrt(x^2 + mu^2) - mu
    smooth_lasso = np.sum(np.sqrt(z**2 + mu**2) - mu)
    if require_grad:
        smooth_lasso_grad = z / np.sqrt(z**2 + mu**2)
    
    loss = 1/2 * np.linalg.norm(y - A @ z)**2 \
        + lam_smooth / 2 * np.linalg.norm(mat_diff @ z)**2 \
        + lam_region / 2 * np.linalg.norm(mat_region @ z)**2 \
        + lam_lasso * smooth_lasso

    if require_grad:
        grad = (
            A.T @ (A @ z - y) \
            + lam_smooth * mat_diff.T @ (mat_diff @ z) \
            + lam_region * mat_region.T @ (mat_region @ z) \
            + lam_lasso * smooth_lasso_grad
        )

    return (loss, grad) if require_grad else loss

def get_loss(z, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso):
    loss = 1/2*np.linalg.norm(y-A@z)**2 \
        + lam_smooth/2*np.linalg.norm(mat_diff@z)**2 \
        + lam_region/2*np.linalg.norm(mat_region@z)**2 \
        + lam_lasso*np.linalg.norm(z, 1)
    return loss




def pred_rcsd_per_time(
        y, 
        A, 
        mat_region, 
        mat_diff, 
        z_ini, 
        lam_smooth, 
        lam_region, 
        lam_lasso,
        lr=1e-3,
        mu=1e-3,
        tol=1e-4,
        max_iter_tol=5,
        max_iter=int(1e3),
        Huber_approx=True,
        Lasso_exact=True,
    ):
    # A few key point in the optimization algorithm:
    
    # First part update all coefficients simultaneously. 
    # nloop: maximum iteration; 
    # nDichotomy: search steps = 1/2**iDichotomy (nDichotomy is maximum of iDichotomy)
    
    # Second part update all coefficients immediately (greedy)
    # non_update: if the nonzero coefficients doesn't change for some iteration, it stops
    
    ny, nz = A.shape
    # if y.ndim == 1:
    #     y = y[:,None]
    # if z_ini.ndim == 1:
    #     z_ini = z_ini[:,None]

    if Huber_approx:
        z_ini = pred_rcsd_per_time_approx(
            y, A, mat_region, mat_diff, z_ini, lam_smooth, lam_region, lam_lasso,
            lr_init=lr, mu=mu, tol=tol, max_iter_tol=max_iter_tol, max_iter=max_iter,
        )
    if not Lasso_exact:
        return z_ini.squeeze()
    
    # return region_conservation_frame(y, A, mat_region.T, mat_diff, z_ini, lam_smooth, lam_region, lam_lasso)

    csd_best = copy.deepcopy(z_ini)
    csd_temp = copy.deepcopy(z_ini)
    csd_best_value = np.inf
    csd_temp_value = np.inf
    csd_record = np.zeros((nz, max_iter))
    csd_record[:, 0] = z_ini.squeeze()
    non_update_iter = 0

    # # update simultaneously
    # max_iter_simul = 10
    # non_update_iter = 0
    # for iter in range(1, max_iter_simul):
    #     update_weight = np.zeros((nz, 1))
    #     csd_temp = csd_record[:, iter-1]
    #     for iz in range(nz):
    #         new_value = coordinate_descent(
    #             iz, csd_temp, y, A, mat_region, lam_smooth, lam_region, lam_lasso
    #         )
    #         update_weight[iz] = new_value - csd_temp[iz]
        
    #     # search for the best value in current direction (1/2 step, 1/4 step, 1/8 step ...)
    #     nDichotomy = 10
    #     csd_search_value = np.nan*np.zeros(nDichotomy)
    #     csd_search_value[0] = np.inf
    #     for iDichotomy in range(1,nDichotomy):
    #         csd_temp = csd_record[:, iter-1] + 1/2**iDichotomy*update_weight.squeeze()
    #         csd_search_value[iDichotomy] = get_loss(
    #             csd_temp, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso
    #         )
    #         if csd_search_value[iDichotomy] > csd_search_value[iDichotomy-1]:
    #             csd_temp = csd_record[:, iter-1] + 1/2**(iDichotomy-1)*update_weight.squeeze()
    #             break
        
    #     csd_record[:, iter] = csd_temp.squeeze()
    #     if csd_search_value[iDichotomy] <= csd_best_value:
    #         csd_best_value = csd_search_value[iDichotomy-1]
    #         csd_best = copy.deepcopy(csd_temp)
    #         non_update_iter = 0
    #     else:
    #         non_update_iter += 1

    #     if non_update_iter >= 3:
    #         break

    # update immediately
    non_update_iter = 0
    csd_record = np.zeros((nz, max_iter))
    for iter in range(1, max_iter):
        csd_temp = copy.deepcopy(csd_best)
        for iz in range(nz):
            new_value = coordinate_descent(
                iz, csd_temp, y, A, mat_region, lam_smooth, lam_region, lam_lasso
            )
            csd_temp[iz] = new_value

        csd_record[:, iter] = csd_temp.squeeze()

        csd_temp_value = get_loss(
            csd_temp, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso
        )
        
        if csd_temp_value < csd_best_value:
            csd_best = copy.deepcopy(csd_temp)
            csd_best_value = csd_temp_value
            
        if all((csd_record[:, iter]==0)==(csd_record[:, iter-1]==0)):
            non_update_iter += 1
        else:
            non_update_iter = 0

        if non_update_iter >= 5:
            break


    return csd_best.squeeze()


def coordinate_descent(iz, csd_temp, y, A, mat_region, lam_smooth, lam_region, lam_lasso):
    nz = csd_temp.shape[0]

    rho = 0
    iregion = np.where(mat_region[:, iz])[0][0]
    csd_mask_j = copy.deepcopy(csd_temp)
    csd_mask_j[iz] = 0

    if iz == 0:
        rho += lam_smooth*(csd_temp[1])
    elif iz == nz-1:
        rho += lam_smooth*(csd_temp[nz-2])
    else:
        rho += lam_smooth*(csd_temp[iz-1] + csd_temp[iz+1])
    rho += -lam_region*(mat_region[iregion, :]*csd_mask_j).sum()
    rho += (A[:, iz]*(y-A@csd_mask_j)).sum()
    k = 2*lam_smooth + lam_region + np.linalg.norm(A[:, iz])**2

    if rho < -lam_lasso:
        new_value = (rho+lam_lasso)/k
    elif rho > lam_lasso:
        new_value = (rho-lam_lasso)/k
    else:
        new_value = 0

    return new_value


def pred_rcsd_per_time_approx(
        y, 
        A, 
        mat_region, 
        mat_diff, 
        z_ini, 
        lam_smooth, 
        lam_region, 
        lam_lasso,
        lr_init=1e-2,
        mu=1e-3,
        tol=1e-4,
        max_iter_tol=1,
        max_iter=int(1e3),
        beta=0.5,
        c=1e-4,
    ):
    z = z_ini.copy()
    best_z = z.copy()
    best_loss = np.inf
    loss_prev = np.inf
    non_update_iter = 0

    for it in range(max_iter):

        loss, grad = get_approx_loss_and_grad(
            z, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso, mu
        )

        # Convergence check
        if loss > best_loss - tol:
            non_update_iter += 1
            if non_update_iter >= max_iter_tol:
                break
        else:
            non_update_iter = 0
            best_loss = loss
            best_z = z.copy()

        # Adaptive learning rate (backtracking line search)
        lr = lr_init
        while True:
            z_new = z - lr * grad
            loss_new = get_approx_loss_and_grad(
                z_new, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso,
                require_grad=False,
            )
            if loss_new <= loss - c * lr * np.linalg.norm(grad)**2:
                z = z_new
                break
            lr *= beta
            if lr < 1e-6:
                break
    
    return best_z