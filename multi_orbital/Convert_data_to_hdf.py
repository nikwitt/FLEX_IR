# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:45:04 2020

@author: nikla
"""

import sys

MKL_THREADS_VAR = str(sys.argv[1])

import os
os.environ["MKL_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["NUMEXPR_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import *
from einsum2 import einsum2
import scipy as sc
import pyfftw
from ir_load import ir_load
from parameters import parameters
from hamiltonian import hamiltonian
from gfunction import gfunction_load
from kpath_extract import kpath_extract
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time


#### For converison now
import h5py
sp_name = "NaxCoO2_calculation_data_T_{}_U_{}_JUratio_{}_n_{}.h5"

##### Please input in order: 
# MKL_NUM_THREADS | T | T_load | JUratio | JU_ratio_load | round_it
n_fill = (7-3.43)/2
T      = float(sys.argv[2])
T_load = float(sys.argv[3])
JU_ratio = float(sys.argv[4])
JU_ratio_load = float(sys.argv[5])

round_it = int(sys.argv[6])

print(T, T_load, JU_ratio, JU_ratio_load, round_it)


### Initiate parameters -------------------------------------------------
start = time.process_time()
p = parameters(round(T, 5), round(n_fill ,5), round(JU_ratio,5), round_it,\
               T_load = round(T_load, 5), JU_ratio_load = round(JU_ratio_load, 5))
print("##################################################"\
      , file=open(p.Logstr,'a'))
print("Conversion of .dat file for:"\
      , file=open(p.Logstr,'a'))
print(datetime.datetime.now().strftime('%d. %B %Y %I:%M%p')\
      , file=open(p.Logstr,'a'))
print("Parameter set: n = " + str(p.n_fill) + ", T = " + str(p.T)\
      + ", U = " + str(p.u0) + ", J_H = " + str(p.JU_ratio) + "U\n"\
      , file=open(p.Logstr,'a'))
print("Elapsed time - parameter init: " + str(time.process_time() - start)\
      , file=open(p.Logstr,'a')) 

    
print("Make hdf5 file"\
      , file=open(p.Logstr,'a')) 
savepath = p.sp_dir + sp_name.format(p.T, p.u0, p.JU_ratio, p.n_fill)
if not os.path.exists(savepath):
    with h5py.File(savepath,'a') as file: 
        metadata = {'System name' : 'Hubbard Square lattice',
                    'N_k1'        : p.nk1,
                    'N_k2'        : p.nk2,
                    'N_k3'        : p.nk3,
                    'Lambda_IR'   : p.Lambda,
                    'g_sfc_tol'   : p.g_sfc_tol,
                    'SC_sfc_tol'  : p.SC_sfc_tol,
                    'n_fill'      : p.n_fill,
                    'T'           : p.T,
                    'U'           : p.u0,
                    'JU_ratio'    : p.JU_ratio,}
                
        file.attrs.update(metadata)

### Load hamiltionian----------------------------------------------------            
t_hset = time.process_time()
h = hamiltonian(p)
print("Elapsed time - hamiltonian set (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_hset), file=open(p.Logstr,'a'))
    

### Load irbasis --------------------------------------------------------            
t_bload = time.process_time()
b = ir_load(p.Lambda, p.beta)
print("Elapsed time - basis load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_bload), file=open(p.Logstr,'a')) 


### Calculate full Greens function---------------------------------------
print("Load data from dat file...", file=open(p.Logstr,'a')) 
t_gcal = time.process_time()
dum = gfunction_load.func_load(p, "_gkio", 2)   
gkio = dum.reshape(len(dum)//(p.nwan**2),p.nwan,p.nwan,order='F')
gkio = gkio.reshape(size(gkio)//(p.nk*p.nwan**2),p.nk,p.nwan,p.nwan)
gkio = transpose(gkio, axes=(0,1,3,2))
print('G(io,k) done', end =" ", file=open(p.Logstr,'a'))

dum = gfunction_load.func_load(p, "_sigma", 2)   
sigma = dum.reshape(len(dum)//(p.nwan**2),p.nwan,p.nwan,order='F')
sigma = sigma.reshape(size(sigma)//(p.nk*p.nwan**2),p.nk,p.nwan,p.nwan)
sigma = transpose(sigma, axes=(0,1,3,2))
print('Sigma(io,k) done', end =" ", file=open(p.Logstr,'a'))

print("Elapsed time - data load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_gcal), file=open(p.Logstr,'a')) 

### Write it to hdf5 file--------------------------------------------------
print("Write data to hdf5 file...", file=open(p.Logstr,'a')) 
with h5py.File(savepath,'a') as file:
    group = file.require_group('gfunction')
            
    group.require_dataset('gkio' , data=gkio , shape=gkio.shape, dtype=gkio.dtype)
    group.require_dataset('sigma', data=sigma, shape=sigma.shape, dtype=sigma.dtype)
    
print("Now calculate BSE kernel and kpath of chi_charge (maximum):")
#G(tau, r)
grit = gkio.reshape(len(b.fm), p.nk1, p.nk2, p.nk3, p.nwan**2)
fft_object = pyfftw.builders.fftn(grit, axes=(1,2,3))
grit = fft_object()
grit = grit.reshape(len(b.fm),p.nk*p.nwan*p.nwan)
result, _, _, _  = sc.linalg.lstsq(b.fermi_Uln, grit, lapack_driver='gelsy')
grit_b = dot(b.fermi_Ulx_boson, result).reshape(len(b.bt),p.nk,p.nwan,p.nwan)
print('G(tau,r) [calc] done', end =" ", file=open(p.Logstr,'a')) 
        
#chi_0(iwn_bose, k)
grit_rev = grit_b[::-1,:,:,:]    #G_lm(r,beta-tau)
ckio = einsum2('ijkm,ijln->ijklmn', grit_b, grit_rev).reshape(len(b.bt),p.nk*p.nwan**4)#km ln
result, _, _, _  = sc.linalg.lstsq(b.bose_Ulx, ckio, lapack_driver='gelsy')
ckio = dot(b.bose_Uln, result)
ckio = ckio.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)
fft_object = pyfftw.builders.ifftn(ckio, axes=(1,2,3))
ckio = fft_object()/p.nk
ckio = ckio.reshape(len(b.bm),p.nk,p.nwan**2,p.nwan**2)
print('| Chi_0(iw,k) [calc] done', file=open(p.Logstr,'a'))     

#chi_spin
E_  = tensordot(ones(len(b.bm)), array([eye(p.nwan**2,p.nwan**2) for it in range(p.nk)]), axes=0)

chi_spin   = ckio@linalg.inv(E_ - ckio@h.S_mat)
chi_s = chi_spin[b.b_iwn_zero_ind].reshape(p.nk1,p.nk2,p.nk3,p.nwan**2,p.nwan**2)

chi_s_eig, _ = linalg.eigh(chi_s)

k_HSP, chi_s_HSP = kpath_extract.kpath_extractor(p, (chi_s_eig[:,:,:,-1]))

print('Chi_spin(0,k) [calc save] done', file=open(p.Logstr,'a'))  


chi_charge   = ckio@linalg.inv(E_ + ckio@h.C_mat)
chi_c = chi_charge[b.b_iwn_zero_ind].reshape(p.nk1,p.nk2,p.nk3,p.nwan**2,p.nwan**2)

chi_c_eig, _ = linalg.eigh(chi_c)

k_HSP2, chi_c_HSP = kpath_extract.kpath_extractor(p, (chi_c_eig[:,:,:,-1]))
kpath_extract.kpath_save_data(k_HSP, chi_c_HSP, p.kpath_savepath.format("chi_c_maxeig"))       
print('Chi_charge(0,k) [calc+kpath save] done', file=open(p.Logstr,'a'))  

_, gkio_HSP = kpath_extract.kpath_extractor(p, (trace(gkio[b.f_iwn_zero_ind],0,1,2)/3).reshape(p.nk1,p.nk2,p.nk3))
print('Gkio [kpath calc] done', file=open(p.Logstr,'a'))  


print(allclose(k_HSP[0],k_HSP2[0]),allclose(k_HSP[1],k_HSP2[1]),allclose(k_HSP[2],k_HSP2[2]), file=open(p.Logstr,'a'))  

k_HSP     = concatenate(k_HSP)
chi_s_HSP = concatenate(chi_s_HSP)
chi_c_HSP = concatenate(chi_c_HSP)
gkio_HSP  = concatenate(gkio_HSP)

with h5py.File(savepath,'a') as file:
    group = file.require_group('kpath')
    
    group.require_dataset('kvalue',data=k_HSP,shape=k_HSP.shape,dtype=k_HSP.dtype)
    group.require_dataset('chi_spin_max_eig',data=chi_s_HSP,shape=chi_s_HSP.shape,dtype=chi_s_HSP.dtype)
    group.require_dataset('chi_charge_max_eig',data=chi_c_HSP,shape=chi_c_HSP.shape,dtype=chi_c_HSP.dtype)        
    group.require_dataset('gkio',data=gkio_HSP,shape=gkio_HSP.shape,dtype=gkio_HSP.dtype)

print('kpath hdf5 save complete', file=open(p.Logstr,'a'))  

#BSE_kernel
X = ckio@h.S_mat
X = X.reshape(-1, p.nk1, p.nk2, p.nk3, p.nwan**2, p.nwan**2)
X_eig, _ = linalg.eigh(X)
BSEK_S = amax(X_eig)

X = ckio@h.C_mat
X = X.reshape(-1, p.nk1, p.nk2, p.nk3, p.nwan**2, p.nwan**2)
X_eig, _ = linalg.eigh(X)
BSEK_C = amin(X_eig)


with open(('BSE_kernel_n_T_data/largest_BSEK_for_' +\
                    'n_{}_JUratio_{}_U_{}.dat').format(p.n_fill,p.JU_ratio,p.u0),"a") as file:
        file.write('{} {} {}\n'.format(p.T, BSEK_S, BSEK_C))
        
with h5py.File(savepath,'a') as file:
    group = file.require_group('gfunction')
    
    group.require_dataset('BSE_kernel_spin_max' ,  data=BSEK_S, shape=(), dtype='float64')
    group.require_dataset('BSE_kernel_charge_max', data=BSEK_C, shape=(), dtype='float64')

print('BSE_kernel charge+spin [calc+save] done', file=open(p.Logstr,'a'))  
print('Finished current parameters. Move to next or quit.', file=open(p.Logstr,'a'))
print('#########################################################', file=open(p.Logstr,'a'))     