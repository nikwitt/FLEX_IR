# -*- encoding: latin-1 -*-

import sys

MKL_THREADS_VAR = str(sys.argv[1])

import os
os.environ["MKL_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["NUMEXPR_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import *
import pyfftw
from einsum2 import einsum2
import scipy as sc
from ir_load import ir_load
from parameters import parameters
from hamiltonian import hamiltonian
from gfunction import gfunction_load
import datetime
import time

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
      , file=open('Data_save_calc_speedtest.dat','a'))
print(datetime.datetime.now().strftime('%d. %B %Y %I:%M%p')\
      , file=open('Data_save_calc_speedtest.dat','a'))
print("Parameter set: n = " + str(p.n_fill) + ", T = " + str(p.T)\
      + ", U = " + str(p.u0) + ", J_H = " + str(p.JU_ratio) + "U\n"\
      , file=open('Data_save_calc_speedtest.dat','a'))
print("Elapsed time - parameter init: " + str(time.process_time() - start)\
      , file=open('Data_save_calc_speedtest.dat','a'))
    
    
### Load hamiltionian----------------------------------------------------            
t_hset = time.process_time()
h = hamiltonian(p)
print("Elapsed time - hamiltonian set (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_hset), file=open('Data_save_calc_speedtest.dat','a'))      
      
      
### Load irbasis --------------------------------------------------------            
t_bload = time.process_time()
b = ir_load(p.Lambda, p.beta)
print("Elapsed time - basis load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_bload), file=open('Data_save_calc_speedtest.dat','a')) 
    
    
#### Now load greens function
dum = gfunction_load.func_load(p, "_gkio", 2)   
gkio = dum.reshape(len(dum)//(p.nwan**2),p.nwan,p.nwan,order='F')
gkio = gkio.reshape(size(gkio)//(p.nk*p.nwan**2),p.nk,p.nwan,p.nwan)
gkio = transpose(gkio, axes=(0,1,3,2))
print('G(io,k) load done', file=open('Data_save_calc_speedtest.dat','a'))

### Calc inverse greens function
t_ginvcalc = time.process_time()
gkio_invk  = gkio.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan**2)
fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
gkio_invk  = fft_object()
fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
gkio_invk  = fft_object()/p.nk
gkio_invk_calc = gkio_invk.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
print(('G(io,-k) calc time: {}').format(str(time.process_time()-t_ginvcalc)),\
      file=open('Data_save_calc_speedtest.dat','a'))
    
### Calc transpose of greens function
gkio_trans = transpose(gkio,axes=(0,1,3,2))
    
# =============================================================================
# print(amax(abs(gkio_trans-gkio_invk_calc)))
# print(amax(abs(gkio_invk_calc-gkio_trans)))
# print(sum(abs(gkio_trans-gkio_invk_calc)), sum(abs(gkio_trans-gkio_invk_calc))/size(gkio_trans))
# print(('Test G_inv_calc == G_transpose (-4): {} ').format(allclose(gkio_trans,gkio_invk_calc,atol=10**(-4))),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# print(('Test G_inv_calc == G_transpose (-6): {} ').format(allclose(gkio_invk_calc,gkio_trans,atol=10**(-6))),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# print(('Test G_inv_calc == G_transpose (-8): {} ').format(allclose(gkio_invk_calc,gkio_trans,atol=10**(-8))),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# print(('Test G_inv_calc == G_transpose (-10): {} ').format(allclose(gkio_invk_calc,gkio_trans,atol=10**(-10))),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# print(('Test G_inv_calc == G_transpose (-12): {} ').format(allclose(gkio_invk_calc,gkio_trans,atol=10**(-12))),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# =============================================================================
    
# =============================================================================
# ### Load susceptibility
# t_ckioload = time.process_time()
# dum = gfunction_load.func_load(p, "_chi", 4)
# ckio = dum.reshape(len(dum)//(p.nwan**4),p.nwan**2,p.nwan**2,order='F')
# ckio = ckio.reshape(size(ckio)//(p.nk*p.nwan**4),p.nk,p.nwan**2,p.nwan**2)
# ckio_load = transpose(ckio,axes=(0,1,3,2))   
# print(('Chi(io,k) load time: {}').format(str(time.process_time()-t_ckioload)),\
#        file=open('Data_save_calc_speedtest.dat','a')) 
# 
# ### Calc susceptibility
# t_ckiocalc = time.process_time()
# grit = gkio.reshape(len(b.fm), p.nk1, p.nk2, p.nk3, p.nwan**2)
# fft_object = pyfftw.builders.fftn(grit, axes=(1,2,3))
# grit = fft_object()
# grit = grit.reshape(len(b.fm),p.nk*p.nwan*p.nwan)
# result, _, _, _  = sc.linalg.lstsq(b.fermi_Uln, grit, lapack_driver='gelsy')
# grit_b = dot(b.fermi_Ulx_boson, result).reshape(len(b.bt),p.nk,p.nwan,p.nwan)
# 
# grit_rev = grit_b[::-1,:,:,:]    #G_lm(r,beta-tau)
# ckio = einsum2('ijkm,ijln->ijklmn', grit_b, grit_rev).reshape(len(b.bt),p.nk*p.nwan**4)#km ln
# result, _, _, _  = sc.linalg.lstsq(b.bose_Ulx, ckio, lapack_driver='gelsy')
# ckio = dot(b.bose_Uln, result)
# ckio = ckio.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)
# fft_object = pyfftw.builders.ifftn(ckio, axes=(1,2,3))
# ckio = fft_object()/p.nk    
# ckio_calc = ckio.reshape(len(b.bm),p.nk,p.nwan**2,p.nwan**2)
# print(('Chi(io,k) calc time: {}').format(str(time.process_time()-t_ckiocalc)),\
#        file=open('Data_save_calc_speedtest.dat','a')) 
#     
# print(('Chi_load == Chi_calc: {}').format(allclose(ckio_load,ckio_calc)),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# =============================================================================
    

    
# =============================================================================
# ### Load inverse greens function
# t_ginvload = time.process_time()
# dum = gfunction_load.func_load(p, "_gkio_invk", 2)   
# gkio_invk = dum.reshape(len(dum)//(p.nwan**2),p.nwan,p.nwan,order='F')
# gkio_invk = gkio_invk.reshape(size(gkio_invk)//(p.nk*p.nwan**2),p.nk,p.nwan,p.nwan)
# gkio_invk_load = transpose(gkio_invk, axes=(0,1,3,2))
# print(('G(io,-k) load time: {}').format(str(time.process_time()-t_ginvload)),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# 
# ### Calc inverse greens function
# t_ginvcalc = time.process_time()
# gkio_invk  = gkio.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan**2)
# fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
# gkio_invk  = fft_object()
# fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
# gkio_invk  = fft_object()/p.nk
# gkio_invk_calc = gkio_invk.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
# print(('G(io,-k) calc time: {}').format(str(time.process_time()-t_ginvcalc)),\
#       file=open('Data_save_calc_speedtest.dat','a'))
#     
# ### Compare greens function
# print(('G_inv_load == G_inv_calc: {} | G_inv_load == G_transpose ').format(allclose(gkio_invk_load,gkio_invk_calc),allclose(gkio_invk_calc,transpose(gkio,axes=(0,1,3,2)))),\
#       file=open('Data_save_calc_speedtest.dat','a'))
# =============================================================================
    
    
print('Finished!',file=open('Data_save_calc_speedtest.dat','a'))