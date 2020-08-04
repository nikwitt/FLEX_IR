# -*- encoding: latin-1 -*-

import sys

MKL_THREADS_VAR = str(sys.argv[1])

import os
os.environ["MKL_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["NUMEXPR_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import *
from ir_load import ir_load
from parameters import parameters
from hamiltonian import hamiltonian
from gfunction import gfunction_calc
from gfunction import gfunction_load
from eliashberg import eliashberg
from kpath_extract import kpath_extract
import matplotlib
import matplotlib.pyplot as plt
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
      , file=open(p.Logstr,'a'))
print(datetime.datetime.now().strftime('%d. %B %Y %I:%M%p')\
      , file=open(p.Logstr,'a'))
print("Parameter set: n = " + str(p.n_fill) + ", T = " + str(p.T)\
      + ", U = " + str(p.u0) + ", J_H = " + str(p.JU_ratio) + "U\n"\
      , file=open(p.Logstr,'a'))
print("Elapsed time - parameter init: " + str(time.process_time() - start)\
      , file=open(p.Logstr,'a')) 


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
t_gcal = time.process_time()
g = gfunction_calc(p,h,b)
#g = gfunction_load(p,b)
print("Elapsed time - g_scf_calc load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_gcal), file=open(p.Logstr,'a')) 

### Security convergence check of greens function
 
if g.tag == 'calc' and p.mode == 'FLEX':
    # U convergence
    if p.u0 != g.u0_pval:
        print("Not calculating eliashberg equation.\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"\
              , file=open(p.Logstr,'a'))
        print(" " * len(p.err_str_begin) + "=> eliashberg skipped."\
              , file=open(p.Logerrstr,'a'))
        print("##################################################"\
              , file=open(p.Logstr,'a'))
        print("\n",file=open(p.Logstr,'a'))
        #continue
          
    # Sigma convergence
    if sum(abs(g.sigma_old-g.sigma))/sum(abs(g.sigma)) > p.g_sfc_tol:
        print("Not calculating eliashberg equation.\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"\
              , file=open(p.Logstr,'a'))
        print(" " * len(p.err_str_begin) + "=> eliashberg skipped."\
              , file=open(p.Logerrstr,'a'))
        print("##################################################"\
              , file=open(p.Logstr,'a'))
        print("\n",file=open(p.Logstr,'a'))
        #continue

#BSE_kernel
X = g.ckio@h.S_mat
X = X.reshape(-1, p.nk1, p.nk2, p.nk3, p.nwan**2, p.nwan**2)
X_eig, _ = linalg.eigh(X)
BSEK_S = amax(X_eig)

X = g.ckio@h.C_mat
X = X.reshape(-1, p.nk1, p.nk2, p.nk3, p.nwan**2, p.nwan**2)
X_eig, _ = linalg.eigh(X)
BSEK_C = amin(X_eig)

with open(('BSE_kernel_n_T_data/largest_BSEK_for_' +\
                    'n_{}_JUratio_{}_U_{}.dat').format(p.n_fill,p.JU_ratio,p.u0),"a") as file:
        file.write('{} {} {}\n'.format(p.T, BSEK_S, BSEK_C))
            
      
# ### Calculate SC parameter ----------------------------------------------
t_eliashberg = time.process_time()
el = eliashberg(g, p, b, h)
print("Elapsed time - Eliashberg calc (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_eliashberg), file=open(p.Logstr,'a'))
print("Done: n = " + str(p.n_fill) + " | T/t = " + str(p.T) + \
      " (" + str(round(p.T*(1.5/8.374)*1.16*10**4,2)) + " K) | JUratio" + str(p.JU_ratio)\
      + " | |lambda| = " + str(abs(el.result)), file=open(p.Logstr,'a')) #1.5/8.374 is factor for NaCoO2 model
        
      
### Save resulting lambda value -----------------------------------------
file = open("lam_n_T_data/" + p.SC_type + "w_lam_for_n_" + str(p.n_fill)\
            + "_JUratio_" + str(p.JU_ratio) + "_U_" + str(p.u0) + ".dat","a")
file.write(str(p.T) + " " + str(real(el.result)) + " " + str(imag(el.result)) + "\n")
file.close()
       
print("Now extract kpath of GF, X_s...", file=open(p.Logstr, 'a'))
kpath_extract(p,h,b,g)
print("Done.", file=open(p.Logstr,'a'))

print("##################################################"\
      , file=open(p.Logstr,'a'))
print("\n",file=open(p.Logstr,'a'))
