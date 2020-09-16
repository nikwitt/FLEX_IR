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

##### Please input in order: 
# MKL_NUM_THREADS | T | T_load | JUratio | JU_ratio_load | round_it
n_fill = 5.7/3
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
print("Extract kpath of gkio,chi_s,chi_c:"\
      , file=open(p.Logstr,'a'))
print(datetime.datetime.now().strftime('%d. %B %Y %I:%M%p')\
      , file=open(p.Logstr,'a'))
print("Parameter set: n = {}, T = {}, U = {}, J/U = {}\n".format\
     (p.n_fill, p.T, p.u0 ,p.JU_ratio), file=open(p.Logstr,'a'))
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
b = ir_load(p.Lambda, p.beta, p.IR_tol)
print("Elapsed time - basis load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_bload), file=open(p.Logstr,'a')) 


### Calculate full Greens function---------------------------------------
t_gcal = time.process_time()
print("Load gkio data.", file=open(p.Logstr,'a')) 
g = gfunction_load(p, b)
print("Elapsed time - data load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_gcal), file=open(p.Logstr,'a')) 

### Use kpath extract for G, chi_s,  chi_x ------------------------------
print("Calculate and save kpath.", file=open(p.Logstr,'a')) 
kpath_extract(p, h, b, g)    
    
print('Finished current parameters. Move to next or quit.', file=open(p.Logstr,'a'))
print('#########################################################', file=open(p.Logstr,'a'))     
