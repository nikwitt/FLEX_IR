# -*- encoding: latin-1 -*-

import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import *
from ir_load import ir_load
from parameters import parameters
from hamiltonian import hamiltonian
from gfunction import gfunction_calc
from gfunction import gfunction_load
from eliashberg import eliashberg
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import sys
from Hexagonal_BZ_quantitiy_plot import Hexagonal_BZ_plot
import cProfile

class DOS_calc:
    def __init__(self):
        pass
    
    def calculation(p, H, delta):
        energy  = linspace(-4*p.t-1,4*p.t+1, p.DOS_n)
        E_      = tensordot(ones(len(energy)), array([eye(h.hk.shape[1],h.hk.shape[2]) for it in range(h.hk.shape[0])]), axes=0)
        h_      = tensordot(ones(len(energy)),h.hk,axes=0)
        energy_ = energy.reshape(len(energy),1,1,1)*E_
        DOS = - 2*pi/p.nk * imag( sum(E_@linalg.inv(energy_ - h_ - (h.mu - 1j*0.05)*E_) , axis=1) )
        
        return DOS

#n_vec  = linspace(1,1,1)*1.857
n_vec  = linspace(1,1,1)*(7-3.43)/2
T_vec  = linspace(0.002,0.001,2)
JU_vec = array([0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27])#linspace(0.30,0.30,1)
#JU_vec = linspace(0.15,0.15,1)

round_it = 1

for JU_it in JU_vec:
  for n_it in n_vec:
    for T_it in T_vec:
      ### Initiate parameters -------------------------------------------------
      start = time.process_time()
      p = parameters(round(T_it,5), round(n_it,5), round(JU_it,5), round_it)


      ### Load hamiltionian----------------------------------------------------            
      t_hset = time.process_time()
      h = hamiltonian(p)  
      
      
      ### Load irbasis --------------------------------------------------------            
      t_bload = time.process_time()
      b = ir_load(p)

       
      ### Load SC parameter ----------------------------------------------

      dum = gfunction_load.func_load(p, "_gap_", 2, sc_state='sc')   
      gap = dum.reshape(len(dum)//(p.nwan**2),p.nwan,p.nwan,order='F')
      gap = gap.reshape(size(gap)//(p.nk*p.nwan**2),p.nk,p.nwan,p.nwan)
      gap = transpose(gap, axes=(0,1,3,2))
      
      ### Plot gapfunction
      if p.SC_type == 's':
          Hexagonal_BZ_plot(p, trace(gap[b.f_iwn_zero_ind],0,1,2)/3,\
                            title=('$\\Delta_{s}$ for $T$ = ' + str(round(p.T,3)) +\
                                   ' | $J/U$ = ' + str(round(JU_it,2))),\
                            save_name=("gap_functions/Delta_"+p.SC_type+"_wave_T_"\
                                       + str(round(p.T,3)) + "_JoverU_" +\
                                       str(round(JU_it,2))+".png"), levels=15)   
      
      elif p.SC_type == 's_ext':
          Hexagonal_BZ_plot(p, trace(gap[b.f_iwn_zero_ind],0,1,2)/3,\
                            title=('$\\Delta_{s_{ext}}$ for $T$ = ' + str(round(p.T,3)) +\
                                   ' | $J/U$ = ' + str(round(JU_it,2))),\
                            save_name=("gap_functions/Delta_"+p.SC_type+"_wave_T_"\
                                       + str(round(p.T,3)) + "_JoverU_" +\
                                       str(round(JU_it,2))+".png"), levels=15)       
      
      elif p.SC_type == 'py':
          Hexagonal_BZ_plot(p, trace(gap[b.f_iwn_zero_ind],0,1,2)/3,\
                            title=('$\\Delta_{p_y}$ for $T$ = ' + str(round(p.T,3)) +\
                                   ' | $J/U$ = ' + str(round(JU_it,2))),\
                            save_name=("gap_functions/Delta_"+p.SC_type+"_wave_T_"\
                                       + str(round(p.T,3)) + "_JoverU_" +\
                                       str(round(JU_it,2))+".png"), levels=15)      
      
      elif p.SC_type == 'd':
          Hexagonal_BZ_plot(p, trace(gap[b.f_iwn_zero_ind],0,1,2)/3,\
                            title=('$\\Delta_{d_{x^2-y^2}}$ for $T$ = ' + str(round(p.T,3)) +\
                                   ' | $J/U$ = ' + str(round(JU_it,2))),\
                            save_name=("gap_functions/Delta_"+p.SC_type+"_wave_T_"\
                                       + str(round(p.T,3)) + "_JoverU_" +\
                                       str(round(JU_it,2))+".png"), levels=15)         
        
      elif p.SC_type == 'f1':
          Hexagonal_BZ_plot(p, trace(gap[b.f_iwn_zero_ind],0,1,2)/3,\
                            title=('$\\Delta_{f_1}$ for $T$ = ' + str(round(p.T,3)) +\
                                   ' | $J/U$ = ' + str(round(JU_it,2))),\
                              save_name=("gap_functions/Delta_"+p.SC_type+"_wave_T_"\
                                       + str(round(p.T,3)) + "_JoverU_" +\
                                       str(round(JU_it,2))+".png"), levels=15)     

      elif p.SC_type == 'f2':
          Hexagonal_BZ_plot(p, trace(gap[b.f_iwn_zero_ind],0,1,2)/3,\
                            title=('$\\Delta_{f_2}$ for $T$ = ' + str(round(p.T,3)) +\
                                   ' | $J/U$ = ' + str(round(JU_it,2))),\
                            save_name=("gap_functions/Delta_"+p.SC_type+"_wave_T_"\
                                       + str(round(p.T,3)) + "_JoverU_" +\
                                       str(round(JU_it,2))+".png"), levels=15)           