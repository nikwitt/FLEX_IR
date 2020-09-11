# -*- encoding: latin-1 -*-

import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import *
import scipy as sc
from ir_load import ir_load
from parameters import parameters
from gfunction import gfunction_calc
from gfunction import gfunction_load
from eliashberg import eliashberg
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import h5py


### Set parameters for calculation
n_vec   = array([0.7,0.75,0.8,0.82,0.84,0.85,0.86,0.88,0.9,0.95])
#n_vec   = array([0.3,0.4, 0.5, 0.6])
#T_vec   = array([0.05])
T_vec   = array([0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11,\
                  0.1, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])#, 0.01])#,\
                  #0.009,0.008,0.007,0.006,0.005,0.004,0.003])tpr_vec = array([0.0])#linspace(0.3,0.3,1)#*0
tpr_vec = array([0.0])
#tpr_vec = array([0.4,0.5,0.6])


for n_it in range(len(n_vec)):
  for tpr_it in range(len(tpr_vec)):
    round_it = 0
    for T_it in range(len(T_vec)):
      T      = T_vec[T_it]
      n_fill = n_vec[n_it]
      tpr    = tpr_vec[tpr_it]
      
        
      ### Initiate parameters -------------------------------------------------
      start = time.process_time()
      p = parameters(round(T,5), round(n_fill,5), round(tpr,5), round_it,\
                     T_load=T_vec[T_it-round_it], tpr_load=tpr_vec[tpr_it])
      print("##################################################"\
            , file=open(p.Logstr,'a'))
      print(datetime.datetime.now().strftime('%d. %B %Y %I:%M%p')\
            , file=open(p.Logstr,'a'))
      print("Parameter set: n = {}, T = {}, U = {}, t_prime = {}\n".format\
           (p.n_fill, p.T, p.u0 ,p.t_prime), file=open(p.Logstr,'a'))
      print("Elapsed time - parameter init: " + str(time.process_time() - start)\
            , file=open(p.Logstr,'a')) 

      ## Skip if load data does not exist.
      ## Prevents further calculation if previous calc did not converge.
      if round_it != 0:
          with h5py.File(p.loadpath,'r') as file: 
              if not 'gfunction' in file.keys():
                  print("File to load " + p.sp_dir + " does not exist. Skip this parameter round!"\
                        , file=open(p.Logerrstr,'a'))
                  print("File to load " + p.sp_dir + " does not exist. Skip this parameter round!"\
                        , file=open(p.Logstr,'a'))
                  continue


      ### Load irbasis --------------------------------------------------------            
      t_bload = time.process_time()
      b = ir_load(p.Lambda, p.beta, p.IR_tol)
      print("Elapsed time - basis load (tot | module): " \
            + str(time.process_time() - start) + " | " \
            + str(time.process_time() - t_bload), file=open(p.Logstr,'a')) 


      ### Calculate full Greens function---------------------------------------
      t_gcal = time.process_time()
      g = gfunction_calc(p,b)
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
              continue
          
          # Sigma convergence
          if sum(abs(g.sigma_old-g.sigma))/sum(abs(g.sigma)) > p.g_sfc_tol:
              print("Not calculating eliashberg equation.\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"\
                    , file=open(p.Logstr,'a'))
              print(" " * len(p.err_str_begin) + "=> eliashberg skipped."\
                    , file=open(p.Logerrstr,'a'))
              print("##################################################"\
                    , file=open(p.Logstr,'a'))
              print("\n",file=open(p.Logstr,'a'))
              continue
      
        
      ### Calculate SC parameter ----------------------------------------------
      t_eliashberg = time.process_time()
      el = eliashberg(g, p, b)
      print("Elapsed time - Eliashberg calc (tot | module): " \
            + str(time.process_time() - start) + " | " \
            + str(time.process_time() - t_eliashberg), file=open(p.Logstr,'a'))
      print("Done: n = {} | T/t = {} ({} K) | t_prime = {} | abs(lambda) = {}".format\
            (p.n_fill, p.T, round(p.T*1.16*10**4,2), p.t_prime, abs(el.result))\
            , file=open(p.Logstr,'a'))
            
      
      ### Save resulting lambda value -----------------------------------------
      with open(p.SC_EV_path, 'a') as file:
          file.write("{} {} {}\n".format(p.T, real(el.result), imag(el.result)))
          
      print("##################################################"\
            , file=open(p.Logstr,'a'))
      print("\n",file=open(p.Logstr,'a'))
      
      round_it = 1