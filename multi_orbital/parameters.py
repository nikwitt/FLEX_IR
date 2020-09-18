# -*- encoding: latin-1 -*-

from numpy import *
from matrices3 import mat3
import os
import h5py

class parameters:
    """
    Setting parameters for current calculation.
    As no extra file is generated, one should make thorough notes on what 
    parameters are set in each calculation!
    """
    def __init__(self,T,n,JU_ratio,round_it, T_load = 0.01, JU_ratio_load = 0.0):        
        ### Calculation parameters
        # General settings
        self.mode = 'FLEX' #'FLEX' 'RPA'
        self.mix  = 0.2 # Value of how much of the new G is to be used!
        self.round_it = round_it
        
        # SC calculation options
        self.SC_type = 'f1' #s s_ext px py d f1 f2
        
        # Cutoffs/accuracy
        self.IR_tol     = 1e-15
        self.g_sfc_tol  = 1e-4
        self.SC_sfc_tol = 1e-4
        
        # Physical quantities
        self.nk1, self.nk2, self.nk3 = 180, 180, 1
        self.T       = T
        self.beta    = 1/self.T
        self.Lambda  = 10**4
        self.n_fill  = n
        self.nspin   = 1
        self.norb    = 3
        self.nwan    = self.nspin * self.norb #spin * orbital   
        
        # Interaction
        self.u0, self.JU_ratio, self.u0_prime, self.J, self.J_prime =\
            parameters.Uval_set(6, JU_ratio)


        ### Setting up k-mesh
        self.nk = self.nk1 *self.nk2 * self.nk3
        self.dk1, self.dk2, self.dk3 = 1./self.nk1, 1./self.nk2, 1./self.nk3
        k1, k2, k3 = meshgrid(arange(self.nk1)*self.dk1,\
                              arange(self.nk2)*self.dk2,\
                              arange(self.nk3)*self.dk3)
        self.k1, self.k2, self.k3 = k1.flatten(), k2.flatten(), k3.flatten()
        
        
        ### Log options
        Log_name = "Log_JU_{}".format(self.JU_ratio)
        self.Logstr = Log_name + ".dat"
        self.Logerrstr = Log_name + "_err.dat"
        self.err_str_begin = "System T = {} | n = {} | U = {} | J/U = {} : ".format(self.T,self.n_fill,self.u0,self.JU_ratio)
           

        ### Setting saving options
        self.sp_dir    = "Odata_JU_{}/"
        self.data_name = "NaxCoO2_calculation_data_T_{}_U_{}_JUratio_{}_n_{}.h5"
        self.calc_name = self.sp_dir + self.data_name

        #formatting middle string
        self.savepath = self.calc_name.format(self.JU_ratio,self.T,self.u0,self.JU_ratio,self.n_fill)
        self.loadpath = self.calc_name.format(JU_ratio_load,T_load,self.u0,JU_ratio_load,self.n_fill)
        
        #eigenvalue strings
        self.BSE_EV_path = "BSE_kernel_EV/max_spin_charge_ev_n_{}_JUratio_{}_U_{}.dat".format(self.n_fill,self.JU_ratio,self.u0)
        self.SC_EV_path  = "SC_EV/{}w_lam_n_{}_JUratio_{}_U_{}.dat".format(self.SC_type,self.n_fill,self.JU_ratio,self.u0)
        self.SC_EV_path_neg = "SC_EV/{}w_lam_n_{}_JUratio_{}_U_{}.dat".format(self.SC_type,self.n_fill,self.JU_ratio,str(self.u0)+"_first_negative")

        ### Generate directories/hdf5 file if not exist
        os.makedirs("SC_EV",         exist_ok=True)  
        os.makedirs("BSE_kernel_EV", exist_ok=True)
        os.makedirs(self.sp_dir.format(self.JU_ratio), exist_ok=True)
        
        if not os.path.exists(self.savepath):
            with h5py.File(self.savepath,'w') as file: 
                metadata = {'System name' : 'Na_xCoO2.yH20',
                            'N_k1'        : self.nk1,
                            'N_k2'        : self.nk2,
                            'N_k3'        : self.nk3,
                            'Lambda_IR'   : self.Lambda,
                            'IR_tol'      : self.IR_tol,
                            'g_sfc_tol'   : self.g_sfc_tol,
                            'SC_sfc_tol'  : self.SC_sfc_tol,
                            'n_fill'      : self.n_fill,
                            'T'           : self.T,
                            'U'           : self.u0,
                            'JU_ratio'    : self.JU_ratio,}
                
                file.attrs.update(metadata)


     
    def Uval_set(u0, JU_ratio):
        J        = JU_ratio*u0
        u0_prime = u0 - 2*J
        J_prime  = J
        return u0, JU_ratio, u0_prime, J, J_prime
