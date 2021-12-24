# -*- encoding: latin-1 -*-

import numpy as np
import h5py
import os

class parameters:
    """
    Setting parameters for current calculation.
    As no extra file is generated, one should make thorough notes on what 
    parameters are set in each calculation!
    """
    def __init__(self,T, n, tpr, round_it, T_load = 0.1, tpr_load = 0.0):        
        ### Calculation parameters
        # General settings
        self.mode = 'FLEX'
        self.mix  = 0.2 # Value of how much of the new G is to be used!
        self.round_it = round_it
        
        # SC calculation options
        self.SC_type = 'd' #'s' or 'p' or 'd'
        
        # Cutoffs/accuracy
        self.IR_tol     = 1e-15
        self.g_sfc_tol  = 1e-5
        self.SC_sfc_tol = 1e-4
        
        # Physical quantities
        self.nk1, self.nk2, self.nk3 = 64, 64, 1
        self.T       = T
        self.beta    = 1/self.T
        self.Lambda  = 10**4
        self.n_fill  = n
        self.nspin   = 1
        self.norb    = 1        
        self.nwan    = self.nspin * self.norb #spin * orbital
        
        # Interaction (for t=1)
        self.u0      = 4


        ### Setting up k-mesh
        self.nk = self.nk1 *self.nk2 * self.nk3
        self.dk1, self.dk2, self.dk3 = 1./self.nk1, 1./self.nk2, 1./self.nk3
        k1, k2, k3 = np.meshgrid(np.arange(self.nk1)*self.dk1,\
                                 np.arange(self.nk2)*self.dk2,\
                                 np.arange(self.nk3)*self.dk3)
        self.k1, self.k2, self.k3 = k1.flatten(), k2.flatten(), k3.flatten()


        ### Energy dispersion (mutli_orb -> hamiltonian.py)
        self.t       = 1
        self.t_prime = tpr*self.t
        
        self.ek  = 2*self.t * (np.cos(2*np.pi*self.k1) + np.cos(2*np.pi*self.k2)) \
            + 4*self.t_prime * np.cos(2*np.pi*self.k1) * np.cos(2*np.pi*self.k2)

        
        ### Setting Log options
        Log_name = 'Log_n_{}'.format(self.n_fill)
        self.Logstr = Log_name + ".dat"
        self.Logerrstr = Log_name + "_err.dat"
        self.err_str_begin = ("System T = {} | n = {} | U = {} | tpr = {} : ").format(self.T,self.n_fill,self.u0,self.t_prime)
        
        ### Seting saving options
        self.sp_dir  = "Odata_n_{}".format(self.n_fill) + "/"
        self.sp_name = "calculation_data_T_{}_U_{}_tpr_{}_n_{}.h5"
        
        #formatting middle string
        self.sp_name_save = self.sp_name.format(self.T,self.u0,self.t_prime,self.n_fill)
        self.sp_name_load = self.sp_name.format(T_load,self.u0,tpr_load,self.n_fill)
        
        #generating full string 
        self.savepath = self.sp_dir + self.sp_name_save
        self.loadpath = self.sp_dir + self.sp_name_load
        
        #eigenvalue strings
        self.BSE_EV_path = "BSE_kernel_EV/max_spin_charge_ev_n_{}_tpr_{}_U_{}.dat".format(self.n_fill,self.t_prime,self.u0)
        self.SC_EV_path  = "SC_EV/{}w_lam_n_{}_tpr_{}_U_{}.dat".format(self.SC_type,self.n_fill,self.t_prime,self.u0)
        self.SC_EV_path_neg = "SC_EV/{}w_lam_n_{}_tpr_{}_U_{}.dat".format(self.SC_type,self.n_fill,self.t_prime,str(self.u0)+"_first_negative")

        
        #generate hdf5 file if it does not exist
        os.makedirs("SC_EV",         exist_ok=True)  
        os.makedirs("BSE_kernel_EV", exist_ok=True)
        os.makedirs(self.sp_dir,     exist_ok=True)
        
        if not os.path.exists(self.savepath):
            with h5py.File(self.savepath,'w') as file: 
                metadata = {'System name' : 'Hubbard Square lattice',
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
                            't_prime'     : self.t_prime,}
                
                file.attrs.update(metadata)
