# -*- encoding: latin-1 -*-

from numpy import *
from matrices3 import mat3

class parameters:
    """
    Setting parameters for current calculation.
    As no extra file is generated, one should make thorough notes on what 
    parameters are set in each calculation!
    """
    def __init__(self,T,n,JU_ratio,round_it, T_load = 0.01, JU_ratio_load = 0.0):        
        ### Calculation parameters
        self.mode = 'FLEX' #'FLEX' 'RPA'
        self.mix  = 0.2 # Value of how much of the new G is to be used!
        self.round_it = round_it
        
        self.nk1, self.nk2, self.nk3 = 180, 180, 1
        self.T       = T
        self.beta    = 1/self.T
        self.Lambda  = 10**4
        self.n_fill  = n
        self.g_sfc_tol = 1e-4
        self.DOS_n   = 150
    
        self.nspin   = 1
        self.norb    = 3
        self.nwan    = self.nspin * self.norb #spin * orbital    
    
        ### Interaction
        self.u0, self.JU_ratio, self.u0_prime, self.J, self.J_prime =\
            parameters.Uval_set(6, JU_ratio)

        ### Setting up k-mesh
        self.nk = self.nk1 *self.nk2 * self.nk3
        self.dk1, self.dk2, self.dk3 = 1./self.nk1, 1./self.nk2, 1./self.nk3
        k1, k2, k3 = meshgrid(arange(self.nk1)*self.dk1, arange(self.nk2)*self.dk2, arange(self.nk3)*self.dk3)
        #k1, k2, k3 = meshgrid(arange(-self.nk1/2,self.nk1/2)*self.dk1, arange(-self.nk2/2,self.nk2/2)*self.dk2, arange(-self.nk3/2,self.nk3/2)*self.dk3)
        self.k1, self.k2, self.k3 = k1.flatten(), k2.flatten(), k3.flatten()

        ### SC calculation options
        self.SC_type = 'f1' #s s_ext px py d f1 f2
        #self.SC_savetype = 'short' # 'short' or 'long'
        self.SC_sfc_tol = 1e-4
        
        ### Log options
        Log_name = "Log_JU_" + str(self.JU_ratio)
        self.Logstr = Log_name + ".dat"
        self.Logerrstr = Log_name + "_err.dat"
        self.err_str_begin = "System T_"+ str(self.T)+" | n="+str(self.n_fill)+\
            " | U="+str(self.u0)+" | J_H="+str(JU_ratio)+"U : "
            
        ### Setting saving options
        self.sp_dir  = "Odata_JU_" + str(self.JU_ratio) + "/"
        self.sp_dir_load  = "Odata_JU_" + str(JU_ratio_load) + "/" 
        sp_add  = ""
        self.sp_name_save = "T_"+ str(self.T) + "_U_" + str(self.u0) +\
                  "_JUratio_" + str(JU_ratio) + "_n_" + str(self.n_fill)
        if self.round_it == 0:
            self.sp_name_load = self.sp_name_save
        else:
            self.sp_name_load = "T_"+ str(round(T_load,3)) + "_U_" + str(self.u0) +\
                  "_JUratio_" + str(round(JU_ratio_load,2)) + "_n_" + str(self.n_fill)
        self.savepath = self.sp_dir + sp_add + self.sp_name_save
        self.loadpath = self.sp_dir_load + sp_add + self.sp_name_load
        self.SC_savepath = self.sp_dir + self.SC_type + "w_" + self.sp_name_save
        self.SC_loadpath = self.sp_dir_load + self.SC_type + "w_" + self.sp_name_load
        self.kpath_savepath = "Odata_kpath/kpath_" + self.sp_name_save + "_{}.dat"
        
    def Uval_set(u0, JU_ratio):
        J        = JU_ratio*u0
        u0_prime = u0 - 2*J
        J_prime  = J
        return u0, JU_ratio, u0_prime, J, J_prime
