#Function for loading irbasis on generated sampling points.

import numpy as np
import irbasis
from .ir_sampling_points import write_data
import os

class ir_load:
    def __init__(self, Lambda, beta, delta):
        """
        Function for loading irbasis on precomputed grid points.
        Takes 'Lambda', 'beta' and 'delta' as input,
        i.e. irbasis parameter, inverse temperature and basis cutoff error.

        Ver. 1.2: Introduced transformation matrices iw <-> tau.
                  Added tau=0 values explicitly for filling evaluation.
	Ver. 1.1: Changed storage of sampling points.
                  Added support of different basis sizes
        Ver. 1.0: Only U vectors on imaginary time and Matsubara frequencies
                  are supported.
        """
        
        ### Load irbasis objects from 'irbasis.py' for both statistics
        self.fermi_basis = irbasis.load('F', Lambda)
        self.bose_basis  = irbasis.load('B', Lambda)

        ### Basis size for given cutoff?
        self.f_N_IR = ir_load.basis_cutoff(self.fermi_basis, delta)
        self.b_N_IR = ir_load.basis_cutoff(self.bose_basis,  delta)
        self.f_l = np.arange(self.f_N_IR)
        self.b_l = np.arange(self.b_N_IR)

        ### Load sampling points for both statistics
        self.f_sp = self.sp_load(Lambda, self.fermi_basis)
        self.b_sp = self.sp_load(Lambda, self.bose_basis )


        #--------------------------------------------------------------
        ### Generate normalized tau grid [add -beta, beta to interval]
        self.x_smpl_fermi = self.f_sp[0][self.f_N_IR-1]
        self.x_smpl_bose  = self.b_sp[0][self.b_N_IR-1]

        ### Generate U functions on tau grids
        # fermi_Ulx       : fermionic U vectors on fermionic tau grid
        # fermi_Ulx_boson : fermionic U vectors on bosonic   tau grid
        # bose_Ulx        : bosonic   U vectors on bosonic   tau grid
        # bose_Ulx_fermi  : bosonic   U vectors on fermionic tau grid
        # Matrix dimension: U(x, l)
        self.fermi_Ulx, self.bose_Ulx =\
            self.Ulx_load(self.x_smpl_fermi,self.x_smpl_bose,beta)
        self.fermi_Ulx_boson, self.bose_Ulx_fermi =\
            self.Ulx_load(self.x_smpl_bose,self.x_smpl_fermi,beta)
            
        ### Generate U functions at tau = 0 
        self.fermi_Ulx_0, self.bose_Ulx_0 =\
            self.Ulx_load(np.array([-1]),np.array([-1]),beta)
            
            
        #--------------------------------------------------------------
        ### Generate indexed Matsubara frequency grid
        self.iwn_smpl_fermi = self.f_sp[2][self.f_N_IR-1]
        self.iwn_smpl_bose  = self.b_sp[2][self.b_N_IR-1]       
        
        
        ### Generate U functions on Matsubara frequency grid
        self.fermi_Uln = np.sqrt(beta)*\
            self.fermi_basis.compute_unl(self.iwn_smpl_fermi,self.f_l[None,:])
        self.bose_Uln  = np.sqrt(beta)*\
             self.bose_basis.compute_unl(self.iwn_smpl_bose, self.b_l[None,:])
        
        
        #--------------------------------------------------------------
        ### Calculate inverse matrices of U vectors        
        self.fermi_Ulx_inv = np.linalg.inv(self.fermi_Ulx)
        self.fermi_Uln_inv = np.linalg.inv(self.fermi_Uln)
        self.bose_Ulx_inv  = np.linalg.inv(self.bose_Ulx )
        self.bose_Uln_inv  = np.linalg.inv(self.bose_Uln )


        #--------------------------------------------------------------
        ### Calculate iw <-> tau matrices      
        self.fermi_iw_to_tau       = np.dot(self.fermi_Ulx       , self.fermi_Uln_inv)
        self.fermi_iw_to_tau_boson = np.dot(self.fermi_Ulx_boson , self.fermi_Uln_inv)
        self.fermi_iw_to_tau_0     = np.dot(self.fermi_Ulx_0[0]  , self.fermi_Uln_inv)
        self.fermi_tau_to_iw       = np.dot(self.fermi_Uln       , self.fermi_Ulx_inv)
        
        self.bose_iw_to_tau        = np.dot(self.bose_Ulx , self.bose_Uln_inv)
        self.bose_iw_to_tau_fermi  = np.dot(self.bose_Ulx_fermi , self.bose_Uln_inv)
        self.bose_tau_to_iw        = np.dot(self.bose_Uln , self.bose_Ulx_inv)

        
        #--------------------------------------------------------------
        ### Calculate tau/Matsubara frequency grid
        self.fm = 1j*np.pi/beta*(2*self.iwn_smpl_fermi + np.ones(len(self.iwn_smpl_fermi)))
        self.bm = 1j*np.pi/beta*(2*self.iwn_smpl_bose)
        self.ft = beta/2*(self.x_smpl_fermi + np.ones(len(self.x_smpl_fermi)))
        self.bt = beta/2*(self.x_smpl_bose  + np.ones(len(self.x_smpl_bose )))
        
        ### Calculate index of lowest matsubara frequency index
        #   (B: iw_0 | F:iw_1)
        self.f_iwn_zero_ind = list(self.iwn_smpl_fermi).index(0)
        self.b_iwn_zero_ind = list(self.iwn_smpl_bose ).index(0)

  
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
        
        
    def sp_load(self, Lambda, b):
        """
        Loads sampling point data from files.
        If they do not exist they will be generated.
        """
        
        ### Check if data is existent
        # Generate path to files
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path_str = dirname + os.path.normcase("/sampling_points/") + "{}_" + b.statistics + \
                              "_L_{}.npy".format(int(Lambda))

        if not os.path.isfile(path_str.format("matsubara_smpl")):
            ### Save sampling points to file for future calculations
            write_data(path_str, b)
        
        ### Load sampling point data
        x_smpl   = np.load(path_str.format("x_smpl"), allow_pickle=True)
        y_smpl   = np.load(path_str.format("y_smpl"), allow_pickle=True)
        mat_smpl = np.load(path_str.format("matsubara_smpl"), allow_pickle=True)
        return x_smpl, y_smpl, mat_smpl
        
    
    def Ulx_load(self, fermi_x, bose_x, beta):
        """
        Load U vectors for given normalized tau grid.
        """

        Ulx_fermi = np.sqrt(2/beta)*self.fermi_basis.ulx(self.f_l[None,:],fermi_x[:,None])
        Ulx_bose  = np.sqrt(2/beta)* self.bose_basis.ulx(self.b_l[None,:], bose_x[:,None])
        
        return Ulx_fermi, Ulx_bose
    
    
    def basis_cutoff(b, delta):
        """
        Parameters
        ----------
        b : IR basis object
        delta : Cutoff for singular values

        Returns
        -------
        ideal basis size for given cutoff

        """
        
        ### Find smallest Nl for s_l/s_0 < delta
        delta_compare = b.sl()/b.sl(0)        
        N_IR = next((i for i, j in enumerate(delta_compare) if j<delta), None)
        
        # Include exception: no Nl found in above's 
        if not max(delta_compare < delta):
            N_IR = b.dim()
        
        ### Find optimal Nl depending on statistics
        # Fermions -> even
        # Bosons   -> odd
        if b.statistics == 'F' and N_IR % 2 == 1:
            N_IR += 1
        elif b.statistics == 'B' and N_IR % 2 == 0:
            N_IR += 1
            
        # Safety net if N_IR > basis size
        if N_IR > b.dim():
            N_IR -= 2
    
        return N_IR
