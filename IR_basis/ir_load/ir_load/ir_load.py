#Function for loading irbasis on generated sampling points.

import numpy as np
import irbasis
from .ir_sampling_points import write_data
import os

class ir_load:
    def __init__(self, Lambda, beta):
        """
        Function for loading irbasis on precomputed grid points.
        Takes object 'p' as an argument that needs values p.Lambda and p.beta,
        i.e. cutoff and inverse temperature for which irbasis is needed.
        
        Ver. 1.0: only U vectors on imaginary time and Matsubara frequencies
        are supported.
        """
        
        ### Load irbasis objects from 'irbasis.py' for both statistics
        self.fermi_basis = irbasis.load('F', Lambda)
        self.bose_basis  = irbasis.load('B', Lambda)

        ### Load sampling points for both statistics
        self.f_sp = self.sp_load(Lambda, self.fermi_basis)
        self.b_sp = self.sp_load(Lambda, self.bose_basis )

        ### Generate normalized tau grid [add -beta, beta to interval]
        self.x_smpl_fermi = np.append(np.append(-1, self.f_sp[0][-1]), 1)
        self.x_smpl_bose  = np.append(np.append(-1, self.b_sp[0][-1]), 1)

        ### Generate U vectors on tau grids
        # fermi_Ulx       : fermionic U vectors on fermionic tau grid
        # fermi_Ulx_boson : fermionic U vectors on bosonic   tau grid
        # bose_Ulx        : bosonic   U vectors on bosonic   tau grid
        # bose_Ulx_fermi  : bosonic   U vectors on fermionic tau grid
        self.fermi_Ulx, self.bose_Ulx = self.Ulx_load(self.x_smpl_fermi,self.x_smpl_bose,beta)
        self.fermi_Ulx_boson, self.bose_Ulx_fermi = self.Ulx_load(self.x_smpl_bose,self.x_smpl_fermi,beta)
        
        ### Generate indexed Matsubara frequency grid
        self.iwn_smpl_fermi = self.f_sp[2][-1]
        self.iwn_smpl_bose  = self.b_sp[2][-1]        
        
        ### Generate U vectors on Matsubara frequency grid
        self.fermi_Uln = np.sqrt(beta)*self.fermi_basis.compute_unl(self.iwn_smpl_fermi)
        self.bose_Uln  = np.sqrt(beta)*self.bose_basis.compute_unl(self.iwn_smpl_bose)
        
        ### Calculate tau/Matsubara frequency grid
        self.fm = 1j*np.pi/beta*(2*self.iwn_smpl_fermi + np.ones(len(self.iwn_smpl_fermi)))
        self.bm = 1j*np.pi/beta*(2*self.iwn_smpl_bose)
        self.ft = beta/2*(self.x_smpl_fermi + np.ones(len(self.x_smpl_fermi)))
        self.bt = beta/2*(self.x_smpl_bose  + np.ones(len(self.x_smpl_bose )))
        
        ### Calculate index of lowest matsubara frequency index
        #   (B: iw_0 | F:iw_1)
        self.f_iwn_zero_ind = list(self.iwn_smpl_fermi).index(0)
        self.b_iwn_zero_ind = list(self.iwn_smpl_bose ).index(0)

    
    def sp_load(self, Lambda, b):
        """
        Loads sampling point data from files.
        If they do not exist they will be generated.
        """
        
        ### Check if data is existent
        # Generate path to files
        dirname, _ = os.path.split(os.path.abspath(__file__))
        path_str = dirname + os.path.normcase("/sampling_points/" + "{}_" + b.statistics + \
                              "_L_" + str(Lambda) + ".npy")

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
        Ulx_fermi = np.zeros((self.fermi_basis.dim(),len(fermi_x)))
        Ulx_bose  = np.zeros((self.bose_basis.dim(), len(bose_x )))
        for l in range(self.fermi_basis.dim()):
            Ulx_fermi[l] = np.sqrt(2/beta)*self.fermi_basis.ulx(l,fermi_x)
        for l in range(self.bose_basis.dim()):
            Ulx_bose[l]  = np.sqrt(2/beta)*self.bose_basis.ulx(l,bose_x)
            
        return np.swapaxes(Ulx_fermi,0,1), np.swapaxes(Ulx_bose,0,1)


##### Old code snipptes...
#self.x_smpl_fermi = np.append(np.append(-1,np.sort(np.concatenate(self.f_sp[0]))),1)
#self.x_smpl_bose  = np.append(np.append(-1,np.sort(np.concatenate(self.b_sp[0]))),1)       
#self.x_smpl_fermi = np.append(-1, irbasis.sampling_points_x(self.fermi_basis,self.fermi_basis.dim()-1))
#self.x_smpl_bose  = np.append(-1, irbasis.sampling_points_x(self.bose_basis, self.bose_basis.dim()-1))

#self.iwn_smpl_fermi = np.sort(np.concatenate(self.f_sp[2]))
#self.iwn_smpl_bose  = np.sort(np.concatenate(self.b_sp[2]))
#self.iwn_smpl_fermi = irbasis.sampling_points_matsubara(self.fermi_basis,self.bose_basis.dim()-1)
#self.iwn_smpl_bose  = irbasis.sampling_points_matsubara(self.bose_basis, self.bose_basis.dim()-1)