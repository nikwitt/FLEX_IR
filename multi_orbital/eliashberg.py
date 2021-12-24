## -*- encoding: latin-1 -*-

from numpy import *
from pauli import sigma as pauli_mat
import scipy as sc
import pyfftw
import multiprocessing as mp
import h5py

##############################################################################
##### Main code: Calculate lin. eliashberg eq. within FLEX approximation #####
##############################################################################

class eliashberg:
    def __init__(self, g, p, b, h):
        self.set_v(g, p, b, h)
        self.set_delta0(g, p, b)

        lam = self.scf(g, p, b, h, 0)

        if real(lam) < 0: # or p.SC_type == 'py':
            print('Another eliashberg cycle will be performed.', file=open(p.Logstr,'a'))
            print(p.err_str_begin + "lambda < 0 : => new round!"\
               , file=open(p.Logerrstr,'a'))
                
            with open(p.SC_EV_path_neg, 'a') as file:
                file.write("{} {} {}\n".format(p.T, real(lam), imag(lam)))
            
            lam_n = self.scf(g, p, b, h, lam)
            lam = lam_n + lam
        
        self.result = lam
        
        ##### Save functions 
        print("Saving all data now...", file=open(p.Logstr,'a'))
        with h5py.File(p.savepath,'a') as file: 
            group = file.require_group('eliashberg')
            
            group.require_dataset('{}_gap'.format(p.SC_type)    , data=self.delta , shape=self.delta.shape, dtype=complex)
            group.require_dataset('{}_lambda'.format(p.SC_type) , data=self.result, shape=(), dtype=complex)
            
            group['{}_gap'.format(p.SC_type)][...]    = self.delta
            group['{}_lambda'.format(p.SC_type)][...] = self.result
            
        
    ### Set Coulomb interaction V(r, tau_fermi)--------------------------------
    def set_v(self, g, p, b, h):
        chi_spin   = linalg.inv(g.E_int - g.ckio@h.S_mat)@g.ckio
        chi_charge = linalg.inv(g.E_int + g.ckio@h.C_mat)@g.ckio
        
        # Set V according to parity/SC wave type
        if p.SC_type in {'s', 's_ext', 'dx2-y2', 'dxy'}: #singlet
            v =   3./2.* h.S_mat@chi_spin@h.S_mat \
                - 1./2.* h.C_mat@chi_charge@h.C_mat
            #self.v_DC = (h.C_mat + h.S_mat)/2
            self.v_DC = (3*h.S_mat + h.C_mat)/4
        elif p.SC_type in {'px', 'py', 'f1', 'f2'}: #triplet
            v = - 1./2.* h.S_mat@chi_spin@h.S_mat \
                - 1./2.* h.C_mat@chi_charge@h.C_mat
            #self.v_DC = (h.C_mat - h.S_mat)/2
            self.v_DC = (-h.S_mat + h.C_mat)/4

        v = v.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)     

        #FFT to (r, tau_fermi) although V is bosonic!
        fft_object = pyfftw.builders.fftn(v, axes=(1,2,3))
        v = fft_object().reshape(len(b.bm),p.nk*p.nwan**4)  
        
        v = dot(b.bose_iw_to_tau_fermi, v)
        self.v = v.reshape(len(b.ft),p.nk,p.nwan,p.nwan,p.nwan,p.nwan)



    ### Set inital gap delta0(k, iw_n) --------------------------------------
    def set_delta0(self, g, p, b):
        """
        Set initial guess for gap function according to system symmetry.
        The setup is carried out in real space and then FT.
        """

	### These are for some triangular lattice symmetry
        ### Set inital delta according to symmetry
        if p.SC_type == 's': 
            #singlet:
            delta_func = ones(p.nk)
        elif p.SC_type == 's_ext':
            #singlet: cos(kx-ky) + const or so
            #### To be implemented, next line is wrong
            delta_func = ones(p.nk) + cos(2*pi*p.k1) + cos(2*pi*p.k2)   #+ cos(2*pi*1/sqrt(3)*(2*p.k1+p.k2)) + cos(2*pi*p.k2)
        elif p.SC_type == 'px':
            #triplet: sin(k1) + sin(k1+k2)
            delta_func = sin(2*pi*p.k1) + sin(2*pi*(p.k1+p.k2))
        elif p.SC_type == 'py':
            #triplet: sin(sqrt(3)/2 kx) * (cos(3/2 ky) - cos(sqrt(3)/2 kx))
            delta_func = -sin(2*pi*p.k1) + 2*sin(2*pi*p.k2) + sin(2*pi*(p.k1+p.k2))
        elif p.SC_type == 'dx2-y2':
            #singlet: cos(k1) - cos(k2) + cos(k1+k2)
            delta_func = cos(2*pi*p.k1) - 2*cos(2*pi*p.k2) + cos(2*pi*(p.k1+p.k2))
        elif p.SC_type == 'dxy':
            #singlet: cos(k1) - cos(k2)
            delta_func = cos(2*pi*p.k1) - cos(2*pi*p.k2)
        elif p.SC_type == 'f1': # x(x²-3y²)
            #triplet: sin(      1/2 ky) * (cos(1/2 ky) - cos(sqrt(3)/2 kx))
            delta_func = sin(2*pi*p.k2/2)*(cos(2*pi*p.k2/2)-cos(2*pi*(2*p.k1+p.k2)/2))
        elif p.SC_type == 'f2': # y(3x²-y²)
            #triplet: sin(sqrt(3)/2 kx) * (cos(3/2 ky) - cos(sqrt(3)/2 kx))
            delta_func = sin(2*pi*(2*p.k1+p.k2)/2)*(cos(2*pi*3*p.k2/2)-cos(2*pi*(2*p.k1+p.k2)/2))
            
        self.delta0 = tensordot(delta_func,eye(p.nwan),axes=0)   
        #self.delta0 = h.uk@self.delta0@h.uk_adj.conj()

        self.delta  = tensordot(ones(len(b.fm)), self.delta0, axes=0).reshape(len(b.fm), p.nk, p.nwan, p.nwan)
        self.delta  = self.delta / linalg.norm(self.delta)


    ### Set anomalous GF F(r, tau_fermi) --------------------------------------
    def set_f(self, g, p, b):
        f = g.gkio@self.delta@conj(g.gkio_invk)
        f = f.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan**2)
        
        fft_object = pyfftw.builders.fftn(f, axes=(1,2,3))
        f = fft_object().reshape(len(b.fm),p.nk*p.nwan*p.nwan)
        
        self.f = dot(b.fermi_iw_to_tau, f).reshape(len(b.ft),p.nk,p.nwan,p.nwan)
        
        f = f.reshape(len(b.fm), p.nk, p.nwan**2)
        self.f_0 = dot(b.fermi_iw_to_tau_0, f[:,0]).reshape(p.nwan,p.nwan)


    ##############
    ### Self consistency loop for linearized Eliashberg equation
    ### Employs power iterative method to solve lam*delta = lam*V*F in (r,tau)-space
    ##############

    def scf(self, g, p, b, h, lam_in):
        """
        Self consistency loop for super conduction parameter via eigenvalue method.
        Implements FLEX approximation in linearized Eliashberg equation.
        Handles depending on SC-type input in p.SC_type(=parameters) the equation differently.
        """
    
        lam1 = 0.1
        lam0 = 0.0
        for n in range(100):
            if abs(lam1-lam0)/abs(lam1) <= p.SC_sfc_tol: break
            # Power iteration method for computing lambda
            lam0 = lam1
            self.set_f(g, p, b)
            
            # K*gap = y = V*F
            y = - einsum('ijkmln,ijml->ijkn', self.v, self.f)
            y = y.reshape(len(b.ft),p.nk1,p.nk2,p.nk3,p.nwan**2)
            
            fft_object = pyfftw.builders.ifftn(y, axes=(1,2,3))
            y = fft_object()/p.nk
            y = y.reshape(len(b.ft),p.nk*p.nwan**2)

            y = dot(b.fermi_tau_to_iw, y)
            y = y.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
            
            y_HF = -einsum('kmln,ml->kn',self.v_DC.reshape(p.nwan,p.nwan,p.nwan,p.nwan),self.f_0)\
                        * ones((len(b.fm),p.nk,p.nwan,p.nwan))/p.nk
            
            ### y + y_HF - lam*y (power iteration method trick)
            y = y + y_HF - real(lam_in)*self.delta

            ### Impose symmetry conditions            
            # Even function of matsubara frequency
            y = (y + y[::-1])/2 
            
            # k-space symmetry depending on singlet/triplet:
            # y_2 corresponds to y(iwn,-k)!
            y_2 = y.reshape(len(b.fm),p.nk1,p.nk2,p.nwan,p.nwan)
            y_2 = roll(y_2,-1,(1,2))[:,::-1,::-1]
            y_2 = y_2.reshape(len(b.fm),p.nk,p.nwan,p.nwan) 
            y_2 = transpose(y_2,axes=(0,1,3,2))
            if p.SC_type in {'s', 's_ext', 'dx2-y2', 'dxy'}:
                # singlet case: delta_ab(k) = delta_ba(-k)
                y = (y + y_2)/2     
            elif p.SC_type in {'px', 'py', 'f1', 'f2'}:     
                # triplet case: delta_ab(k) = - delta_ba(-k)
                y = (y - y_2)/2  

            # Subtract highest matsubara frequency
            y = y - y[0]

            ### Calculating lambda
            lam1 = sum(conj(y)*self.delta)
            
            self.delta = y/linalg.norm(y)
            print(n,lam1,linalg.norm(y), file=open(p.Logstr,'a'))
            
        return lam1
