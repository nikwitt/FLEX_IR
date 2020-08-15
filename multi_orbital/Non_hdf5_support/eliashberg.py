## -*- encoding: latin-1 -*-

from numpy import *
from pauli import sigma as pauli_mat
import scipy as sc
import pyfftw
import multiprocessing as mp

# Function for opening and writing data
def open_and_write(path, quantity):
        
    q_shape = quantity.shape
    file = open(path, "w")
        
    for it0 in range(q_shape[0]):             #tau or iwn
        for it1 in range(q_shape[1]):         #k1
            for it2 in range(q_shape[2]):     #k2
                file.write(str(it0) + " " + str(it1) + " " + str(it2) +\
                    " " + str(real(quantity[it0][it1][it2][0])) +\
                    " " + str(imag(quantity[it0][it1][it2][0])) + "\n")
            file.write("\n")
        file.write("\n")
    file.close()

def open_and_write2(path, quantity):
        
    q_shape = quantity.shape
    file = open(path, "w")
        
    for it0 in range(q_shape[0]):          #k1
        for it1 in range(q_shape[1]):      #k2
            file.write(str(it0) + " " + str(it1) +\
                " " + str(real(quantity[it0][it1][0])) +\
                " " + str(imag(quantity[it0][it1][0])) + "\n")
        file.write("\n")
    file.close()

##############################################################################
##### Main code: Calculate lin. eliashberg eq. within FLEX approximation #####
##############################################################################


class eliashberg:
    def __init__(self, g, p, b, h):
        self.set_v(g, p, b, h)
        self.set_delta0(g, p, b)

        lam = self.scf(g, p, b, h, 0)

        if real(lam) < 0 and p.SC_type != 's':
            print('Not s-wave type but found negative lambda. Another cycle will be performed.', file=open(p.Logstr,'a'))
            print(p.err_str_begin + "lambda < 0 : => new round!"\
               , file=open(p.Logerrstr,'a'))
            lam_n = self.scf(g, p, b, h, lam)
            lam = lam_n + lam
        if p.SC_type in {'px','py'}:
            print('p-wave converged to f1 (probably!). Another cycle will be performed.', file=open(p.Logstr,'a'))
            lam_n = self.scf(g, p, b, h, lam)
            print(p.err_str_begin + "p-wave change: " + str(real(lam)) + " to " + str(real(lam+lam_n))\
               , file=open(p.Logerrstr,'a'))
            lam = lam_n + lam   
        
        self.result = lam
        
        ##### Save functions 
# =============================================================================
#         # Write F(r, tau_fermi)
#         self.func_write(p, self.f.reshape(len(b.ft),p.nk1,p.nk2,p.nk3,p.nwan,p.nwan),\
#                 "_frt_")     
# =============================================================================
        print("Save gap function...", file=open(p.Logstr,'a'))     
        # Write Gap(k, iw_n)
        self.func_write(p, self.delta.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan,p.nwan),\
                "_gap_") 
        # only at iw_1 and trace
        self.func_write(p, (trace(self.delta[b.f_iwn_zero_ind],0,1,2)/3).reshape(p.nk1,p.nk2,p.nk3),\
                "_gap_") 
        
# =============================================================================
#         # Write V(k,iwn)
#         self.func_write(p, self.v.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan,p.nwan,p.nwan,p.nwan),\
#                 "_v_") 
# =============================================================================
        
        
    ### Set Coulomb interaction V(r, tau_fermi)--------------------------------
    def set_v(self, g, p, b, h):
        chi_spin   = g.ckio@linalg.inv(g.E_int - h.S_mat@g.ckio)
        chi_charge = g.ckio@linalg.inv(g.E_int + h.C_mat@g.ckio)
        
        # Set V according to parity/SC wave type
        if p.SC_type in {'s', 's_ext', 'd'}: #singulett
            #Complete singlet: v = p.u0*E + 3./2.*p.u0*p.u0*spin - 1./2.*p.u0*p.u0*charge    
            v =   3./2.* h.S_mat@chi_spin@h.S_mat \
                - 1./2.* h.C_mat@chi_charge@h.C_mat
            self.v_DC = (h.C_mat + h.S_mat)/2
        elif p.SC_type in {'px', 'py', 'f1', 'f2'}: #triplet
            v = - 1./2.* h.S_mat@chi_spin@h.S_mat \
                - 1./2.* h.C_mat@chi_charge@h.C_mat
            self.v_DC = (h.C_mat - h.S_mat)/2

        v = v.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)     

        #FFT to (r, tau_fermi) although V is bosonic!
        fft_object = pyfftw.builders.fftn(v, axes=(1,2,3))
        v = fft_object().reshape(len(b.bm),p.nk*p.nwan**4)  
        result, _, _, _ = sc.linalg.lstsq(b.bose_Uln, v, lapack_driver='gelsy')
        v = dot(b.bose_Ulx_fermi, result)
        self.v = v.reshape(len(b.ft),p.nk,p.nwan,p.nwan,p.nwan,p.nwan)



    ### Set inital gap delta0(k, iw_n) --------------------------------------
    def set_delta0(self, g, p, b):
        """
        Set initial guess for gap function according to system symmetry.
        The setup is carried out in real space and then FT.
        """

        ### Set inital delta according to symmetry
        if p.SC_type == 's': 
            #singlet:
            delta_func = ones(p.nk)
        elif p.SC_type == 's_ext':
            #singlet: cos(kx-ky) + const or so
            #### To be implemented
            delta_func = ones(p.nk) + cos(2*pi*p.k1) + cos(2*pi*p.k2)   #+ cos(2*pi*1/sqrt(3)*(2*p.k1+p.k2)) + cos(2*pi*p.k2)
        elif p.SC_type == 'px':
            #triplet: sin(2*pi*kx)
            delta_func = sin(2*pi*p.k1)  #sin(2*pi*1/sqrt(3)*(2*p.k1+p.k2))
        elif p.SC_type == 'py':
            #triplet: sin(2*pi*p.ky)
            delta_func = sin(2*pi*p.k2)
        elif p.SC_type == 'd':
            #singlet: cos(2*pi*kx) - cos(2*pi*ky)
            delta_func = cos(2*pi*p.k1) - cos(2*pi*p.k2)  #cos(2*pi*1/sqrt(3)*(2*p.k1+p.k2)) - cos(2*pi*p.k2)
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
        
        result, _, _, _ = sc.linalg.lstsq(b.fermi_Uln, f, lapack_driver='gelsy')
        result[abs(result) < 10**(-13)] = 0
        self.f = dot(b.fermi_Ulx, result).reshape(len(b.ft),p.nk,p.nwan,p.nwan)


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
            
            
            # Calculate - V*F [carefull about minus!]
            y = - einsum('ijkmln,ijml->ijkn', self.v, self.f)
            y = y.reshape(len(b.ft),p.nk1,p.nk2,p.nk3,p.nwan**2)
            
            fft_object = pyfftw.builders.ifftn(y, axes=(1,2,3))
            y = fft_object()/p.nk
            y = y.reshape(len(b.ft),p.nk*p.nwan**2)

            result, _, _, _ = sc.linalg.lstsq(b.fermi_Ulx, y, lapack_driver='gelsy')
            y = dot(b.fermi_Uln, result)
            y = y.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
            
            y_const = -einsum('kmln,ml->kn',self.v_DC.reshape(p.nwan,p.nwan,p.nwan,p.nwan),self.f[0,0])*ones((len(b.fm),p.nk,p.nwan,p.nwan))/p.nk
            y = y + y_const - real(lam_in)*self.delta

            ### Impose symmetry conditions            
            # Even function of matsubara frequency
            y = (y + y[::-1])/2 
            
            # k-space symmetry depending on singlet/triplet:
            # y_2 corresponds to Delta_ba(-k)!
            y_2 = y.reshape(len(b.fm),p.nk1,p.nk2,p.nwan,p.nwan)
            y_2 = roll(y_2,-1,(1,2))[:,::-1,::-1]
            y_2 = y_2.reshape(len(b.fm),p.nk,p.nwan,p.nwan) 
            y_2 = transpose(y_2,axes=(0,1,3,2))
            if p.SC_type in {'s', 's_ext', 'd'}:
                # singlet case: f_ab(k) = f_ba(-k)
                y = (y + y_2)/2     
            elif p.SC_type in {'px', 'py', 'f1', 'f2'}:     
                # triplet case: f_ab(k) = - f_ba(-k)
                y = (y - y_2)/2  

            # Subtract highest matsubara frequency
            y = y - y[0]

            ### Calculating lambda
            lam1 = sum(conj(y)*self.delta)

            
            self.delta = y/linalg.norm(y)
            print(n,lam1,linalg.norm(y), file=open(p.Logstr,'a'))
            
        return lam1
    
    ##############
    # Function for saving given data to files
    ##############
    
    def func_write(self, p, func, savename):
        '''
        Function to save total x = (tau,r) or x = (iwn,k) dependence of calculated functions.
        Expects function in shape (tau/iwn,k_1,k_2,k_3,nwan,nwan[,nwan,nwan]).
        '''
        
        # Initialize core number for data writing
        pool = mp.Pool(mp.cpu_count())
        
        # Dummy path string 
        save_str = p.SC_savepath + savename + '{}{}{}{}.dat'
        
        # For rank two tensor (matrix)
        if len(func.shape) == 3:
             open_and_write2(save_str.format('trace_kspace','','',''),func)
        
        if len(func.shape) == 6:
             pool.starmap(open_and_write, [(save_str.format(n1,n2,'',''),\
                                            func[:,:,:,:,n1,n2])\
                                            for n1 in range(p.nwan)\
                                            for n2 in range(p.nwan)])

        # For rank four tensor
        elif len(func.shape) == 8:
             pool.starmap(open_and_write, [(save_str.format(n1,n2,m1,m2),\
                                            func[:,:,:,:,n1,n2,m1,m2])\
                                            for n1 in range(p.nwan)\
                                            for n2 in range(p.nwan)\
                                            for m1 in range(p.nwan)\
                                            for m2 in range(p.nwan)])
        
        pool.close()
