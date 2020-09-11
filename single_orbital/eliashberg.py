## -*- encoding: latin-1 -*-

from numpy import *
import scipy as sc
import pyfftw
import h5py

class eliashberg:
    def __init__(self, g, p, b):
        self.set_v(g, p, b)
        self.set_delta0(g, p, b)

        lam = self.scf(g, p, b, 0)

        if real(lam) < 0 and p.SC_type != 's' and p.mode == 'FLEX':
            print('Not s-wave type but found negative lambda. Another cycle will be performed.', file=open(p.Logstr,'a'))
            print(p.err_str_begin + "lambda < 0 : => new round!",\
                  file=open(p.Logerrstr,'a'))
            with open(p.SC_EV_path_neg, 'a') as file:
                file.write("{} {} {}\n".format(p.T, real(lam), imag(lam)))
            
            lam_n = self.scf(g, p, b, lam)
            lam = lam_n + lam           
            
        self.result = lam
        
        ##### Finished loop, save results
        print("Saving all data now...", file=open(p.Logstr,'a'))
        with h5py.File(p.savepath,'a') as file: 
            group = file.require_group('eliashberg')
            
            group.require_dataset('{}_gap'.format(p.SC_type)    , data=self.delta , shape=self.delta.shape, dtype=complex)
            group.require_dataset('{}_lambda'.format(p.SC_type) , data=self.result, shape=(), dtype=complex)
         
        
        
    ### Set Coulomb interaction V(r, tau_fermi)--------------------------------
    def set_v(self, g, p, b):
        spin   = g.ckio / (1 - p.u0*g.ckio)
        charge = g.ckio / (1 + p.u0*g.ckio)
        
        # Set V according to parity/SC wave type
        if p.SC_type in {'s', 'd'}: #singulett
            v =  3./2.*p.u0*p.u0*spin - 1./2.*p.u0*p.u0*charge
            self.v_DC = p.u0
        elif p.SC_type in {'p'}: #triplett
            v = -1./2.*p.u0*p.u0*spin - 1./2.*p.u0*p.u0*charge
            self.v_DC = 0
            
        v = v.reshape(len(b.bm),p.nk1,p.nk2,p.nk3)
        #self.v_ = v
        
        fft_object = pyfftw.builders.fftn(v, axes=(1,2,3))
        v = fft_object().reshape(len(b.bm),p.nk)
        
        self.v = dot(b.bose_iw_to_tau_fermi, v)


    ### Set inital gap delta0(k, iw_n) --------------------------------------
    def set_delta0(self, g, p, b):
        if p.SC_type == 's':
            self.delta0 = ones(p.nk)
        elif p.SC_type == 'p':
            self.delta0 = sin(2*pi*p.k1)
        elif p.SC_type == 'd':
            self.delta0 = cos(2*pi*p.k1) - cos(2*pi*p.k2)
            
        self.delta  = tensordot(ones(len(b.fm)), self.delta0, axes=0).reshape(len(b.fm), p.nk)
        self.delta  = self.delta / linalg.norm(self.delta)


    ### Set anomalous GF F(r, tau_fermi) --------------------------------------
    def set_f(self, g, p, b):
        self.f = - g.gkio*conj(g.gkio)*self.delta #G(k) = G(-k) used!
        self.f_= self.f
        self.f = self.f.reshape(len(b.fm),p.nk1,p.nk2,p.nk3)
    
        fft_object = pyfftw.builders.fftn(self.f, axes=(1,2,3))
        self.f = fft_object().reshape(len(b.fm),p.nk)
        
        self.f_0 = dot(b.fermi_iw_to_tau_0, self.f[:,0])
        self.f = dot(b.fermi_iw_to_tau, self.f)
        

    ##############
    ### Self consistency loop for linearized Eliashberg equation
    ### Employs power iterative method to solve lam*delta = lam*V*F in (r,tau)-space
    ##############

    def scf(self, g, p, b, lam_in):
        """
        Power iteration method for super conduction eigenvalue.
        Implements FLEX approximation in linearized Eliashberg equation.
        Sets spin chanel depending on SC-type input in p.SC_type(=parameters).
        """
    
        lam1 = 0.1
        lam0 = 0.0
        for n in range(300):
            if abs(lam1-lam0)/abs(lam1) <= p.SC_sfc_tol: break
            # Power iteration method for computing lambda
            lam0 = lam1
            self.set_f(g, p, b)
            
            # K*gap = y = V*F
            y = self.v*self.f
            y = y.reshape(len(b.ft),p.nk1,p.nk2,p.nk3)
            
            fft_object = pyfftw.builders.ifftn(y, axes=(1,2,3))
            y = fft_object()/p.nk
            y = y.reshape(len(b.ft),p.nk)

            y = dot(b.fermi_tau_to_iw, y)
            y = y - self.v_DC*self.f_0/p.nk
    
            # Shift eigenvalue if necessary (power iteration method trick)
            y = y - real(lam_in)*self.delta
            
            # Gap is even function of iw_n
            y = (y + y[::-1])/2 

            
            # Ensuring even/odd parity of singlet/tripled SC!
            y = y.reshape(len(b.fm),p.nk1,p.nk2)
            if p.SC_type in {'s','d'}:
                y = (y + roll(y,-1,(1,2))[:,::-1,::-1])/2
            if p.SC_type in {'p'}:
                y = (y - roll(y,-1,(1,2))[:,::-1,::-1])/2
                
            y = y.reshape(len(b.fm),p.nk)

            # Calculating lambda = gap^T*K*gap 
            lam1 = sum(conj(y)*self.delta)
            
            self.delta = y/linalg.norm(y)
            print(n,lam1,linalg.norm(y), file=open(p.Logstr,'a'))
        

        return lam1