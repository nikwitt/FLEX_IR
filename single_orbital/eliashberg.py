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
            
            file = open("lam_n_T_data/" + p.SC_type + "w_lam_for_n_" + str(p.n_fill)\
                  + "_tpr_" + str(p.t_prime) + "_U_" + str(p.u0) + "_first_negative.dat","a")
            file.write(str(p.T) + " " + str(real(lam)) + " " + str(imag(lam)) + "\n")
            file.close()
            
            lam_n = self.scf(g, p, b, lam)
            lam = lam_n + lam           
            
        self.result = lam
        
        ##### Finished loop, save results
        print("Saving all data now...", file=open(p.Logstr,'a'))
        with h5py.File(p.savepath,'a') as file: 
            group = file.create_group('eliashberg')
            
            group.create_dataset('{}_gap'.format(p.SC_type)    , data=self.delta )
            group.create_dataset('{}_lambda'.format(p.SC_type) , data=self.result)
        
        
        
    ### Set Coulomb interaction V(r, tau_fermi)--------------------------------
    def set_v(self, g, p, b):
        #E = ones((len(b.bm),p.nk))
        spin   = g.ckio / (1 - p.u0*g.ckio)
        charge = g.ckio / (1 + p.u0*g.ckio)
        
        # Set V according to parity/SC wave type
        if p.SC_type in {'s', 'd'}: #singulett
            #Complete singulett: v = p.u0*E + 3./2.*p.u0*p.u0*spin - 1./2.*p.u0*p.u0*charge    
            v =  3./2.*p.u0*p.u0*spin - 1./2.*p.u0*p.u0*charge
            self.v_DC = p.u0
        elif p.SC_type in {'p'}: #triplett
            v = -1./2.*p.u0*p.u0*spin - 1./2.*p.u0*p.u0*charge
            self.v_DC = 0
            
        v = v.reshape(len(b.bm),p.nk1,p.nk2,p.nk3)
        self.v_ = v
        
        #FFT to (r, tau_fermi) althoguh V is bosonic!
        fft_object = pyfftw.builders.fftn(v, axes=(1,2,3))
        v = fft_object().reshape(len(b.bm),p.nk)
        
        result, _, _, _ = sc.linalg.lstsq(b.bose_Uln, v, lapack_driver='gelsy')
        self.v = dot(b.bose_Ulx_fermi, result)


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
        self.f = self.f.reshape(len(b.fm), p.nk)
        
        result, _, _, _ = sc.linalg.lstsq(b.fermi_Uln, self.f, lapack_driver='gelsy')
        self.f_l = result
        ## Cutting everything small off!
        #result[abs(result) < 10**(-8)] = 0
        self.f = dot(b.fermi_Ulx, result)
        

    ##############
    ### Self consistency loop for linearized Eliashberg equation
    ### Employs power iterative method to solve lam*delta = lam*V*F in (r,tau)-space
    ##############

    def scf(self, g, p, b, lam_in):
        """
        Self consistency loop for super conduction parameter via eigenvalue method.
        Implements FLEX approximation in linearized Eliashberg equation.
        Handles depending on SC-type input in p.SC_type(=parameters) the equation differently.
        """
    
        lam1 = 0.1
        lam0 = 0.0
        for n in range(300):
            if abs(lam1-lam0)/abs(lam1) <= p.SC_sfc_tol: break
            # Power iteration method for computing lambda
            lam0 = lam1
            self.set_f(g, p, b)
            
            # Calculate - V*F [carefull about minus!]
            y = self.v*self.f
            y = y.reshape(len(b.ft),p.nk1,p.nk2,p.nk3)
            
            fft_object = pyfftw.builders.ifftn(y, axes=(1,2,3))
            y = fft_object()/p.nk
            y = y.reshape(len(b.ft),p.nk)

            result, _, _, _ = sc.linalg.lstsq(b.fermi_Ulx, y, lapack_driver='gelsy')
            y = dot(b.fermi_Uln, result)
            y = y - self.v_DC*self.f[0,0]/p.nk
    
            # Shift eigenvalue if necessary (trick power iteration method)
            y = y - real(lam_in)*self.delta
            
            # gap is even function of ion
            y = (y + y[::-1])/2 

            
            # Ensuring even/odd parity of singlet/tripled SC!
            y = y.reshape(len(b.fm),p.nk1,p.nk2)
            if p.SC_type in {'s','d'}:
                y = (y + roll(y,-1,(1,2))[:,::-1,::-1])/2
            if p.SC_type in {'p'}:
                y = (y - roll(y,-1,(1,2))[:,::-1,::-1])/2
                
            y = y.reshape(len(b.fm),p.nk)

            # Calculating lambda, v^TAv = lambda 
            lam1 = sum(conj(y)*self.delta)
            
            self.delta = y/linalg.norm(y)
            print(n,lam1,linalg.norm(y), file=open(p.Logstr,'a'))
        

        return lam1
    
    def func_write(self, p, x, func, savestr):
        '''
        Function to save total x = (tau,r) or x = (iwn,k) dependence of calculated functions
        Expects function in shape (tau/iwn,k_1,k_2,k_3)
        '''
            
        if p.SC_savetype == 'short':
            zero = list(x).index(0)
            file = open(p.SC_savepath + savestr,"w") 
            for i in range(p.nk1):
                for j in range(p.nk2):
                    file.write(str(i) + " " + str(j) + " " + str(real(func[zero][i][j][0])) + " " + str(imag(func[zero][i][j][0])) + "\n")
                file.write("\n")
            file.close()
                
        elif p.SC_savetype == 'long':
            file = open(p.SC_savepath + savestr,"w")
            for x_it in range(len(x)): 
                for i in range(p.nk1):
                    for j in range(p.nk2):
                        file.write(str(x[x_it]) + " " + str(i) + " " + str(j) +\
                                   " " + str(real(func[x_it][i][j][0])) +\
                                   " " + str(imag(func[x_it][i][j][0])) + "\n")
                    file.write("\n")
                file.write("\n")
            file.close()