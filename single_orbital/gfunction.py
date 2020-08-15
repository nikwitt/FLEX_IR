## -*- encoding: latin-1 -*-

from numpy import *
from scipy import optimize
import scipy as sc
import pyfftw
import h5py

##############
# Calculate Greens function within FLEX approximation
##############

class gfunction_calc:
    """
    Self consistency loop for Greens function, using FLEX approximation
    """
    def __init__(self,p,b):
        ##### Tag for convergence check
        self.tag = 'calc'
        
        ##### Static parameters in calculations
        self.io_ = tensordot(      b.fm, ones(p.nk), axes=0).reshape(len(b.fm),p.nk)
        self.ek_ = tensordot(ones(len(b.fm)),  p.ek, axes=0).reshape(len(b.fm),p.nk)
        
        ##### Set parameters for U convergence
        u0_delta = 0.01 #change this parameter for U convergence if not working
        self.u0_pval  = p.u0
        u0_store = -1
        u0_it    = 0
        
        ##### Set initial round (bare quantities)
        if p.round_it == 0:
            print("G convergence from ZERO.", file=open(p.Logstr,'a'))
            self.set_mu_0(p)
            self.sigma = zeros((len(b.fm), p.nk))
        else:
            print("G convergence from pre converged G.", file=open(p.Logstr,'a'))
            with h5py.File(p.loadpath,'r') as file: 
                self.sigma = file['gfunction/sigma'][()]
                self.mu    = file['gfunction/mu'][()]
            
        self.set_gkio(p,b,self.mu)
        self.gkio = self.gkio.reshape(len(b.fm),p.nk1,p.nk2)
        self.gkio = (self.gkio + roll(self.gkio,-1,(1,2))[:,::-1,::-1])/2
        self.gkio = self.gkio.reshape(len(b.fm),p.nk)
        
        gkio_old = self.gkio
        self.set_grit(p,b)
        self.set_ckit(p,b)
        
        
        ##### Self consistency loop
        if p.mode == 'FLEX':
            while amax(abs(p.u0*self.ckio)) >= 1 or p.u0 != self.u0_pval or u0_it == 0:

                # Safety check for too long running calculations
                u0_store = p.u0
                u0_it += 1 
                if u0_it == 50:
                    print("U iteration reached step 50. Everything okay?", file=open(p.Logstr,'a'))
                    print(p.err_str_begin + "U iteration reached step 100", file=open(p.Logerrstr,'a'))
                if u0_it == 100:
                    print("U iteration reached step 100. It will be stopped!", file=open(p.Logstr,'a'))
                    print(p.err_str_begin + "U iteration reached step 100", file=open(p.Logerrstr,'a'))
                    break
            
                # Setting new U if max(|chi0*U|) >= 1
                print('### Check for renormalization |U*chi|: ' + str(amax(abs(p.u0*self.ckio))) + ', U = ' + str(p.u0), file=open(p.Logstr,'a'))
                if amax(abs(p.u0*self.ckio)) >= 1:
                    p.u0 = min(self.u0_pval,
                        amax(abs(p.u0*self.ckio))/(amax(abs(p.u0*self.ckio)) + u0_delta)\
                        * 1/amax(abs(self.ckio)))
                elif u0_it > 1:
                    p.u0 = min(self.u0_pval,
                        amax(abs(p.u0*self.ckio))/(amax(abs(p.u0*self.ckio)) + u0_delta)\
                        * 1/amax(abs(self.ckio)))

                print('New U value: ' + str(p.u0) + ', with |U*chi|: ' + str(amax(abs(p.u0*self.ckio))), file=open(p.Logstr,'a'))
            
                # Safety check if U is the same value twice -> some error has occured!
                if p.u0 == u0_store and u0_it != 1:
                    print("Same U value as before!", file=open(p.Logstr,'a'))
                    print(p.err_str_begin + "Same U value reached twice, abbortion! No convergence! (U = "+str(p.u0) +")", file=open(p.Logerrstr,'a'))
                    break
            
                # Setting of convergence tolerance and iteration number (U ~= U_in -> oneshot calculation)
                if p.u0 == self.u0_pval:
                    conv_tol = p.g_sfc_tol
                    sfc_it_max = 150
                else:
                    conv_tol = 8e-2
                    sfc_it_max = 1
            
                # Convergence cycle of self energy sigma for given U
                print('Convergence round for U = ' + str(p.u0) + \
                      ',|U*chi|: ' + str(amax(abs(p.u0*self.ckio))),\
                      file=open(p.Logstr,'a'))
                for it_sfc in range(sfc_it_max):
                    self.sigma_old = self.sigma                
                    self.set_V(p,b)
                    self.set_sigma(p,b)
           
                    self.set_mu_from_gkio(p,b)
                    self.set_gkio(p,b,self.mu)
                    self.gkio = self.gkio.reshape(len(b.fm),p.nk1,p.nk2)
                    self.gkio = (self.gkio + roll(self.gkio,-1,(1,2))[:,::-1,::-1])/2
                    self.gkio = self.gkio.reshape(len(b.fm),p.nk)
                    #Mixing: Change values if needed!
                    if p.u0 != 1:
                        self.gkio = p.mix*self.gkio + (1-p.mix)*gkio_old

                    gkio_old = self.gkio
                    self.set_grit(p,b)
                    self.set_ckit(p,b)
            
                    print(it_sfc, sum(abs(self.sigma_old-self.sigma))/sum(abs(self.sigma)),\
                          file=open(p.Logstr,'a'))
                    if sum(abs(self.sigma_old-self.sigma))/sum(abs(self.sigma)) <= conv_tol:
                        break         
                    
        
            ##### Security convergence check
            ### U convergence
            if p.u0 != self.u0_pval and p.mode == 'FLEX':
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nU is not initial input. Stopping gfunction."\
                      , file=open(p.Logstr,'a'))
                print(p.err_str_begin + "U != U_init | gfunction stopped."\
                      , file=open(p.Logerrstr,'a'))
                return
        
            ### Sigma convergence
            if sum(abs(self.sigma_old-self.sigma))/sum(abs(self.sigma)) > conv_tol and p.mode == 'FLEX':
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nSigma not converged. Stopping gfunction."\
                      , file=open(p.Logstr,'a'))
                print(p.err_str_begin + "Sigma not converged (diff = "\
                      + str(sum(abs(self.sigma_old-self.sigma))/sum(abs(self.sigma)))\
                      + ") | gfunction stopped.", file=open(p.Logerrstr,'a'))
                return
        
        

        ### Save largest eigenvalue of BSE kernel U*chi
        print("Done. Extract largest BSE kernel value.", file=open(p.Logstr,'a'))
        # Extract index of max{chi}
        max_index = unravel_index(argmax(self.ckio), self.ckio.shape)
        BSE_max_kernel = p.u0*real(self.ckio[max_index])
        if max_index[0] != b.b_iwn_zero_ind:
            print(p.err_str_begin + ": Maximal BSE kernel value not at iw=0.", file=open(p.Logstr,'a'))
            print("Maximal BSE kernel value not at iw=0.", file=open(p.Logerrstr,'a'))
        
        # Save file
        file = open(('BSE_kernel_n_T_data/largest_BSEK_for_' +\
                    'n_{}_tpr_{}_U_{}.dat').format(p.n_fill,p.t_prime,p.u0),"a")
        file.write('{} {} {} {} {}\n'.format(p.T, max_index[0], p.k1[max_index[1]], p.k2[max_index[1]], BSE_max_kernel))
        
        print("*** Maximal BSE kernel value U*chi = " + str(BSE_max_kernel),\
              file=open(p.Logstr,'a'))
        
        
        ##### Finished loop, save results
        print("Saving all data now...", file=open(p.Logstr,'a'))
        with h5py.File(p.savepath,'a') as file: 
            group = file.require_group('gfunction')
            
            group.require_dataset('gkio' , data=self.gkio , shape=self.gkio.shape,  dtype=complex)
            group.require_dataset('sigma', data=self.sigma, shape=self.sigma.shape, dtype=complex)
            group.require_dataset('mu'   , data=self.mu,    shape=(),               dtype=float)
            group.require_dataset('BSE_max_kernel', data=BSE_max_kernel, shape=(),  dtype=float)
            #group.create_dataset('ckio' , data=self.ckio )
            #group.create_dataset('sigma', data=self.sigma)
        
        
        print("Done! -> Move to SC calculation.", file=open(p.Logstr,'a'))
            
            
    ##############
    # Calculate chemical potential via bisection method.
    # Either from fermi function (initial value) or from G(k, iw_n)
    ##############

    ### Set from fermi function -----------------------------------------------
    def set_mu_0(self,p):
        pass
        """
        Determining inital mu from bisection method of sum(fermi_k)
        """
        # Set calculation parameters
        eps = 1e-14
        it  = 1
        it_max = 10000
        
        # Set electron number for bisection difference
        n_0 = p.n_fill        
        n = self.calc_electron_density
        f = lambda mu : 2*n(p,mu) - n_0
        
        self.mu = sc.optimize.bisect(f, 50, -50)
        

    #--------------------------------------------------------------------------
    def calc_electron_density(self,p,mu):
        
        #E    = ones(len(p.ek))
        n = sum( 1 / (1 + exp(p.beta*(p.ek - mu)) ) ) / p.nk
        return n
    
    
    ### Set from Greens function ----------------------------------------------
    def set_mu_from_gkio(self,p,basis):
        pass
        """
        Determining  iteration mu from bisection method of 1 + sum(gkio)
        """
        # Set electron number from bisection difference
        n_0 = p.n_fill
        n = self.calc_electron_density_from_gkio
        f = lambda mu : 2*n(p,basis,mu) - n_0


        self.mu = sc.optimize.bisect(f, 50, -50)
      
        
    #--------------------------------------------------------------------------
    def calc_electron_density_from_gkio(self,p,b,mu):
        self.set_gkio(p,b,mu)
        gio = sum(self.gkio,axis=1)/p.nk
        n   = 1 + real(dot(b.fermi_iw_to_tau_0, gio))
        return n


    ##############
    ### Set functions for self consistency loop.
    # set_gkio  : p, b | calculates G(k, iw_n)
    # set_grit  : p, b | calculates G(r, tau) via FFT + irbasis on bosonic tau
    # set_ckit  : p, b | calculates chi_0(k, iv_m) via G(r, tau) and FFT + irbasis
    # set_v     : p, b | calculates V(r, tau) on fermionic tau via chi_0, FFT + irbasis
    # set_sigma : p, b | calculates Sigma(k, iw_n) via V and G
    ##############
    
    
    ### Set G(k, iw_n) --------------------------------------------------------
    def set_gkio(self, p, b, mu):
        self.gkio = 1/(self.io_ - (self.ek_ - mu) - self.sigma)
 
       
    ### Set G(r, tau) ---------------------------------------------------------
    def set_grit(self, p, b):
        if not hasattr(self, 'gkio'): exit("Error_set_grit")
        grit = self.gkio.reshape(len(b.fm), p.nk1, p.nk2, p.nk3)
        
        fft_object = pyfftw.builders.fftn(grit, axes=(1,2,3))
        grit = fft_object()
        grit = grit.reshape(len(b.fm),p.nk)
 
        self.grit_b = dot(b.fermi_iw_to_tau_boson, grit)
        self.grit_f = dot(b.fermi_iw_to_tau,       grit)



    ### Set chi_0(k, iv_m) ----------------------------------------------------
    def set_ckit(self, p, b):
        if not hasattr(self, 'grit_b'): exit("Error_set_ckit")
        ckio = self.grit_b*self.grit_b[::-1,:]

        ckio = dot(b.bose_tau_to_iw, ckio)
        ckio = ckio.reshape(len(b.bm),p.nk1,p.nk2,p.nk3)

        fft_object = pyfftw.builders.ifftn(ckio, axes=(1,2,3))
        ckio = fft_object()/p.nk
        self.ckio = ckio.reshape(len(b.bm),p.nk)



    ### V(r, tau) -------------------------------------------------------------
    def set_V(self,p,b):       
        chi_spin   = self.ckio / (1 - p.u0*self.ckio)
        chi_charge = self.ckio / (1 + p.u0*self.ckio)

        V = 3./2.*p.u0*p.u0*chi_spin + 1./2.*p.u0*p.u0*chi_charge - p.u0*p.u0*self.ckio
        V = V.reshape(len(b.bm),p.nk1,p.nk2,p.nk3)
        self.V_ = V

        fft_object = pyfftw.builders.fftn(V, axes=(1,2,3))
        V = fft_object().reshape(len(b.bm),p.nk)    
        
        self.V = dot(b.bose_iw_to_tau_fermi, V)  
        ### No constant term - it is absorbed into mu
        
        
    ### Sigma(k, iw_n) --------------------------------------------------------
    def set_sigma(self,p,b):
        sigma = self.V * self.grit_f
        
        sigma = dot(b.fermi_tau_to_iw, sigma)
        sigma = sigma.reshape(len(b.fm),p.nk1,p.nk2,p.nk3)
        
        fft_object = pyfftw.builders.ifftn(sigma, axes=(1,2,3))
        sigma = fft_object()/p.nk
        
        self.sigma = sigma.reshape(len(b.fm),p.nk1,p.nk2)
        self.sigma = (self.sigma + roll(self.sigma,-1,(1,2))[:,::-1,::-1])/2
        self.sigma = self.sigma.reshape(len(b.fm),p.nk)
        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
       
##############
# Load Greens function if already calculated
##############       
        
class gfunction_load:
    """
    Load already calculated and saved quantities
    """
    def __init__(self,p,b):
        ##### Tag for convergence check
        self.tag = 'load'
        
        ##### Calculation already finished, load data
        print("Self consistency loop already done. Loading data now...", file=open(p.Logstr,'a'))
        #G(iwn_fermi, k)
        with h5py.File(p.loadpath,'r') as file: 
            self.gkio = file['gfunction/gkio'][()]
       
        #chi_0(iwn_bose, k)
        grit = self.gkio.reshape(len(b.fm), p.nk1, p.nk2, p.nk3)
        fft_object = pyfftw.builders.fftn(grit, axes=(1,2,3))
        grit = fft_object()
        grit = grit.reshape(len(b.fm),p.nk)
        self.grit_b = dot(b.fermi_iw_to_tau_boson, grit)
        
        ckio = self.grit_b*self.grit_b[::-1,:]
        ckio = dot(b.bose_tau_to_iw, ckio)
        ckio = ckio.reshape(len(b.bm),p.nk1,p.nk2,p.nk3)
        fft_object = pyfftw.builders.ifftn(ckio, axes=(1,2,3))
        ckio = fft_object()/p.nk
        self.ckio = ckio.reshape(len(b.bm),p.nk)