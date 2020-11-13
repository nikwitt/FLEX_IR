## -*- encoding: latin-1 -*-

from numpy import *
from einsum2 import einsum2
from scipy import optimize
import scipy as sc
import pyfftw
import h5py

### Set U, U', J, J' as in parameters file (reduce module dependency)
def Uval_set(u0, JU_ratio):
    J        = JU_ratio*u0
    u0_prime = u0 - 2*J
    J_prime  = J
    return u0, JU_ratio, u0_prime, J, J_prime

##########################################################################
##### Main code: Calculate Greens function within FLEX approximation #####
##########################################################################

class gfunction_calc:
    """
    Self consistency loop for Greens function, using FLEX approximation
    """
    def __init__(self,p,h,b):        
        ##### Tag for convergence check
        self.tag = 'calc'
        
        ##### Set parameters for U convergence
        u0_delta = 0.01 #change this parameter for U convergence if not working
        conv_it  = 2
        self.u0_pval  = p.u0
        u0_store = -1
        u0_it    = 0
        
        ##### Set static quantities (unit matrix, Hamiltonian in (iw, k)-space)
        self.set_static_quantities(p,h,b)
        
        
        ##### Set initial round (bare quantities)
        ## Initial guess for mu, sigma
        if p.round_it == 0:
            print("G convergence from ZERO.", file=open(p.Logstr,'a'))
            self.mu = h.mu
            self.sigma = zeros((len(b.fm), p.nk, p.nwan, p.nwan))
        else:
            print("G convergence from pre converged G.", file=open(p.Logstr,'a'))
            with h5py.File(p.loadpath,'r') as file: 
                self.sigma = file['gfunction/sigma'][()]
                self.mu    = file['gfunction/mu'][()]

        ## Set G(iwn, k), G(tau, r), Chi_0(iwn, k)
        self.set_gkio(p,h,b,self.mu)
        gkio_old = self.gkio
        self.set_grit(p,b)
        self.set_ckit(p,b)
        
        
        ##### Self consistency loop
        if p.mode == 'FLEX':
            div_check_param = self.Max_eigval_ChiU(p,h)
            
            while div_check_param >= 1 or p.u0 != self.u0_pval or u0_it == 0:

                ##### Check for 1-max(chi@U) > 0, otherwise decrease U
                ### ONLY eigval_magnetic >= 1 check included, not charge!
                # Safety check for too long running calculations
                u0_store = p.u0
                u0_it += 1 
                if u0_it == 100:
                    print("U iteration reached step 100. Everything okay?", file=open(p.Logstr,'a'))
                    print(p.err_str_begin + "U iteration reached step 100", file=open(p.Logerrstr,'a'))
                if u0_it == 150:
                    print("U iteration reached step 150. It will be stopped!", file=open(p.Logstr,'a'))
                    print(p.err_str_begin + "U iteration reached step 150", file=open(p.Logerrstr,'a'))
                    break
            
                # If it's not good already after one cycle, reset U
                if p.u0 != self.u0_pval or u0_it > 1:
                    p.u0 = min(self.u0_pval, p.u0*1.5)    
                    _, _, p.u0_prime, p.J, p.J_prime = Uval_set(p.u0, p.JU_ratio)
                    h.set_interaction(p)                    
                    div_check_param = self.Max_eigval_ChiU(p,h)
                    conv_it -= 10
                    conv_it  = max(conv_it, 1)
                    
                    
                ##### Setting new U if max(|chi0*U|) >= 1
                print('### Check for renormalization |chi@U_S|: ' + str(div_check_param) + ', U = ' + str(p.u0), file=open(p.Logstr,'a'))
                
                ckio_max = amax(linalg.eigh(self.ckio.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**2,p.nwan**2))[0])
                while div_check_param >= 1:
                    while div_check_param/(div_check_param + u0_delta*conv_it)*1/ckio_max - self.u0_pval > 0.01:
                        conv_it += 5
                        #print(conv_it, div_check_param/(div_check_param + u0_delta*conv_it)*1/ckio_max)
                    p.u0 = min(self.u0_pval,\
                        div_check_param/(div_check_param + u0_delta*conv_it)*1/ckio_max)
                    #print(ckio_max, div_check_param/(div_check_param + u0_delta*conv_it)*1/ckio_max)
                    
                    print("New U set to " + str(p.u0), conv_it)
                    if p.u0 == self.u0_pval:
                        conv_it += 1
                    else:
                        _, _, p.u0_prime, p.J, p.J_prime = Uval_set(p.u0, p.JU_ratio)
                        h.set_interaction(p)
                        div_check_param = self.Max_eigval_ChiU(p,h)
                        if div_check_param >=1:
                            conv_it += 1
                    

                print('New U value: ' + str(p.u0) + ', with |chi@U_S|: ' + str(div_check_param), file=open(p.Logstr,'a'))
            
                # Safety check if U is the same value twice -> some error has occured!
                if p.u0 == u0_store and u0_it != 1:
                    print("Same U value as before!", file=open(p.Logstr,'a'))
                    print(p.err_str_begin + "Same U value reached twice, abbortion! No convergence! (U = "+str(p.u0) +")", file=open(p.Logerrstr,'a'))
                    break
            
            
                #--------------------------------------------------------------
                #### Convergence cycle of FLEX self-energy for given U
                # Setting of convergence tolerance and iteration number (U ~= U_in -> oneshot calculation)
                if p.u0 == self.u0_pval:
                    conv_tol   = p.g_sfc_tol
                    sfc_it_max = 150
                    mix        = p.mix
                else:
                    conv_tol   = 8e-2
                    sfc_it_max = 1
                    mix        = p.mix
            
                # Convergence loop
                print('Convergence round for U = ' + str(p.u0) + \
                      ',|chi@U_S|: ' + str(div_check_param),\
                      file=open(p.Logstr,'a'))
                for it_sfc in range(sfc_it_max):
                    self.sigma_old = self.sigma                
                    self.set_V(p,h,b)
                    self.set_sigma(p,b)

                    self.set_mu_from_gkio(p,h,b)
                    self.set_gkio(p,h,b,self.mu)
                    self.symmetrize_gkio(p,b)
                    
                    #Mixing: Change values if needed!
                    if p.u0 != 1:
                        self.gkio = mix*self.gkio + (1-mix)*gkio_old

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
        if p.mode == 'FLEX' and sum(abs(self.sigma_old-self.sigma))/sum(abs(self.sigma)) > conv_tol:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nSigma not converged. Stopping gfunction."\
                  , file=open(p.Logstr,'a'))
            print(p.err_str_begin + "Sigma not converged (diff = "\
                  + str(sum(abs(self.sigma_old-self.sigma))/sum(abs(self.sigma)))\
                  + ") | gfunction stopped.", file=open(p.Logerrstr,'a'))
            return

        ### Calculate G function of negative k
        gkio_invk  = self.gkio.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan**2)
        fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
        gkio_invk  = fft_object()
        fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
        gkio_invk  = fft_object()/p.nk
        self.gkio_invk = gkio_invk.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
        print("Self consistency loop finished!", file=open(p.Logstr,'a'))
        
        
        ### Calculate maximal magnetic/charge eigenvalue
        print("Extract largest BSE kernel values.", file=open(p.Logstr,'a'))
        BSE_max_spin   = self.Max_eigval_ChiU(p,h,'S')
        BSE_max_charge = self.Max_eigval_ChiU(p,h,'C')

        with open(p.BSE_EV_path,"a") as file:
            file.write('{} {} {}\n'.format(p.T, BSE_max_spin, BSE_max_charge))

        print("### Maximal BSE kernel value Chi@U_S = " + str(BSE_max_spin)\
              + " | Chi@U_C = " + str(BSE_max_charge), file=open(p.Logstr,'a'))


        ##### Finished loop, save results
        print("Saving all data now...", file=open(p.Logstr,'a'))

        with h5py.File(p.savepath,'a') as file:
            group = file.require_group('gfunction')
            
            group.require_dataset('gkio' , data=self.gkio , shape=self.gkio.shape, dtype=complex)
            group.require_dataset('sigma', data=self.sigma, shape=self.sigma.shape, dtype=complex)
            group.require_dataset('mu', data=self.mu, shape=(), dtype=float)
            
            group.require_dataset('BSE_max_spin'  , data = BSE_max_spin  , shape=(), dtype=float)
            group.require_dataset('BSE_max_charge', data = BSE_max_charge, shape=(), dtype=float)


        print("Done! -> Move to SC calculation.", file=open(p.Logstr,'a'))

            
    ##############
    # Calculate chemical potential via bisection method from G(k, iw_n).
    ##############
    
    ### Set from Greens function ----------------------------------------------
    def set_mu_from_gkio(self,p,h,basis):
        pass
        """
        Determining iteration mu from bisection method of 1 + sum(gkio)
        """
        
        ### Set electron number for bisection difference
        # n_0 is per electron orbital!
        n_0 = p.n_fill
        n = self.calc_electron_density_from_gkio
        f = lambda mu : 2*n(p,h,basis,mu) - n_0
       
        self.mu = sc.optimize.bisect(f, 50, -50)

    #--------------------------------------------------------------------------
    def calc_electron_density_from_gkio(self,p,h,b,mu):
        self.set_gkio(p,h,b,mu)
        gio = sum(self.gkio,axis=1)/p.nk
        gio = trace(gio,0,1,2)/p.norb #offset=0, axis=1,2
        
        n = 1 + real(dot(b.fermi_iw_to_tau_0, gio))
        return n


    ##############
    ### Set functions for self consistency loop.
    # set_gkio  : p, b | calculates G(k, iw_n)
    # set_grit  : p, b | calculates G(r, tau) via FFT + irbasis on bosonic tau
    # set_ckit  : p, b | calculates chi_0(k, iv_m) via G(r, tau) and FFT + irbasis
    # set_v     : p, b | calculates V(r, tau) on fermionic tau via chi_0, FFT + irbasis
    # set_sigma : p, b | calculates Sigma(k, iw_n) via V and G
    ##############
    
    ### Set static quantities
    def set_static_quantities(self, p, h, b):
        self.E_  = tensordot(ones(len(b.fm)), array([eye(h.hk.shape[1],h.hk.shape[2]) for it in range(h.hk.shape[0])]), axes=0)
        self.hk_ = tensordot(ones(len(b.fm)), h.hk, axes=0)
        self.io_ = b.fm.reshape(len(b.fm),1,1,1)*self.E_
        
        #For calculating V
        self.E_int = tensordot(ones(len(b.bm)), array([eye(p.nwan**2,p.nwan**2) for it in range(p.nk)]), axes=0)
 
    
    ### Set G(k, iw_n) --------------------------------------------------------
    def set_gkio(self, p, h, b, mu):
        self.gkio = linalg.inv(self.io_ + mu*self.E_ - self.hk_ - self.sigma)


    def symmetrize_gkio(self,p,b):
        # G(k, iwn) = G^T(-k, iwn) 
        gkio_invk = roll(self.gkio.reshape(len(b.fm),p.nk1,p.nk2,p.nwan,p.nwan),-1,axis=(1,2))[:,::-1,::-1]
        gkio_invk = gkio_invk.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
        self.gkio = (self.gkio + transpose(gkio_invk,axes=(0,1,3,2)))/2


    ### Set G(r, tau) ---------------------------------------------------------
    def set_grit(self, p, b):
        if not hasattr(self, 'gkio'): exit("Error_set_grit")
        grit = self.gkio.reshape(len(b.fm), p.nk1, p.nk2, p.nk3, p.nwan**2)
        
        fft_object = pyfftw.builders.fftn(grit, axes=(1,2,3))
        grit = fft_object()

        grit_0 = grit[:,0,0,0].reshape(len(b.fm),p.nwan**2)
        grit = grit.reshape(len(b.fm),p.nk*p.nwan**2)
        self.grit_b = dot(b.fermi_iw_to_tau_boson, grit).reshape(len(b.bt),p.nk,p.nwan,p.nwan)
        self.grit_f = dot(b.fermi_iw_to_tau,       grit).reshape(len(b.ft),p.nk,p.nwan,p.nwan)
        self.grit_f_0 = dot(b.fermi_iw_to_tau_0, grit_0).reshape(p.nwan,p.nwan)
        

    ### Set chi_0(k, iv_m) ----------------------------------------------------
    def set_ckit(self, p, b):
        if not hasattr(self, 'grit_b'): exit("Error_set_ckit")
        grit_rev = self.grit_b[::-1,:,:,:]    #G_lm(r,beta-tau)
        ckio = einsum2('ijkm,ijln->ijklmn', self.grit_b, grit_rev).reshape(len(b.bt),p.nk*p.nwan**4)#km ln

        ckio = dot(b.bose_tau_to_iw, ckio)
        ckio = ckio.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)
        
        fft_object = pyfftw.builders.ifftn(ckio, axes=(1,2,3))
        ckio = fft_object()/p.nk
        
        self.ckio = ckio.reshape(len(b.bm),p.nk,p.nwan**2,p.nwan**2)
        

    ### V(r, tau) -------------------------------------------------------------
    def set_V(self,p,h,b):
        if p.nspin == 1 and p.nwan >= 2:
            chi_spin   = linalg.inv(self.E_int - self.ckio@h.S_mat)@self.ckio
            chi_charge = linalg.inv(self.E_int + self.ckio@h.C_mat)@self.ckio
            
            V = 3./2.* h.S_mat@(chi_spin   - 1/2*self.ckio)@h.S_mat \
              + 1./2.* h.C_mat@(chi_charge - 1/2*self.ckio)@h.C_mat
        else:
            print('No multiorbital system detected! Use different script...'\
                  , file=open(p.Logstr,'a'))

        V = V.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)
        fft_object = pyfftw.builders.fftn(V, axes=(1,2,3))
        V = fft_object().reshape(len(b.bm),p.nk*p.nwan**4)      
        
        V      = dot(b.bose_iw_to_tau_fermi, V)
        self.V = V.reshape(len(b.ft),p.nk,p.nwan,p.nwan,p.nwan,p.nwan) 

        V_DC = 3./2.*h.S_mat - 1./2.*h.C_mat
        self.V_DC = V_DC.reshape(p.nwan,p.nwan,p.nwan,p.nwan)
        
        
    ### Sigma(k, iw_n) --------------------------------------------------------
    def set_sigma(self,p,b):
        sigma = einsum('ijklmn,ijln->ijkm',self.V, self.grit_f)
        sigma = sigma.reshape(len(b.ft),p.nk*p.nwan**2)
        
        sigma = dot(b.fermi_tau_to_iw, sigma)
        sigma = sigma.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan**2)

        fft_object = pyfftw.builders.ifftn(sigma, axes=(1,2,3))
        sigma = fft_object()/p.nk
        self.sigma = sigma.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
        
        sigma_DC   = einsum2('klmn,ln->km',self.V_DC , self.grit_f_0)*ones((len(b.fm),p.nk,p.nwan,p.nwan))/p.nk
        self.sigma = self.sigma + sigma_DC
        
        
        
    ##############
    # Function for calculating max eig(chi@U_S) for diverging check
    ##############
    
    def Max_eigval_ChiU(self, p, h, channel='S'):        
        '''
        Calculate max{eig(ckio@S_mat)} as a measure for divergence check in chi_spin
        '''
        chan_dic = {'S': h.S_mat, 'C': h.C_mat, 'S_ask' : amax, 'C_ask': amin}
        
        X = self.ckio@chan_dic[channel]
        X = X.reshape(-1, p.nk1, p.nk2, p.nk3, p.nwan**2, p.nwan**2)
        X_eig, _ = linalg.eigh(X)
        return chan_dic[channel+'_ask'](X_eig)
        

        
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
               
        ##### Full G already calculated, load results
        print("Self consistency loop already done. Loading data now...", file=open(p.Logstr,'a'))
        
        # G(iwn_fermi, k)
        with h5py.File(p.loadpath,'r') as file: 
            self.gkio = file['gfunction/gkio'][()]
        print('G(io,k) done', end =" ", file=open(p.Logstr,'a'))
        
        # G(iwn_fermi, -k)
        gkio_invk  = self.gkio.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan**2)
        fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
        gkio_invk  = fft_object()
        fft_object = pyfftw.builders.fftn(gkio_invk,axes=(1,2,3))
        gkio_invk  = fft_object()/p.nk
        self.gkio_invk = gkio_invk.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
        print('| G(io,-k) [calc] done', end =" ", file=open(p.Logstr,'a')) 
        
        # G(tau, r)
        grit = self.gkio.reshape(len(b.fm), p.nk1, p.nk2, p.nk3, p.nwan**2)
        fft_object = pyfftw.builders.fftn(grit, axes=(1,2,3))
        grit = fft_object()
        grit = grit.reshape(len(b.fm),p.nk*p.nwan**2)
        self.grit_b = dot(b.fermi_iw_to_tau_boson, grit).reshape(len(b.bt),p.nk,p.nwan,p.nwan)
        print('| G(tau,r) [calc] done', end =" ", file=open(p.Logstr,'a')) 
        
        # chi_0(iwn_bose, k)
        grit_rev = self.grit_b[::-1,:,:,:]    #G_lm(r,beta-tau)
        ckio = einsum2('ijkm,ijln->ijklmn', self.grit_b, grit_rev).reshape(len(b.bt),p.nk*p.nwan**4)#km ln
        ckio = dot(b.bose_tau_to_iw, ckio)
        ckio = ckio.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)
        fft_object = pyfftw.builders.ifftn(ckio, axes=(1,2,3))
        ckio = fft_object()/p.nk
        self.ckio = ckio.reshape(len(b.bm),p.nk,p.nwan**2,p.nwan**2)
        print('| Chi_0(iw,k) [calc] done', file=open(p.Logstr,'a'))         
        
        ##### Interaction identity
        self.E_int = tensordot(ones(self.ckio.shape[0]), array([eye(p.nwan**2,p.nwan**2) for it in range(p.nk)]), axes=0)
