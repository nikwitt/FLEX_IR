## -*- encoding: latin-1 -*-

from numpy import *
import scipy.optimize as sci_opt
from pauli import sigma as pauli_mat
from matrices3 import mat3

##############
# Set Hamiltonian for calculation
##############

class hamiltonian:
    def __init__(self,p):
        
        self.set_hamiltionan_matrix(p)
        self.ek0, self.uk = linalg.eigh(self.hk)
        self.set_mu(p, self.ek0)
        self.ek = self.ek0 - self.mu*ones(self.ek0.shape)
        self.uk_adj = transpose(self.uk,(0,2,1)).conj()
        self.emin = amin(self.ek0)
        self.emax = amax(self.ek0)
        
        self.set_interaction(p)
    
    def set_hamiltionan_matrix(self,p):
        """
        Here the systems Hamiltonian matrix (in orbital basis) has to be set up.
        """
        
        ### k-vectors in orbital space
        #dimension: 1 = xy, 2 = yz, 3 = zx
        ka = tensordot( p.k1,         mat3(11,13,31),axes=0)\
            +tensordot( p.k2,         mat3(22,12,21),axes=0)\
            -tensordot((p.k1 + p.k2), mat3(33,32,23),axes=0)
        kb = tensordot( p.k1,         mat3(33,32,23),axes=0)\
            +tensordot( p.k2,         mat3(11,13,31),axes=0)\
            -tensordot((p.k1 + p.k2), mat3(22,12,21),axes=0)
            
        ### Model parameters
        t1 = 0.45
        t2 = 0.05
        t3 = 1
        t4 = 0.2
        t5 = -0.15
        t6 = -0.05
        t7 = 0.12
        t8 = 0.12
        t9 = -0.45
        cry_Delta = 0.4
        
        ### H_kin without chemical potential
        Htb11 = 2*t1* cos(2*pi*ka[:,0,0])\
              + 2*t2*(cos(2*pi*kb[:,0,0]) + cos(2*pi*(ka[:,0,0] + kb[:,0,0])) )\
              + 2*t4*(cos(2*pi*(2*ka[:,0,0] + kb[:,0,0])) + cos(2*pi*(ka[:,0,0] - kb[:,0,0])) )\
              + 2*t5* cos(2*2*pi*ka[:,0,0])
        Htb22 = 2*t1* cos(2*pi*ka[:,1,1])\
              + 2*t2*(cos(2*pi*kb[:,1,1]) + cos(2*pi*(ka[:,1,1] + kb[:,1,1])) )\
              + 2*t4*(cos(2*pi*(2*ka[:,1,1] + kb[:,1,1])) + cos(2*pi*(ka[:,1,1] - kb[:,1,1])) )\
              + 2*t5* cos(2*2*pi*ka[:,1,1])             
        Htb33 = 2*t1* cos(2*pi*ka[:,2,2])\
              + 2*t2*(cos(2*pi*kb[:,2,2]) + cos(2*pi*(ka[:,2,2] + kb[:,2,2])) )\
              + 2*t4*(cos(2*pi*(2*ka[:,2,2] + kb[:,2,2])) + cos(2*pi*(ka[:,2,2] - kb[:,2,2])) )\
              + 2*t5* cos(2*2*pi*ka[:,2,2])           
              
        Htb12 = 2*t3*cos(2*pi*kb[:,0,1])\
              + 2*t6*cos(2*2*pi*kb[:,0,1])\
              + 2*t7*cos(2*pi*(  ka[:,0,1] + 2*kb[:,0,1]))\
              + 2*t8*cos(2*pi*(  ka[:,0,1] -   kb[:,0,1]))\
              + 2*t9*cos(2*pi*(2*ka[:,0,1] +   kb[:,0,1]))        
        Htb13 = 2*t3*cos(2*pi*kb[:,0,2])\
              + 2*t6*cos(2*2*pi*kb[:,0,2])\
              + 2*t7*cos(2*pi*(  ka[:,0,2] + 2*kb[:,0,2]))\
              + 2*t8*cos(2*pi*(  ka[:,0,2] -   kb[:,0,2]))\
              + 2*t9*cos(2*pi*(2*ka[:,0,2] +   kb[:,0,2]))
        Htb23 = 2*t3*cos(2*pi*kb[:,1,2])\
              + 2*t6*cos(2*2*pi*kb[:,1,2])\
              + 2*t7*cos(2*pi*(  ka[:,1,2] + 2*kb[:,1,2]))\
              + 2*t8*cos(2*pi*(  ka[:,1,2] -   kb[:,1,2]))\
              + 2*t9*cos(2*pi*(2*ka[:,1,2] +   kb[:,1,2]))
              
        self.htb = tensordot(Htb11, mat3(11),    axes=0)\
                  +tensordot(Htb22, mat3(22),    axes=0)\
                  +tensordot(Htb33, mat3(33),    axes=0)\
                  +tensordot(Htb12, mat3(12,21), axes=0)\
                  +tensordot(Htb13, mat3(13,31), axes=0)\
                  +tensordot(Htb23, mat3(23,32), axes=0)
                  
        self.hcry = cry_Delta/3*(eye(3) - ones((3,3)))
        self.hk = self.htb + self.hcry
        
        #If 2 spin system (without SOC!)
        if p.nspin == 2:
            self.hk = kron(pauli_mat(0),self.hk)
        
        
    
    def set_interaction(self,p):
        self.S_mat = zeros((p.nwan, p.nwan, p.nwan, p.nwan))
        self.C_mat = zeros((p.nwan, p.nwan, p.nwan, p.nwan))
        
        for i in range(p.nwan):
            for j in range(p.nwan):
                for k in range(p.nwan):
                    for l in range(p.nwan):
                        if i == j == k == l:
                            self.S_mat[i,j,k,l] += p.u0
                            self.C_mat[i,j,k,l] += p.u0
                        elif i == k != j == l:
                            self.S_mat[i,j,k,l] +=  p.u0_prime
                            self.C_mat[i,j,k,l] += -p.u0_prime + 2*p.J          
                        elif i == j != k == l:
                            self.S_mat[i,j,k,l] +=  p.J
                            self.C_mat[i,j,k,l] += -p.J + 2*p.u0_prime  
                        elif i == l != j == k:
                            self.S_mat[i,j,k,l] += p.J_prime
                            self.C_mat[i,j,k,l] += p.J_prime

        self.S_mat = self.S_mat.reshape(p.nwan**2, p.nwan**2)
        self.C_mat = self.C_mat.reshape(p.nwan**2, p.nwan**2)        
        
        ##### This MAYBE has to be modified to fit with the order of indices in Hk!!
        #if p.nspin == 2:
        #    sig_S = tensordot(pauli_mat(1),pauli_mat(1),axes=0).reshape(4,4)\
        #          + tensordot(pauli_mat(2),pauli_mat(2),axes=0).reshape(4,4)\
        #          + tensordot(pauli_mat(3),pauli_mat(3),axes=0).reshape(4,4)
        #    sig_C = tensordot(sigma(0),sigma(0),axes=0).reshape(4,4)
        #
        #    self.gamma0 = 1/2*(tensordot(self.C_mat,sig_C,axes=0).reshape(p.nwan**2,p.nwan**2)\
        #                      -tensordot(self.S_mat,sig_S,axes=0).reshape(p.nwan**2,p.nwan**2))
            
        
    def set_mu(self,p,ek):
        """
        Determining inital mu from bisection method of sum(fermi_k,n) (so for all orbitals!)
        """
       
        ### Set electron number for bisection difference
        # n_0 is per orbital
        n_0 = p.n_fill        
        n = self.calc_electron_density
        f = lambda mu : 2/p.nspin*n(p,ek,mu) - n_0
       
        self.mu = sci_opt.bisect(f, amax(ek)+5, amin(ek)-5)
        
        
    def calc_electron_density(self,p,ek,mu):
        
        E = ones(ek.shape)
        n = sum( E / (E + exp(p.beta*(ek - mu)) ) ) / p.nk /p.norb
        return n