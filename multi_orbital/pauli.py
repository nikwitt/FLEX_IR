from numpy import *

def sigma(i, j="dummy", k="dummy", l="dummy"): 

    def sigma0(n):
        # pauli matrix
        if   n==0: return array([[1,0],[0,1]])
        elif n==1: return array([[0,1],[1,0]])
        elif n==2: return array([[0,-1j],[1j,0]])
        elif n==3: return array([[1,0],[0,-1]])
        # kronecker delta
        elif n==11: return array([[1,0],[0,0]])
        elif n==12: return array([[0,1],[0,0]])
        elif n==21: return array([[0,0],[1,0]])
        elif n==22: return array([[0,0],[0,1]])

    if   type(j) is str: return sigma0(i) 
    elif type(k) is str: return kron(sigma0(i), sigma0(j))
    elif type(l) is str: return kron(kron(sigma0(i), sigma0(j)),sigma(k))
    else:                return kron(kron(kron(sigma0(i), sigma0(j)),sigma(k)),sigma(l))