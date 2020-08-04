# -*- encoding: latin-1 -*-

from numpy import *

def mat3(i, j="dummy", k="dummy"):
    def generator(n):
        if   n==11: return array([[1,0,0],[0,0,0],[0,0,0]])
        elif n==12: return array([[0,1,0],[0,0,0],[0,0,0]])
        elif n==13: return array([[0,0,1],[0,0,0],[0,0,0]])
        elif n==21: return array([[0,0,0],[1,0,0],[0,0,0]])
        elif n==22: return array([[0,0,0],[0,1,0],[0,0,0]])
        elif n==23: return array([[0,0,0],[0,0,1],[0,0,0]])
        elif n==31: return array([[0,0,0],[0,0,0],[1,0,0]])
        elif n==32: return array([[0,0,0],[0,0,0],[0,1,0]])
        elif n==33: return array([[0,0,0],[0,0,0],[0,0,1]])
        
    if   type (j) is str: return generator(i)
    elif type (k) is str: return generator(i) + generator(j)
    else:                 return generator(i) + generator(j) + generator(k)
    