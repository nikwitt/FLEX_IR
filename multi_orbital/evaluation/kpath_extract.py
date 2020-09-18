## -*- encoding: latin-1 -*-

from numpy import *
import h5py
    
class kpath_extract:
    def __init__(self,p,h,b,g):
        
        ### G function
        k_HSP, gkio_HSP = kpath_extract.kpath_extractor(p,\
                            (trace(g.gkio[b.f_iwn_zero_ind],0,1,2)/3).reshape(p.nk1,p.nk2,p.nk3))

        ### max eigenvalue spin susceptibility        
        chi_spin   = g.ckio@linalg.inv(g.E_int - g.ckio@h.S_mat)
        chi_s = chi_spin[b.b_iwn_zero_ind].reshape(p.nk1,p.nk2,p.nk3,p.nwan**2,p.nwan**2)
        chi_s_eig, _ = linalg.eigh(chi_s)
        _, chi_s_HSP = kpath_extract.kpath_extractor(p, (chi_s_eig[:,:,:,-1]))
        
        ### max eigenvalue charge susceptibility
        chi_charge = g.ckio@linalg.inv(g.E_int + g.ckio@h.C_mat)
        chi_c = chi_charge[b.b_iwn_zero_ind].reshape(p.nk1,p.nk2,p.nk3,p.nwan**2,p.nwan**2)
        chi_c_eig, _ = linalg.eigh(chi_c)
        _, chi_c_HSP = kpath_extract.kpath_extractor(p, (chi_c_eig[:,:,:,-1]))

# =============================================================================
#         ### gap function
#         _ gap_HSP = kpath_extract.kpath_extractor(p,\
#                             (trace(el.delta[b.f_iwn_zero_ind],0,1,2)/3).reshape(p.nk1,p.nk2,p.nk3))
# =============================================================================
             
        with h5py.File(p.savepath,'a') as file:
            group = file.require_group('kpath_KGMK')
    
            group.require_dataset('kvalue',data=k_HSP,shape=k_HSP.shape,dtype=k_HSP.dtype)
            group.require_dataset('chi_spin_max_eig',data=chi_s_HSP,shape=chi_s_HSP.shape,dtype=chi_s_HSP.dtype)
            group.require_dataset('chi_charge_max_eig',data=chi_c_HSP,shape=chi_c_HSP.shape,dtype=chi_c_HSP.dtype)        
            group.require_dataset('gkio_trace',data=gkio_HSP,shape=gkio_HSP.shape,dtype=gkio_HSP.dtype)
            #group.require_dataset('gap_{}'.format(p.SC_type),data=gap_HSP,shape=gap_HSP.shape,dtype=gap_HSP.dtype)
      
        
    def kpath_extractor(p, quant):
        """
        Extracts points of given quantity along HSP k-path  K->G->M->K.
        """
        ##### Path extraction
        k_HSP_pos = array([0, 4/3, 4/3+2/sqrt(3), 4/3+2/sqrt(3)+2/3])/(4/3+2/3+2/sqrt(3))
        
        ### K->Gamma [Gamma -> K]
        k_HSP_GK  = linspace(k_HSP_pos[0],k_HSP_pos[1],p.nk1//3+1) 
        quant_HSP_GK = zeros((p.nk1//3+1),dtype='complex')
        for it in range(p.nk1//3+1):
            quant_HSP_GK[it] = quant[it,it,0]
        quant_HSP_GK = quant_HSP_GK[::-1]
        
        ### M -> Gamma [Gamma -> M]
        k_HSP_MG  = linspace(k_HSP_pos[1],k_HSP_pos[2],p.nk1//2+1)
        quant_HSP_MG = quant[0:p.nk1//2+1,0,0]
        
        ### K -> M [K ->M]
        k_HSP_KM  = linspace(k_HSP_pos[2],k_HSP_pos[3],p.nk1//6+1)
        quant_HSP_KM = zeros((p.nk1//6+1),dtype='complex')
        for it in range(p.nk1//6+1):
            quant_HSP_KM[it] = quant[p.nk1//2-it,2*it,0]
        
        # Extract along HSP k-line:
        k_HSP     = concatenate(array([k_HSP_GK, k_HSP_MG, k_HSP_KM]))
        quant_HSP = concatenate(array([quant_HSP_GK, quant_HSP_MG, quant_HSP_KM]))

        return k_HSP, quant_HSP
