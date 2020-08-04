## -*- encoding: latin-1 -*-

from numpy import *

    
class kpath_extract:
    def __init__(self,p,h,b,g):
        
        ### G function
        k_HSP, gkio_HSP = kpath_extract.kpath_extractor(p,\
                            (trace(g.gkio[b.f_iwn_zero_ind],0,1,2)/3).reshape(p.nk1,p.nk2,p.nk3))
        kpath_extract.kpath_save_data(k_HSP, gkio_HSP, p.kpath_savepath.format("gkio"))

        ### max eigenvalue spin susceptibility        
        E_  = tensordot(ones(len(b.bm)), array([eye(p.nwan**2,p.nwan**2) for it in range(p.nk)]), axes=0)
        chi_spin   = g.ckio@linalg.inv(E_ - g.ckio@h.S_mat)
        chi_s = chi_spin[b.b_iwn_zero_ind].reshape(p.nk1,p.nk2,p.nk3,p.nwan**2,p.nwan**2)
        chi_s_eig, _ = linalg.eigh(chi_s)
        
        k_HSP, chi_s_HSP = kpath_extract.kpath_extractor(p, (chi_s_eig[:,:,:,-1]))
        kpath_extract.kpath_save_data(k_HSP, chi_s_HSP, p.kpath_savepath.format("chi_s_maxeig"))       
        
# =============================================================================
#         ### gap function
#         k_HSP, gap_HSP = kpath_extract.kpath_extractor(p,\
#                             (trace(el.delta[b.f_iwn_zero_ind],0,1,2)/3).reshape(p.nk1,p.nk2,p.nk3))
#         kpath_extract.kpath_save_data(k_HSP, gap_HSP, p.kpath_savepath.format("gap_"+p.SC_type+"w"))
# =============================================================================
        
        
    def kpath_extractor(p, quant):
        """
        Extracts points of given quantity along HSP k-path  K->G->M->K.
        """
        ##### Path extraction
        ### K->Gamma [Gamma -> K]
        k_HSP_GK  = linspace(0,1.057,p.nk2//3+1)
        quant_HSP_GK = zeros((p.nk2//3+1),dtype='complex')
        for it in range(p.nk2//3+1):
            quant_HSP_GK[it] = quant[it,it,0]
        quant_HSP_GK = quant_HSP_GK[::-1]

        ### K -> M [K ->M]
        k_HSP_KM  = linspace(1.972,2.5,p.nk2//6+1)
        quant_HSP_KM = zeros((p.nk2//6+1),dtype='complex')
        for it in range(p.nk2//6+1):
            quant_HSP_KM[it] = quant[p.nk2//2-it,2*it,0]

        ### M -> Gamma [Gamma -> M]
        k_HSP_MG  = linspace(1.057,1.972,p.nk2//2+1)
        quant_HSP_MG = quant[0:p.nk1//2+1,0,0]
        
        # Extract along HSP k-line:
        k_HSP  = array([k_HSP_GK, k_HSP_MG, k_HSP_KM])
        quant_HSP = array([quant_HSP_GK, quant_HSP_MG, quant_HSP_KM])
        #k_HSP_pos = array([0, 1.057, 1.972, 2.5])
        return k_HSP, quant_HSP
    
    def kpath_save_data(k, quant, path):
        
        file = open(path, "w")
        for it_path in range(3):
            q_shape = k[it_path].shape
            for it_k in range(q_shape[0]):
                file.write(str(k[it_path][it_k]) +\
                    " " + str(real(quant[it_path][it_k])) +\
                    " " + str(imag(quant[it_path][it_k])) + "\n")
            file.write("\n")
        file.close()

