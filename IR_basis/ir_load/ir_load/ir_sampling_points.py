#Script for calculating and writing best sampling points (from irbasis.sampling_points_...)

import numpy as np
import irbasis

class write_data:
    """
    Calculate and save grid data for irbasis if data does not exist (yet).
    """
    def __init__(self, path_str, b):
        
        self.write_x_smpl(path_str, b)
        self.write_y_smpl(path_str, b)
        self.write_matsubara_smpl(path_str, b)
    
    
    def write_x_smpl(self, path_str, b):
        """
        Calculate and save normalized tau grid data.
        """
        
        Nl = b.dim()
        
        print("Start writing x sample data")
        x_smpl = np.empty(Nl,dtype=object)
        for l in range(Nl):
            print("l = " + str(l))
            x_smpl[l] = irbasis.sampling_points_x(b,l)
            
        
        np.save(path_str.format("x_smpl"),x_smpl)
        print("Done")
            
        
    def write_y_smpl(self, path_str, b):
        """
        Calculate and save normalized omega grid data.
        """
        
        Nl = b.dim()
        
        print("Start writing y sample data")
        y_smpl = np.empty(Nl,dtype=object)
        for l in range(Nl):
            print("l = " + str(l))
            y_smpl[l] = irbasis.sampling_points_y(b,l)
        
        np.save(path_str.format("y_smpl"),y_smpl)
        
        
    def write_matsubara_smpl(self, path_str, b):
        """
        Calculate and save indexed Matsubara frequency grid data.
        """
        
        Nl = b.dim()
        
        print("Start writing matsubara sample data")
        mat_smpl = np.empty(Nl,dtype=object)
        for l in range(Nl):
            print("l = " + str(l))
            mat_smpl[l] = irbasis.sampling_points_matsubara(b,l)
        
        np.save(path_str.format("matsubara_smpl"),mat_smpl)