#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:22:31 2020

@author: nwitt
"""

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.ticker as tick
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
            
            
class Hexagonal_BZ_plot:
    def __init__(self, p ,quantity, title='', xlab='', ylab='',\
                title_fontsize = 18, cbar_labelsize = 12,\
                line_width=1.8, alpha=1, levels=20, cmap = 'plasma',\
                subplots=False, show_grid=True, fig_size=(7,6),
                save_name = ''):


        ### Define k-vector components
        kx = 1/sqrt(3)*(2*p.k1+p.k2).reshape(p.nk1,p.nk2)
        ky = p.k2.reshape(p.nk1,p.nk2)
        

        ### Plot quantity
        fig, ax = plt.subplots(figsize=fig_size)
        for it in range(2):
            for it2 in range(2):
               cf = ax.contourf(kx - (it+it2/2)*max(kx[0]), ky - it2*max(ky[-1]),\
               quantity.reshape(p.nk1,p.nk2),\
               cmap = cmap, levels = levels)
                  

        ### Plot BZ border
        # Rotation matrix
        R = array([[cos(pi/3),-sin(pi/3)],[sin(pi/3),cos(pi/3)]])
        
        # HSP in k space
        K = array([kx[p.nk1//3,p.nk1//3], ky[p.nk1//3,p.nk1//3]])
        M = array([kx[p.nk1//2,0],        ky[p.nk1//2,0]])
        
        # Hide everything right/left to M
        ax.fill_between( array([kx[p.nk1//3,p.nk1//3],2*kx[p.nk1//3,p.nk1//3]]),\
                         array([-1,-1]), array([1,1]), color='w')
        ax.fill_between(-array([kx[p.nk1//3,p.nk1//3],2*kx[p.nk1//3,p.nk1//3]]),\
                         array([-1,-1]), array([1,1]), color='w')
        
        for it in range(6):
            
            K = K@R
            M = M@R
            
            # Hide everything outside of BZ
            ax.fill_between( array([K[0],M[0]]), array([K[1],M[1]]), array([sign(K[1]),sign(K[1])]), color='w')
            ax.fill_between(-array([K[0],M[0]]), array([K[1],M[1]]), array([sign(K[1]),sign(K[1])]), color='w')
            # Plotting BZ lines
            ax.plot( array([K[0],M[0]]), array([K[1],M[1]]),'k', linewidth = line_width)
            ax.plot(-array([K[0],M[0]]), array([K[1],M[1]]),'k', linewidth = line_width)
            
            # Setting ticks
            tlen = 0.615
            yK = sign(K[1])*(sqrt(tlen**2/((K[0]/K[1])**2+1)))
            xK = sign(K[0])*sqrt(tlen**2 - yK**2)
            plt.plot(array([K[0],xK]),array([K[1],yK]),'k')
            tlen = 0.54
            yM = sign(M[1])*sqrt(tlen**2/((M[0]/M[1])**2+1))
            xM = sign(M[0])*(sqrt(tlen**2 - yM**2))
            plt.plot(array([xM,M[0]]),array([yM,M[1]]),'k')
        
        # Tick labels
        M = M@R
        plt.axis('off')
        #plt.text(-0.02, -0.05, '$\\Gamma$', fontsize=22)
        plt.text(K[0]*1.02, K[1]*1.02, 'K', fontsize=22)
        plt.text(M[0]*1.02, M[1]-0.03, 'M', fontsize=22)       
        
        
        ### Plot details
        sfmt= ticker.ScalarFormatter(useMathText=True) 
        sfmt.set_powerlimits((0, 0))
        cbar = fig.colorbar(cf, format=sfmt)
        cbar.ax.yaxis.offsetText.set_fontsize(cbar_labelsize+4)
        cbar.ax.xaxis.get_offset_text().set_position((2,0))
        cbar.ax.tick_params(labelsize=cbar_labelsize)
        cbar.update_ticks()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(-M[0]-0.01, M[0]+0.01)
        ax.set_ylim(-(R@K)[1]-0.001, (R@K)[1]+0.001)
        ax.set_title(title, fontsize=title_fontsize)
        
        plt.tight_layout()
        
        #fig.savefig(save_name)
        
        #plt.close()