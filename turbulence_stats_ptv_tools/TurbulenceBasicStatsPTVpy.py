# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:36:09 2023

@author: yaxili
"""

import os
import scipy.io
import numpy as np
import scipy.stats as stats
from scipy.interpolate import griddata

class TurbulenceBasicStatsPTV:
    '''
    Class for computing various of turbulence statics from PTV data
    '''
    
    def __init__(self,data_folder,file_name,data_name):
        # Get the current working directory
        cwd = os.getcwd()

        # Construct the path to the .mat file in the subfolder
        file_path = os.path.join(cwd, data_folder, file_name)

        # Load the MATLAB .mat file
        data = scipy.io.loadmat(file_path)

        # Access the variables in the data dictopnary
        self.smtracks = data[data_name]
        
    def RemoveOutlier(self,range_u,range_v):
        # range_u, range_v is the range of velocity fluctuation

        # Get the raw global mean of streamwise and spanwise velocities
        umean = np.nanmean(self.smtracks[:,2])
        vmean = np.nanmean(self.smtracks[:,3])

        ## Remove the outliers
        # remove the outliers in velocity matrix
        indx_broken = np.where(abs(self.smtracks[:,2]-umean) > range_u)
        indy_broken = np.where(abs(self.smtracks[:,3]-vmean) > range_v)
        ind_broken = np.union1d(indx_broken[0],indy_broken[0])
        self.smtracks[ind_broken,:] = np.nan

        # remove the nan_rows from the original array
        nan_rows = np.isnan(self.smtracks).any(axis=1)
        self.smtracks = self.smtracks[~nan_rows]
        
    def GlobalMeanRMSvelocity(self):
        
        # Get the global mean of streamwise and spanwise velocities with nonNaN smtracks
        umean = np.nanmean(self.smtracks[:,2])
        vmean = np.nanmean(self.smtracks[:,3])

        # Calculate the r.m.s. velocity
        u_rms = np.sqrt(np.mean(np.square(self.smtracks[:,2]-umean)))
        v_rms = np.sqrt(np.mean(np.square(self.smtracks[:,3]-vmean)))

        # velocity fluctuations
        u_f = self.smtracks[:,2]-umean
        v_f = self.smtracks[:,3]-vmean
        
        return umean, vmean, u_rms, v_rms, u_f, v_f
    
    def PdfFluctuationVel(self,u_f,v_f,bin_num):
        # bin_num = 500

        # Create a histogram of the data to estimate the probability density function
        hist_u_f, bin_edges_u_f = np.histogram(u_f, bins=bin_num, density=True)
        hist_v_f, bin_edges_v_f = np.histogram(v_f, bins=bin_num, density=True)

        # Define the probability distribution function using the estimated PDF
        pdf_u_f = stats.rv_histogram((hist_u_f, bin_edges_u_f))
        pdf_v_f = stats.rv_histogram((hist_v_f, bin_edges_v_f))

        # Generate some values from the PDF for plotting
        step_u_f = np.linspace(min(u_f), max(u_f), bin_num)
        pdf_u_f = pdf_u_f.pdf(step_u_f)
        step_v_f = np.linspace(min(u_f), max(v_f), bin_num)
        pdf_v_f = pdf_v_f.pdf(step_v_f)

        # accelerations 
        a_u = self.smtracks[:,7]
        a_v = self.smtracks[:,8]

        # Create a histogram of the data to estimate the probability density function
        hist_a_u, bin_edges_a_u = np.histogram(a_u, bins=bin_num, density=True)
        hist_a_v, bin_edges_a_v = np.histogram(a_v, bins=bin_num, density=True)

        # Define the probability distribution function using the estimated PDF
        pdf_a_u = stats.rv_histogram((hist_a_u, bin_edges_a_u))
        pdf_a_v = stats.rv_histogram((hist_a_v, bin_edges_a_v))

        # Generate some values from the PDF for plotting
        step_a_u = np.linspace(min(a_u), max(a_u), bin_num)
        pdf_a_u = pdf_a_u.pdf(step_a_u)
        step_a_v = np.linspace(min(a_v), max(a_v), bin_num)
        pdf_a_v = pdf_a_v.pdf(step_a_v)
        
        return step_u_f, pdf_u_f, step_v_f, pdf_v_f, step_a_u, pdf_a_u, step_a_v, pdf_a_v
        
    def EulerianMeanRMS(self,Nx,Ny,intepolate_method):
        
        # intepolate_method='linear'

        ## Compute Eulerian mean and r.m.s. velocity field from Lagrangian trajectory data
        # Define the size of the grid
        # Nx = 100   # number of cells in the x-direction
        # Ny = 50    # number of cells in the y-direction

        # Define the spacing between grid points
        dx = (max(self.smtracks[:,0])-min(self.smtracks[:,0]))/Nx   # spacing in the streamwise-direction
        dy = (max(self.smtracks[:,1])-min(self.smtracks[:,1]))/Ny   # spacing in the spanwise-direction

        # Create a 2D Eulerian grid using a matrix
        x = np.arange(0, Nx) * dx   # x-coordinates of the grid points
        y = np.arange(0, Ny) * dy   # y-coordinates of the grid points
        X, Y = np.meshgrid(x, y)    # create a meshgrid of x and y coordinates

        # Define the Lagrangian particle trajectories
        x_traj = self.smtracks[:,0]   # x-positions of the particles
        y_traj = self.smtracks[:,1]   # y-positions of the particles
        u_traj = self.smtracks[:,2]   # x-velocities of the particles
        v_traj = self.smtracks[:,3]   # y-velocities of the particles

        # Use interpolation method to get Eulerian fields
        # Initialize the Eulerian velocity field to zero
        u = np.zeros((Ny, Nx))   # x-velocity
        v = np.zeros((Ny, Nx))   # y-velocity

        # Interpolate the Lagrangian particle velocities onto the Eulerian grid
        u_grid = griddata((x_traj, y_traj), u_traj, (X, Y), method=intepolate_method)
        v_grid = griddata((x_traj, y_traj), v_traj, (X, Y), method=intepolate_method)

        # Assign the interpolated velocities to the Eulerian velocity field
        u[:,:] = u_grid[:,:]
        v[:,:] = v_grid[:,:]

        # Setup the matrix for Eulerian velocity fields
        ug_mean = np.zeros((Ny, Nx))
        vg_mean = np.zeros((Ny, Nx)) # Eulerian mean velocity

        ug_rms = np.zeros((Ny, Nx))
        vg_rms = np.zeros((Ny, Nx)) # Eulerian r.m.s. velocity

        for i in range(Ny-1):
            for j in range(Nx-1):
                indices = np.where((x_traj >= X[i,j]) & (x_traj < X[i,j+1]) & (y_traj >= Y[i,j]) & (y_traj < Y[i+1,j]))[0]
                if len(indices) > 0:
                    ug_mean[i,j] = np.mean(u_traj[indices])
                    vg_mean[i,j] = np.mean(v_traj[indices])
                    ug_rms[i,j] = np.sqrt(np.mean(np.square(u_traj[indices]-ug_mean[i,j])))
                    vg_rms[i,j] = np.sqrt(np.mean(np.square(v_traj[indices]-vg_mean[i,j])))
                    
        return ug_mean, vg_mean, ug_rms, vg_rms, dx, dy
