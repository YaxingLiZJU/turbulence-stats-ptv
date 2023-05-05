# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:01:37 2023

@author: yaxili
"""

import os
import scipy.io
import numpy as np
from scipy.interpolate import griddata
import multiprocessing
from collections import Counter
from scipy.spatial.distance import cdist
import math

class TurbulenceAdvancedStatsPTV:
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
        
    '''
    Section to compute Eulerian Structure Function
    ''' 
    def Set_Eulerian_Structure_Parameters(self, p, Nx, Ny, full_range, data_range):

        # Determine pth order of structure function, e.g., p=2, 2nd order SF
        self.p = p

        # ## Compute Eulerian velocity field from Lagrangian trajectory data
        # # Define the size of the grid
        # Nx = 100   # number of cells in the x-direction
        # Ny = 50   # number of cells in the y-direction

        # Define the spacing between grid points
        dx = (max(self.smtracks[:,0])-min(self.smtracks[:,0]))/Nx   # spacing in the streamwise-direction
        dy = (max(self.smtracks[:,1])-min(self.smtracks[:,1]))/Ny   # spacing in the spanwise-direction

        # Create a 2D Eulerian grid using a matrix
        x = np.arange(0, Nx) * dx   # x-coordinates of the grid points
        y = np.arange(0, Ny) * dy   # y-coordinates of the grid points
        self.X, self.Y = np.meshgrid(x, y)    # create a meshgrid of x and y coordinates

        r = np.zeros((Ny*Nx, 2))   # distance between grids
        r[:,0] = np.ravel(self.X)
        r[:,1] = np.ravel(self.Y)
        r_x = np.ravel(self.X)
        r_y = np.ravel(self.Y)

        # Compute the Euclidean distance between each pair of elements
        self.delta_r_x = r_x - r_x[:, np.newaxis]
        self.delta_r_y = r_y - r_y[:, np.newaxis]
        self.distances_r = np.sqrt(self.delta_r_x**2+self.delta_r_y**2)

        # how many data used for computing
        if full_range == True:
            self.data_range = self.smtracks.shape[0]
        else:
            self.data_range = data_range

        self.frame = set(self.smtracks[:self.data_range,6])
    
    def Eulerian_Structure_func(self,frame):
        """function to be executed in parallel"""
        ind = np.where(self.smtracks[:self.data_range,6] == frame)
        
        # Initialize the Eulerian velocity field to zero
        delta_long = np.zeros(self.delta_r_x.shape[0]-1)   # longitudinal
        delta_trans = np.zeros(self.delta_r_y.shape[0]-1)   # transverse
        
        if ind[0].shape[0] >= 5: # at least 5 particles in one frame
            
            # Define the Lagrangian particle trajectories
            x_traj = self.smtracks[ind[0],0]   # x-positions of the particles
            y_traj = self.smtracks[ind[0],1]   # y-positions of the particles
            u_traj = self.smtracks[ind[0],2]   # x-velocities of the particles
            v_traj = self.smtracks[ind[0],3]   # y-velocities of the particles

            # Interpolate the Lagrangian particle velocities onto the Eulerian grid
            u_grid = griddata((x_traj, y_traj), u_traj, (self.X, self.Y), method='nearest')
            v_grid = griddata((x_traj, y_traj), v_traj, (self.X, self.Y), method='nearest')

            # Assign the interpolated velocities to the Eulerian velocity field
            u = u_grid.flatten()
            v = v_grid.flatten()
            
            delta_u = u - u[:, np.newaxis]
            delta_v = v - v[:, np.newaxis]
            
            delta_long_p = ((delta_u*self.delta_r_x + delta_v*self.delta_r_y)/self.distances_r)**self.p
            delta_trans_p = ((delta_u*self.delta_r_y - delta_v*self.delta_r_x)/self.distances_r)**self.p
            
            for kk in range(1,delta_long_p.shape[0]):
                delta_long[kk-1] = np.nanmean(np.diag(delta_long_p, k=kk))
                delta_trans[kk-1] = np.nanmean(np.diag(delta_trans_p, k=kk))
            
        else:
            delta_long = np.full(delta_long.shape, np.nan)
            delta_trans = np.full(delta_trans.shape, np.nan)
            
        return delta_long, delta_trans 
    
    def Eulerian_Structure_Func_main(self,num_cores):
        
        # create a pool of 5 worker processes
        pool = multiprocessing.Pool(processes=num_cores)
        
        # parallelize the worker function with a range of arguments
        target_function = self.Eulerian_Structure_func
        results = pool.map(target_function, self.frame)
        
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()
        
        # convert the results into two matrix of 'delta_long' and 'delta_trans'
        delta_long = np.stack(results, axis=1)[0]
        delta_trans = np.stack(results, axis=1)[1]

        long_delta_u = np.nanmean(delta_long, axis=0)
        trans_delta_u = np.nanmean(delta_trans, axis=0)

        # Define an array for separation
        Ns = 100
        ds = (max(self.distances_r[0,1:])-min(self.distances_r[0,1:]))/Ns
        separation = np.arange(1, Ns) * ds  

        SF_long = np.zeros_like(separation)
        SF_trans = np.zeros_like(separation)

        for count, sep in enumerate(separation):
            ind_s = np.where(abs(self.distances_r[0,1:]-sep)<=(ds/2))
            if ind_s[0].size > 0:
                SF_long[count] = np.nanmean(long_delta_u[ind_s[0]])
                SF_trans[count] = np.nanmean(trans_delta_u[ind_s[0]])
            else:
                SF_long[count] = np.nan
                SF_trans[count] = np.nan
                
        return separation, SF_long, SF_trans
    
    '''
    Section to compute Mean Square Displacement
    ''' 
    def Set_Mean_Square_Displacement_Parameters(self, dt, full_range, data_range):

        # how many data used for computing
        if full_range == True:
            self.data_range = self.smtracks.shape[0]
        else:
            self.data_range = data_range

        self.traj_ID = set(self.smtracks[:self.data_range,4])
        self.traj_max_length = Counter(self.smtracks[:self.data_range,4]).most_common(1)[0][1]

        # dt = 1/80 # [s]
        self.lag_t = np.arange(self.traj_max_length)*dt
    
    def Mean_Sqaure_Dispalcement_func(self,traj_ID):
        """function to be executed in parallel"""
        ind = np.where(self.smtracks[:self.data_range,4] == traj_ID)
        x_traj = self.smtracks[ind[0],0]
        y_traj = self.smtracks[ind[0],1]
        
        msd = np.full(self.traj_max_length, np.nan)
        max_tau = len(ind[0])
        
        for tau in range(1, max_tau):
            
            # compute the displacement vectors
            dx = x_traj[tau:] - x_traj[:-tau]
            dy = y_traj[tau:] - y_traj[:-tau]
            
            # compute the squared magnitude of the displacement vectors
            msd[tau] = np.nanmean(dx**2 + dy**2)
            
        return msd
    
    def Mean_Sqaure_Dispalcement_main(self,num_cores):
        
        # create a pool of 5 worker processes
        pool = multiprocessing.Pool(processes = num_cores)
        
        # parallelize the worker function with a range of arguments
        target_function = self.Mean_Sqaure_Dispalcement_func
        results = pool.map(target_function, self.traj_ID)
        
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()
        
        # convert the results into matrix 'msd'
        msd = np.stack(results, axis=0)
        
        time_msd = np.nanmean(msd, axis=0)
        
        return self.lag_t, time_msd
    
    '''
    Section to compute Mean Square Separation of particle pairs
    ''' 
    def Set_Mean_Square_Separation_Parameters(self, dt, initial_sep_low, initial_sep_high, full_range, data_range):

        # how many data used for computing
        if full_range == True:
            self.data_range = self.smtracks.shape[0]
        else:
            self.data_range = data_range

        self.traj_ID = set(self.smtracks[:self.data_range,4])
        self.traj_max_length = Counter(self.smtracks[:self.data_range,4]).most_common(1)[0][1]

        # dt = 1/80 # [s]
        self.lag_t = np.arange(self.traj_max_length)*dt
        
        self.frame = set(self.smtracks[:self.data_range,6])
        
        self.initial_sep_low = initial_sep_low
        self.initial_sep_high = initial_sep_high
        
    def Mean_Square_Separation_func(self, frame):
        """function to be executed in parallel"""
        ind = np.where(self.smtracks[:self.data_range,6] == frame)
        separation = np.empty((0,self.traj_max_length))
        
        for ii in range(ind[0].shape[0]-1):
            ind_1 = np.where(self.smtracks[ind[0][ii]:,4] == self.smtracks[ind[0][ii],4])
            traj1 = self.smtracks[ind[0][ii]+ind_1[0],:2]
            for jj in range(ii+1, ind[0].shape[0]):
                ind_2 = np.where(self.smtracks[ind[0][jj]:,4] == self.smtracks[ind[0][jj],4])
                traj2 = self.smtracks[ind[0][jj]+ind_2[0],:2]
                
                # compute the number of time steps in the shorter trajectory
                n_steps = min(traj1.shape[0], traj2.shape[0])
                
                # compute the separation vectors between all pairs of particles at each time step
                separation_ij = np.full(self.traj_max_length,np.nan)
                
                sep_initial = np.sqrt((traj2[0,0] - traj1[0,0])**2 + (traj2[0,1] - traj1[0,1])**2)
                
                if sep_initial >= self.initial_sep_low and sep_initial <= self.initial_sep_high:
                    
                    separation_ij[:n_steps] = (np.sqrt((traj2[:n_steps,0]-traj1[:n_steps,0])**2+(traj2[:n_steps,1]-traj1[:n_steps,1])**2)-sep_initial)**2
                    
                    separation = np.vstack((separation,separation_ij))
                        
        separation = np.nanmean(separation, axis=0)
            
        return separation
    
    def Mean_Sqaure_Separation_main(self,num_cores):
        
        # create a pool of 5 worker processes
        pool = multiprocessing.Pool(processes = num_cores)
        
        # parallelize the worker function with a range of arguments
        target_function = self.Mean_Square_Separation_func
        results = pool.map(target_function, self.frame)
        
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()
        
        # convert the results into a matrix of 'separation_square'
        separation_square = np.stack(results, axis=0)
        
        # compute the average MSS over all trajectory pairs and time steps
        mss = np.nanmean(separation_square, axis=0)
        
        return self.lag_t, mss
    
    '''
    Section to compute the Lagrangian autocorrelation along trjectories
    ''' 
    def Set_Lagrangian_Correlation_Parameters(self, dt, full_range, data_range):

        # how many data used for computing
        if full_range == True:
            self.data_range = self.smtracks.shape[0]
        else:
            self.data_range = data_range

        self.traj_ID = set(self.smtracks[:self.data_range,4])
        self.traj_max_length = Counter(self.smtracks[:self.data_range,4]).most_common(1)[0][1]

        # dt = 1/80 # [s]
        self.lag_t = np.arange(self.traj_max_length)*dt
        
        self.u_mean = np.mean(self.smtracks[:self.data_range,2])
        self.v_mean = np.mean(self.smtracks[:self.data_range,3])
        
    def Lagrangian_Correlation_func(self, traj_ID):
        """
        Compute the Lagrangian correlation function of particle positions in a Lagrangian trajectory
        """
        
        ind = np.where(self.smtracks[:self.data_range,4] == traj_ID)
        u_traj = self.smtracks[ind[0],2]
        v_traj = self.smtracks[ind[0],3]
        
        lag_diff_uu = np.full(self.traj_max_length, np.nan)
        lag_diff_vv = np.full(self.traj_max_length, np.nan)
        var_x = np.full(self.traj_max_length, np.nan)
        var_y = np.full(self.traj_max_length, np.nan)
        n = len(ind[0])
        
        for i in range(n):
            
            # substract the trajectory mean
            lag_diff_uu[i] = np.sum((u_traj[i:] - self.u_mean) * (u_traj[:n-i] - self.u_mean))
            lag_diff_vv[i] = np.sum((v_traj[i:] - self.v_mean) * (v_traj[:n-i] - self.v_mean))
            var_x[i] = np.sum((u_traj-self.u_mean)**2)
            var_y[i] = np.sum((v_traj-self.v_mean)**2)
            
        return lag_diff_uu, lag_diff_vv, var_x, var_y
    
    def Lagrangian_Correlation_main(self,num_cores):
        
        # create a pool of 5 worker processes
        pool = multiprocessing.Pool(processes = num_cores)
        
        # parallelize the worker function with a range of arguments
        target_function = self.Lagrangian_Correlation_func
        results = pool.map(target_function, self.traj_ID)
        
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()
        
        # convert the results into two matrix of 'lag_corr_UU' and 'lag_corr_VV'
        lag_diff_UU = np.stack(results, axis=1)[0]
        lag_diff_VV = np.stack(results, axis=1)[1]
        var_X = np.stack(results, axis=1)[2]
        var_Y = np.stack(results, axis=1)[3]
        
        lag_corr_UU = np.nansum(lag_diff_UU, axis=0)/np.nansum(var_X, axis=0)
        lag_corr_VV = np.nansum(lag_diff_VV, axis=0)/np.nansum(var_Y, axis=0)
        
        return self.lag_t, lag_corr_UU, lag_corr_VV
    
    '''
    Section to compute the Probability Distribution Function of temporal pair separation distance
    ''' 
    def Set_Pair_Separation_Temporal_PDF_Parameters(self, full_range, data_range, initial_sep_low, initial_sep_high, tp):

        # how many data used for computing
        if full_range == True:
            self.data_range = self.smtracks.shape[0]
        else:
            self.data_range = data_range

        self.low_range = initial_sep_low 
        self.high_range = initial_sep_high
        self.tp = tp

        self.frame = set(self.smtracks[:self.data_range,6])
        
    def Pair_Separation_Temporal_PDF_func(self, frame):
        """function to be executed in parallel"""
        
        ind = np.where(self.smtracks[:self.data_range,6] == frame)
        
        positions = np.vstack((self.smtracks[ind[0],0], self.smtracks[ind[0],1])).T
        
        dis_t0 = np.triu(cdist(positions, positions))
        
        indices = np.argwhere((dis_t0 >= self.low_range) & (dis_t0 <= self.high_range))
        
        dis_tp = np.full(indices.shape[0], np.nan)
        
        for count, ii in enumerate(indices[:,0]):
            ind1 = ind[0][ii]
            ind2 = ind[0][indices[count,1]]
            traj_id1 = np.where(self.smtracks[ind1:self.data_range,4] == self.smtracks[ind1,4])
            traj_id2 = np.where(self.smtracks[ind2:self.data_range,4] == self.smtracks[ind2,4])
            
            t_range =  min(len(traj_id1[0]),len(traj_id2[0]))
            
            if t_range >= self.tp:
                dis_tp[count] = math.sqrt((self.smtracks[ind1+self.tp,0]-self.smtracks[ind2+self.tp,0])**2+(self.smtracks[ind1+self.tp,1]-self.smtracks[ind2+self.tp,1])**2)
                
        return dis_tp
    
    def Pair_Separation_Temporal_PDF_main(self,num_cores):
        
        # create a pool of 5 worker processes
        pool = multiprocessing.Pool(processes = num_cores)
        
        # parallelize the worker function with a range of arguments
        target_function = self.Pair_Separation_Temporal_PDF_func
        results = pool.map(target_function, self.frame)
        
        # convert the results into two matrix of 'lag_corr_UU' and 'lag_corr_VV'
        max_len = max(len(arr) for arr in results[:])
        
        separation_dis = np.full((len(results[:]),max_len), np.nan)
        
        for ii in range(len(results[:])):
            separation_dis[ii,:results[ii].shape[0]] = results[ii]
        
        return separation_dis
    
    '''
    Section to compute the Separation Timescale Ratio and the probability distribution function of it
    ''' 
    def Set_Separation_Timescale_Ratio_PDF_Parameters(self, full_range, data_range, sep_range_low, sep_range_high, epsilon):

        # how many data used for computing
        if full_range == True:
            self.data_range = self.smtracks.shape[0]
        else:
            self.data_range = data_range

        self.low_range = sep_range_low 
        self.high_range = sep_range_high

        self.epsilon = epsilon

        self.frame = set(self.smtracks[:self.data_range,6])
        
    def Separation_Timescale_Ratio_PDF_func(self, frame):
        """function to be executed in parallel"""
        
        ind = np.where(self.smtracks[:self.data_range,6] == frame)
        
        positions = np.vstack((self.smtracks[ind[0],0], self.smtracks[ind[0],1])).T
        
        dis_t0 = np.triu(cdist(positions, positions))
        
        indices = np.argwhere((dis_t0 >= self.low_range) & (dis_t0 <= self.high_range))
        
        gamma = np.full(indices.shape[0], np.nan)
        
        ind1 = ind[0][indices[:,0]]
        ind2 = ind[0][indices[:,1]]
        
        delta_u = self.smtracks[ind1,2] - self.smtracks[ind2,2]
        delta_v = self.smtracks[ind1,3] - self.smtracks[ind2,3]
        delta_x = self.smtracks[ind1,0] - self.smtracks[ind2,0]
        delta_y = self.smtracks[ind1,1] - self.smtracks[ind2,1]
        
        r0 = np.sqrt(delta_x**2 + delta_y**2)
        v0 = (delta_u*delta_x + delta_v*delta_y)/r0
        
        gamma = abs(v0)*v0/(r0*self.epsilon)**(2/3)
                
        return gamma
    
    def Separation_Timescale_Ratio_PDF_main(self,num_cores):
        
        # create a pool of 5 worker processes
        pool = multiprocessing.Pool(processes = num_cores)
        
        # parallelize the worker function with a range of arguments
        target_function = self.Separation_Timescale_Ratio_PDF_func
        results = pool.map(target_function, self.frame)
        
        # convert the results into two matrix of 'lag_corr_UU' and 'lag_corr_VV'
        max_len = max(len(arr) for arr in results[:])
        
        Gamma = np.full((len(results[:]),max_len), np.nan)
        
        for ii in range(len(results[:])):
            Gamma[ii,:results[ii].shape[0]] = results[ii]
            
        return Gamma
        
    


        
            
            
            