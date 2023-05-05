# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:46:19 2023

@author: yaxili
"""

import matplotlib.pyplot as plt
import numpy as np

from turbulence_stats_ptv_tools.TurbulenceAdvancedStatsPTVpy import TurbulenceAdvancedStatsPTV

if __name__ == '__main__':
    # name of data folder
    data_folder = r'C:\Users\yaxili\Documents\Work\EMPA_2023_Feb\PTV\trajectory_data\16Hz_fps_125Hz\\'
    # data_folder = r'C:\Users\yaxili\Documents\Work\water_tank\data\20220727_75-90um_PTV\full_data\\'

    # name of mat.file
    file_name = 'smtracks_1.mat'
    # file_name = 'Exp2_71_KS20_sv100.mat'

    # trajectory data name in the mat.file
    data_name = 'smtracks'
    # data_name = 'PTV_tracks'

    # load the trajectory data
    TA = TurbulenceAdvancedStatsPTV(data_folder,file_name,data_name)

    # remove outliers if necessary
    range_u = 0.1
    range_v = 0.1

    TA.RemoveOutlier(range_u,range_v)
    
    '''
    Section to compute 2nd Eulerian Structure Function
    '''    
    # set the parameters for structure function
    p = 2 # order, 2 mean 2nd order SF
    Nx = 100
    Ny = 50 # define the Eulerian field
    full_range = False # use full range of data, "True" use the full range, "False" use the range set by "data_range" 
    data_range = 1000 # how much data used for analysis

    TA.Set_Eulerian_Structure_Parameters(p,Nx,Ny,full_range,data_range)

    num_cores = 5

    separation, SF_long, SF_trans = TA.Eulerian_Structure_Func_main(num_cores)
    
    # power law fitting, inertial range 2/3
    C0 = 8e-3
    power = 2/3
    law_fitting = C0*separation[:20]**power

    # Plot structure function vs r
    plt.figure()
    plt.loglog(separation, SF_long)
    plt.loglog(separation[:20], law_fitting)
    plt.xlabel('Separation distance r [m]')
    plt.ylabel('Structure function S(r)')
    plt.show()
    
    '''
    Section to compute Mean Square Displacement
    ''' 
    # set the parameters for mean square displacement
    dt = 1/125 # [s] time interval between frames
    full_range = False # use full range of data, "True" use the full range, "False" use the range set by "data_range" 
    data_range = 50000 # how much data used for analysis
    
    TA.Set_Mean_Square_Displacement_Parameters(dt, full_range, data_range)
    
    num_cores = 5
    
    lag_t, time_msd = TA.Mean_Sqaure_Dispalcement_main(num_cores)
    
    # Plot structure function vs r
    plt.figure()
    plt.loglog(lag_t,time_msd,'-o')
    plt.xlabel('Time Lag [s]')
    plt.ylabel('MSD')
    plt.show()
    
    '''
    Section to compute Mean Square Separation
    ''' 
    # set the parameters for mean square displacement
    dt = 1/125 # [s] time interval between frames
    full_range = False # use full range of data, "True" use the full range, "False" use the range set by "data_range" 
    data_range = 5000 # how much data used for analysis
    initial_sep_low = 0.01
    initial_sep_high = 0.1
    
    TA.Set_Mean_Square_Separation_Parameters(dt, initial_sep_low, initial_sep_high, full_range, data_range)
    
    num_cores = 5
    
    lag_t, mss = TA.Mean_Sqaure_Separation_main(num_cores)
    
    # Plot structure function vs r
    plt.figure()
    plt.loglog(lag_t[1:],mss[1:],'-o')
    plt.xlabel('Time Lag')
    plt.ylabel('MSS')
    plt.show()
    
    '''
    Section to compute Lagrangian Autocorrelation Function along trajectories
    ''' 
    # set the parameters for mean square displacement
    dt = 1/125 # [s] time interval between frames
    full_range = False # use full range of data, "True" use the full range, "False" use the range set by "data_range" 
    data_range = 5000 # how much data used for analysis
    
    TA.Set_Lagrangian_Correlation_Parameters(dt, full_range, data_range)
    
    num_cores = 5
    
    lag_t, lag_corr_UU, lag_corr_VV = TA.Lagrangian_Correlation_main(num_cores)
    
    # Plot structure function vs r
    plt.figure()
    plt.plot(lag_t,lag_corr_UU)
    plt.plot(lag_t,lag_corr_VV)
    plt.xlabel('Time Lag [s]')
    plt.ylabel('Lagrangian Correlation')
    plt.show()
    
    '''
    Section to compute the Probability Distribution Function of temporal pair separation distance
    ''' 
    # set the parameters for mean square displacement
    full_range = False # use full range of data, "True" use the full range, "False" use the range set by "data_range" 
    data_range = 5000 # how much data used for analysis
    initial_sep_low = 0.01
    initial_sep_high = 0.1
    tp = 5 # how many frames after the initial timing
    
    TA.Set_Pair_Separation_Temporal_PDF_Parameters(full_range, data_range, initial_sep_low, initial_sep_high, tp)
    
    num_cores = 5
    
    separation_dis = TA.Pair_Separation_Temporal_PDF_main(num_cores)
    
    '''
    Section to compute the Separation Timescale Ratio and the probability distribution function of it
    ''' 
    # set the parameters for mean square displacement
    full_range = False # use full range of data, "True" use the full range, "False" use the range set by "data_range" 
    data_range = 500000 # how much data used for analysis
    sep_range_low = 0.01
    sep_range_high = 0.1
    epsilon = 2.8e-4 # dissipation rate
    
    TA.Set_Separation_Timescale_Ratio_PDF_Parameters(full_range, data_range, sep_range_low, sep_range_high, epsilon)
    
    num_cores = 5
    
    Gamma = TA.Separation_Timescale_Ratio_PDF_main(num_cores)
    
    # Compute the histogram of the data
    # hist, bins = np.histogram(~np.isnan(Gamma), bins=100, range=(np.nanmin(Gamma), np.nanmax(Gamma)), density=True)
    hist, bins = np.histogram(Gamma[~np.isnan(Gamma)], bins=100, range=(-10, 10), density=True)

    # Compute the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the histogram and PDF
    # plt.hist(Gamma[~np.isnan(Gamma)], bins=100, density=True, range=(np.nanmin(Gamma), np.nanmax(Gamma)), log=True)
    plt.hist(Gamma[~np.isnan(Gamma)], bins=100, density=True, range=(-10,10), log=True)
    plt.semilogy(bin_centers, hist, linewidth=2)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    

