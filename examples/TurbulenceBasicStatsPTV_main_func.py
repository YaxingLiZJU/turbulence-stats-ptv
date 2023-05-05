# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:19:07 2023

@author: yaxili
"""

import matplotlib.pyplot as plt

from turbulence_stats_ptv_tools.TurbulenceBasicStatsPTVpy import TurbulenceBasicStatsPTV

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
TB = TurbulenceBasicStatsPTV(data_folder,file_name,data_name)

# remove outliers if necessary
range_u = 0.1
range_v = 0.1

TB.RemoveOutlier(range_u,range_v)

# get global mean and rms velocities
umean, vmean, u_rms, v_rms, u_f, v_f = TB.GlobalMeanRMSvelocity()

# get the PDF of fluctuated velocities and acceleration
bin_num = 500 # number of bins for computing P.D.F

step_u_f, pdf_u_f, step_v_f, pdf_v_f, step_a_u, pdf_a_u, step_a_v, pdf_a_v = TB.PdfFluctuationVel(u_f,v_f,bin_num)

# Plot the PDF of velocity fluctuations
plt.figure(1)
plt.semilogy(step_u_f, pdf_u_f,'o')
plt.semilogy(step_v_f, pdf_v_f,'o')
plt.xlabel('u_f/v_f')
plt.ylabel('P.D.F.')
plt.title('Probability Distribution Function')
plt.show()

# Plot the PDF of accelerations
plt.figure(2)
plt.semilogy(step_a_u, pdf_a_u,'o')
plt.semilogy(step_a_v, pdf_a_v,'o')
plt.xlim(-2,2)
plt.xlabel('a_u/a_v')
plt.ylabel('P.D.F.')
plt.title('Probability Distribution Function')
plt.show()

# get the Eulerian mean and rms velocity fields
Nx = 100
Ny = 50
intepolate_method = 'linear' # 'linear','nearest','cubic'

ug_mean, vg_mean, ug_rms, vg_rms, dx, dy = TB.EulerianMeanRMS(Nx,Ny,intepolate_method)

# Create a colormap of the mean velocity field
fig, ax = plt.subplots()
im = ax.imshow(vg_mean, cmap='hot', vmin=-0.03, vmax=0.03, origin='lower', extent=[0, Nx*dx, 0, Ny*dy])
cbar = fig.colorbar(im)
cbar.set_label('Mean Spanwise Mean Velocity')
# cbar.set_label('Mean Spanwise Velocity', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlim([dx, (Nx-1)*dx])
ax.set_ylim([dy, (Ny-1)*dy])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('2D Eulerian Mean Velocity Field')
plt.show()

# Create a colormap of the magnitude of the velocity field
fig, ax = plt.subplots()
im = ax.imshow(vg_rms, cmap='hot', vmin=0, vmax=0.02, origin='lower', extent=[0, Nx*dx, 0, Ny*dy])
cbar = fig.colorbar(im)
cbar.set_label('Mean Spanwise R.M.S. Velocity')
# cbar.set_label('Mean Spanwise Velocity', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlim([dx, (Nx-1)*dx])
ax.set_ylim([dy, (Ny-1)*dy])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('2D Eulerian R.M.S. Velocity Field')
plt.show()

















        

