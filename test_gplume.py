# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:54:11 2021

@author: kangsun
"""
import numpy as np
import matplotlib.pyplot as plt
# https://github.com/Kang-Sun-CfA/arpae/blob/main/area_Gaussian_plume.py
from area_Gaussian_plume import checkerboard_polygons
# create an instance of square sources aligned in a checkerboard pattern
# centering at x = -1, 0, 1 km and y = 1, 0, -1 km with width = 0.75 km
sources = checkerboard_polygons(field_center_x=[-1,0,1],
                                field_center_y=[1,0,-1],
                                field_width=0.75)
# height of receptors above ground in m
sensor_height = 2 
# wind speed in m/s
wind_speed = 5
# wind direction. 0/360: north; 90: east; 270: west; 180: south 
wind_direction = 80 
# Pasquill stability class. See, for example, https://www.ready.noaa.gov/READYpgclass.php
plume_class='D'
# flux in each source polygon, 1 mass unit/m2/s for all sources here
beta_true=np.array([1,1,1,1,1,1,1,1,1])
# number of unknowns, i.e., length of the state vector
nn = len(beta_true)
# min/max color scale in concentration plots, in mass unit/m3
vmin = 0
vmax = 20
# define domain to plot Gaussian plumes. 100 points from -2 to 2 km
xgrid = np.linspace(-2,2,100)
ygrid = np.linspace(-2,2,100)
# define full coordinates of the domain
xmesh,ymesh = np.meshgrid(xgrid,ygrid)
zmesh = np.ones_like(xmesh)*sensor_height
# generate concentration field from all sources, no need jacobians
C,_ = sources.evaluate_gplume(field_fluxes=beta_true,
                        xmesh=xmesh,ymesh=ymesh,zmesh=zmesh,
                        wind_speed=wind_speed,wind_direction=wind_direction,
                        plume_class=plume_class,save_jacobian=False)
#%% plot the plumes
fig,ax = plt.subplots(1,1,figsize=(10,10))
pc = ax.pcolormesh(xmesh,ymesh,C,vmin=vmin,vmax=vmax,shading='auto')
sources.plot_polygons(ax=ax,color='k')

ax.set_xlabel('x coordinate [km]')
ax.set_ylabel('y coordinate [km]')
cb = fig.colorbar(pc,ax=ax,label='Concentration [mass unit/m3]',shrink=0.5)
#%%
# place sensors on the map
sensor_xs = np.hstack((-1.5*np.ones(6),np.linspace(-1.5,1.5,7)))
sensor_ys = np.hstack((np.linspace(1.5,-1.5,7),-1.5*np.ones(6)))
# initialize synthetic observation vector
y_obs = np.zeros_like(sensor_xs)
# number of observations
mm = len(sensor_xs)
# initialize jacobian matrix
K = np.zeros((mm,nn))
for i,(x,y) in enumerate(zip(sensor_xs,sensor_ys)):
    y_obs[i],jac = sources.evaluate_gplume(field_fluxes=beta_true,
                                           xmesh=x,ymesh=y,zmesh=sensor_height,
                                           wind_speed=wind_speed,wind_direction=wind_direction,
                                           plume_class=plume_class,save_jacobian=True)
    K[i,:] = jac
#%% plot the synthetic observations
fig,ax = plt.subplots(1,1,figsize=(10,10))
sources.plot_polygons(ax=ax,color='k')
ax.scatter(sensor_xs,sensor_ys,s=100,c=y_obs,vmin=vmin,vmax=vmax,edgecolor='k')
ax.set_xlabel('x coordinate [km]')
ax.set_ylabel('y coordinate [km]')
cb = fig.colorbar(pc,ax=ax,label='Concentration [mass unit/m3]',shrink=0.5)