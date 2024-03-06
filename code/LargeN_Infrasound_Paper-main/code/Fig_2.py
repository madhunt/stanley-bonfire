# Fig.2 Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import pickle
import numpy as np


#%% ADDED BY MADELINE
import os
path_curr = os.path.dirname(os.path.realpath(__file__))
path_home = os.path.abspath(os.path.join(path_curr, '..'))
print(path_home)




#%% Imports infrasound data
# 24 hour Data
def datelist(r1, r2):
    return [item for item in range(r1, r2+1)]

r1, r2 = 20200415, 20200430
date1 = datelist(r1, r2)
r3, r4 = 20200501, 20200531
r5, r6 = 20200601, 20200611
date2 = datelist(r3,r4)
date3 = datelist(r5,r6)
dates = date1 + date2 + date3
dates = list(map(str,dates))
data_list = []
#
for date in dates:
    with open(path_home+'/pickle_files/N_full/'+ date + '_0600all_day.pkl','rb') as f:
        data_list.append(pickle.load(f))
        
#Imports 6_element data
data_list_6 = []
dates_6 = ['20200511','20200519','20200522','20200524']
for days in dates_6:
    with open(path_home+'/pickle_files/N_6/'+days+'_0600_N_6.pkl','rb') as z:
        data_list_6.append(pickle.load(z))
for d in data_list_6:
    for q in range(len(d)):
        if d[q,3] <=0:
            d[q,3] = (d[q,3] + 360)

#%% Multi-day data processing
for data in data_list: # Converts from (-)180-180 to 0-360
    for i in range(17279):
        if data[i,3] <= -2.5:
            data[i,3] = data[i,3] + 360
            
day_index = 1
for data in data_list:
    bins_all = []
    slowness_all = []
    bin_centers = np.arange(0,360,2.5) # 2.5 degree bins
    for bin_center in bin_centers:
        bin_1 = np.sum((data[:,3] > (bin_center-1.25)) & (data[:,3] <= (bin_center+1.25)))
        bins_all.append(bin_1)
    
    slow_centers = np.arange(0,4.2,.2)
    for slow in slow_centers:
        slow_bin1 = np.sum((data[:,4]> (slow-.1)) & (data[:,4]<= (slow+.1))) / (slow + .1)
        slowness_all.append(slow_bin1)
    
    slowness_all = np.array(slowness_all)
    slowness_all = np.reshape(slowness_all,(21,1))
    bins_all = np.array(bins_all)
    bins_all = np.reshape(bins_all,(144,1))
    if day_index == 1:
        detections_list = bins_all
        slowness_detection = slowness_all
        
    else:
        detections_list = np.concatenate((detections_list, bins_all), axis=1) 
        slowness_detection = np.concatenate((slowness_detection, slowness_all),axis=1)
    day_index = day_index + 1
    
detections_list = np.swapaxes(detections_list,0,1)
slowness_detection = np.swapaxes(slowness_detection,0,1)

# Output: detections_list
# detections_list is a 58 x 71 numpy array, with the x axis representing binned backazimuth values,
# with row index=0 corresponding with 0 - 5 deg, and index=71 corresponding with 355 - 360 degrees. 
# y axis values represent time in days. Container values are # of points detected. 
            
#%% Backazimuth colormap
wyor = LinearSegmentedColormap('wyor', {'red': [[0, 1, 1], 
                                                [1, 1, 1]],
                                        'green': [[0, 1, 1],
                                                  [0.075, 1, 1],
                                                  [0.25, 1, 1],
                                                  [0.925, 0, 0],
                                                  [1, 0, 0]],
                                        'blue': [[0, 1, 1],
                                                 [0.075, 1, 1],
                                                 [0.25, 0, 0],
                                                 [0.925, 0, 0],
                                                 [1, 0, 0]]
                                        })

def image(Z, x = None, y = None, aspect = 'equal', zmin = None, zmax = None, ax = plt, crosshairs=False):
    # Z rows are x, columns are y
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])
    #im = ax.imshow(Z.transpose(), extent = [x[0], x[-1], y[0], y[-1]], aspect = aspect, 
    #           origin = 'lower', vmin = zmin, vmax = zmax, cmap = 'YlOrRd')
    im = ax.pcolormesh(x, y, Z.T, norm=colors.LogNorm(), vmin = zmin, vmax = zmax, cmap= wyor, shading = 'auto')
    if crosshairs:
        ax.hlines(0, x[0], x[-1], 'k', linewidth=0.5)
        ax.vlines(0, y[0], y[-1], 'k', linewidth=0.5)
    return im
#%% Slowness space colormap
def image_bg(Z, x = None, y = None, aspect = 'equal', zmin = None, zmax = None, ax = plt, crosshairs=False):
    # Z rows are x, columns are y
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])
    #im = ax.imshow(Z.transpose(), extent = [x[0], x[-1], y[0], y[-1]], aspect = aspect, 
    #           origin = 'lower', vmin = zmin, vmax = zmax, cmap = 'YlOrRd')
    im = ax.pcolormesh(x, y, Z.T, norm=colors.LogNorm(), vmin = zmin, vmax = zmax, cmap='GnBu', shading = 'auto')
    if crosshairs:
        ax.hlines(0, x[0], x[-1], 'k', linewidth=0.5)
        ax.vlines(0, y[0], y[-1], 'k', linewidth=0.5)
    return im
#%% Final plotting:
time = np.arange(17279)/720
thunder = data_list[34] # 5/19
EQ = data_list[37] # 5/22
shoshone = data_list[39] # 5/24
waterfall = data_list[26] # 5/11

waterfall_6 = data_list_6[0]
thunder_6 = data_list_6[1]
EQ_6 = data_list_6[2]
shoshone_6 = data_list_6[3]



fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6,4) # defines gridspace and initializes subplot locations
f_ax1 = fig.add_subplot(gs[0,:]) # backazimuth
f_ax2 = fig.add_subplot(gs[1,:]) # slowness
f_ax3 = fig.add_subplot(gs[2,0])
f_ax4 = fig.add_subplot(gs[2,1])
f_ax5 = fig.add_subplot(gs[2,2])
f_ax6 = fig.add_subplot(gs[2,3])
f_ax7 = fig.add_subplot(gs[3,0])
f_ax8 = fig.add_subplot(gs[3,1])
f_ax9 = fig.add_subplot(gs[3,2])
f_ax10 = fig.add_subplot(gs[3,3])
#N=6 grids
f_ax11 = fig.add_subplot(gs[4,0])
f_ax12 = fig.add_subplot(gs[4,1])
f_ax13 = fig.add_subplot(gs[4,2])
f_ax14 = fig.add_subplot(gs[4,3])
f_ax15 = fig.add_subplot(gs[5,0])
f_ax16 = fig.add_subplot(gs[5,1])
f_ax17 = fig.add_subplot(gs[5,2])
f_ax18 = fig.add_subplot(gs[5,3])
# Colormaps for backazimuth and slowness
backazimuth = image(detections_list,y=bin_centers,ax=f_ax1)
slowness = image_bg(slowness_detection,y=slow_centers,ax=f_ax2)
# backazimuth subplotting
f_ax3.plot(time,waterfall[:,3],'k.',markersize=.7)
f_ax4.plot(time,shoshone[:,3],'k.',markersize=.7)
f_ax5.plot(time,thunder[:,3],'k.',markersize=.7) # Adjust
f_ax6.plot(time,EQ[:,3],'k.',markersize=.7)
# N=6 backazimuth subplotting
f_ax11.plot(time,waterfall_6[:,3],'k.',markersize=.7)
f_ax12.plot(time,shoshone_6[:,3],'k.',markersize=.7)
f_ax13.plot(time,thunder_6[:,3],'k.',markersize=.7)
f_ax14.plot(time,EQ_6[:,3],'k.',markersize=.7)

# slowness subplotting
f_ax7.plot(time,waterfall[:,4],'k.',markersize=.7)
f_ax8.plot(time,shoshone[:,4],'k.',markersize=.7)
f_ax9.plot(time,thunder[:,4],'k.',markersize=.7) # Adjust
f_ax10.plot(time,EQ[:,4],'k.',markersize=.7)
f_ax10.axvline(2568.8/720,color='r',markersize=0.6) # HIGHLIGHTS M3.8 EQ
# N=6 slowness subplotting
f_ax15.plot(time,waterfall_6[:,4],'k.',markersize=.7)
f_ax16.plot(time,shoshone_6[:,4],'k.',markersize=.7)
f_ax17.plot(time,thunder_6[:,4],'k.',markersize=.7)
f_ax18.plot(time,EQ_6[:,4],'k.',markersize=.7)
f_ax18.axvline(2568.8/720,color='r',markersize=0.6) # HIGHLIGHTS M3.8 EQ

# Setting and emptying x and y ticks to conserve space
f_ax1.set_yticks(np.arange(0,420,60))
f_ax1.set_xticks([]) # MODIFIED
f_ax2.set_xticks(np.arange(0,58,4))
f_ax2.set_yticks(np.arange(0,5,1))
f_ax3.set_xticks([])
f_ax3.set_yticks(np.arange(0,420,60))
f_ax3.set_xlim(left=-2)
f_ax4.set_yticks([]) #
f_ax4.set_xticks([])
f_ax4.set_xlim(left=-2)
f_ax5.set_yticks([])#
f_ax5.set_xticks([])
f_ax5.set_xlim(left=-2)
f_ax6.set_yticks([])#
f_ax6.set_xticks([])
f_ax6.set_xlim(left=-2)
f_ax8.set_yticks([])#
f_ax8.set_xlim(left=-2)
f_ax9.set_yticks([])#
f_ax9.set_xlim(left=-2)
f_ax10.set_yticks([])#
f_ax10.set_xlim(left=-2)
f_ax7.set_xticks([])
f_ax7.set_yticks(np.arange(0,5,1))
f_ax7.set_ylim(top=4)
f_ax7.set_xlim(left=-2)
f_ax8.set_xticks([])
f_ax8.set_ylim(top=4)
f_ax9.set_xticks([])
f_ax9.set_ylim(top=4)
f_ax10.set_xticks([])
f_ax10.set_ylim(top=4)
# for N=6
f_ax11.set_xticks([])
f_ax11.set_yticks(np.arange(0,420,60))
f_ax11.set_xlim(left=-2)
f_ax12.set_yticks([])
f_ax12.set_xticks([])
f_ax12.set_xlim(left=-2)
f_ax13.set_yticks([])
f_ax13.set_xticks([])
f_ax13.set_xlim(left=-2)
f_ax14.set_yticks([])
f_ax14.set_xticks([])
f_ax14.set_xlim(left=-2)
f_ax16.set_yticks([])
f_ax16.set_xlim(left=-2)
f_ax17.set_yticks([])
f_ax17.set_xlim(left=-2)
f_ax18.set_yticks([])
f_ax18.set_xlim(left=-2)
f_ax15.set_xticks(np.arange(0,30,6))
f_ax15.set_yticks(np.arange(0,5,1))
f_ax15.set_ylim(top=4)
f_ax15.set_xlim(left=-2)
f_ax16.set_xticks(np.arange(0,30,6))
f_ax16.set_ylim(top=4)
f_ax17.set_xticks(np.arange(0,30,6))
f_ax17.set_ylim(top=4)
f_ax18.set_xticks(np.arange(0,30,6))
f_ax18.set_ylim(top=4)

# Subplot titles
f_ax1.set_title('a.',x=0.01,y=0.8, weight='bold',fontsize=14)
f_ax2.set_title('b.',x=0.01,y=0.8, weight='bold',fontsize=14)

f_ax3.set_title('c.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax3.text(s='2020-05-11 (Lady Face Falls), Day 26',x=-2,y=390,fontsize=12)

f_ax4.set_title('e.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax4.text(s='2020-05-24 (Shoshone/Twin), Day 39',x=-2,y=390,fontsize=12)

f_ax5.text(s='2020-05-19 (Thunder), Day 34',x=-2,y=390,fontsize=12)
f_ax5.set_title('g.',x=0.04,y=0.8, weight='bold',fontsize=14)


f_ax6.set_title('i.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax6.text(s='2020-05-22 (Earthquakes), Day 37',x=-2,y=390,fontsize=12)

f_ax7.set_title('d.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax8.set_title('f.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax9.set_title('h.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax10.set_title('j.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax10.text(s='M3.8',x=0.6,y=.5,color='r',weight='bold',fontsize=10)
# N=6 subplot titles
f_ax11.set_title('k.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax12.set_title('m.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax13.set_title('o.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax14.set_title('q.',x=0.04,y=0.8, weight='bold',fontsize=14)



f_ax15.set_title('l.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax16.set_title('n.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax17.set_title('p.',x=0.04,y=0.8, weight='bold',fontsize=14)
f_ax18.set_title('r.',x=0.04,y=0.8, weight='bold',fontsize=14)

# Earthquake Titles on f_ax2
f_ax2.text(s='M4.2\n    |',x=54.1,y=0.85,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.8\n    |',x=36.1,y=0.85,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.7\n    |',x=48.1,y=0.85,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.7\n    |',x=16.1,y=0.45,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.6\n    |',x=6.1,y=0.45,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.5\n    |',x=35.1,y=0.45,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.4\n    |',x=8.1,y=0.45,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.3\n|',x=0,y=0.45,color='k',fontsize=12,weight='bold')
f_ax2.text(s='M3.3\n    |',x=47.1,y=0.45,color='k',fontsize=11,weight='bold') # 6/2
f_ax2.text(s='M3.3\n    |',x=38.1,y=0.45,color='k',fontsize=12,weight='bold') # 5/24
f_ax2.text(s='M3.3\n    |',x=30.1,y=0.45,color='k',fontsize=12,weight='bold') # 5/16
f_ax2.text(s='M3.2\n    |',x=53.1,y=0.45,color='k',fontsize=12,weight='bold') # 6/8
f_ax2.text(s='M3.2\n    |',x=12.1,y=0.45,color='k',fontsize=12,weight='bold')

# X and Y labels
f_ax1.set_ylabel('Backazimuth '+chr(176),fontsize=17.5)
f_ax2.set_ylabel('Slowness (s/km)',fontsize=17.5)
f_ax2.set_xlabel('Days after 2020-04-15',fontsize=18)

f_ax3.set_ylabel('Backazimuth '+chr(176),fontsize=16)
f_ax7.set_ylabel('Slowness (s/km)',fontsize=16)
f_ax11.set_ylabel('Backazimuth '+chr(176),fontsize=16)
f_ax15.set_ylabel('Slowness (s/km)',fontsize=16)

f_ax7.set_xlabel('2020-05-11, N=6',loc='left',fontsize=12)
f_ax8.set_xlabel('2020-05-24, N=6',loc='left',fontsize=12)
f_ax9.set_xlabel('2020-05-19, N=6',loc='left',fontsize=12)
f_ax10.set_xlabel('2020-05-22, N=6',loc='left',fontsize=12)
# Bottom x axis labels
f_ax15.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax16.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax17.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax18.set_xlabel('Hours after 12AM MDT',fontsize=14)
# Colorbars
cb = fig.colorbar(backazimuth,ax=f_ax1)
cb.ax.tick_params(labelsize='x-large')
cb1 = fig.colorbar(slowness,ax=f_ax2)
cb1.ax.tick_params(labelsize='x-large')
cb1.set_label(label='     # of detections',loc='bottom',fontsize=18)

# Title, size, and saving
plt.gcf().set_size_inches(18,14)

plt.show()
#plt.savefig('Figures/Fig_2.jpg',dpi=600)
