# Supporting Information Figure S1
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import pickle
import numpy as np
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

for date in dates:
    with open('pickle_files/0_1_to_0_5Hz/'+ date + '_06000_1to0_5Hz.pkl','rb') as f:
        data_list.append(pickle.load(f))

#%% Multi-day Infrasound data processing
for data in data_list:
    for i in range(5759):
        if data[i,3] <= -2.5:
            data[i,3] = data[i,3] + 360
    
for i,data in enumerate(data_list):
    for row in range(len(data)):
        if data[row,4] > 5.5: # removing slowness values > 5.5
            data[row,:] = np.nan
#%%
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
#%% Multi-Day Plotting, binned backazimuth
## white-yellow-orange-red colormap
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
#%% slowness space
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
time = np.arange(5759)/240.8333
thunder = data_list[34] # 5/19
EQ = data_list[37] # 5/22
shoshone = data_list[39] # 5/24
waterfall = data_list[26] # 5/11

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(4,4) # defines gridspace and initializes subplot locations
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
# Colormaps for backazimuth and slowness
backazimuth = image(detections_list,y=bin_centers,ax=f_ax1)
slowness = image_bg(slowness_detection,y=slow_centers,ax=f_ax2)
# backazimuth subplotting
f_ax3.plot(time,waterfall[:,3],'k.',markersize=.8)
f_ax4.plot(time,shoshone[:,3],'k.',markersize=.8)
f_ax5.plot(time,thunder[:,3],'k.',markersize=.8) # Adjust
f_ax6.plot(time,EQ[:,3],'k.',markersize=.8)

# slowness subplotting
f_ax7.plot(time,waterfall[:,4],'k.',markersize=.75)
f_ax8.plot(time,shoshone[:,4],'k.',markersize=.75)
f_ax9.plot(time,thunder[:,4],'k.',markersize=.75) # Adjust
f_ax10.plot(time,EQ[:,4],'k.',markersize=.75)

# Setting and emptying x and y ticks to conserve space
f_ax1.set_yticks(np.arange(0,420,60))
f_ax1.set_xticks(np.arange(0,58,4))
f_ax2.set_xticks(np.arange(0,58,4))
f_ax2.set_yticks(np.arange(0,5,1))
f_ax3.set_xticks([])
f_ax3.set_yticks(np.arange(0,420,60))
f_ax3.set_xlim(left=-2)
f_ax4.set_yticks([],fontsize=0)
f_ax4.set_xticks([])
f_ax4.set_xlim(left=-2)
f_ax5.set_yticks([],fontsize=0)
f_ax5.set_xticks([])
f_ax5.set_xlim(left=-2)
f_ax6.set_yticks([],fontsize=0)
f_ax6.set_xticks([])
f_ax6.set_xlim(left=-2)
f_ax8.set_yticks([],fontsize=0)
f_ax8.set_xlim(left=-2)
f_ax9.set_yticks([],fontsize=0)
f_ax9.set_xlim(left=-2)
f_ax10.set_yticks([],fontsize=0)
f_ax10.set_xlim(left=-2)
f_ax7.set_xticks(np.arange(0,30,6))
f_ax7.set_yticks(np.arange(0,5,1))
f_ax7.set_ylim(top=4)
f_ax7.set_xlim(left=-2)
f_ax8.set_xticks(np.arange(0,30,6))
f_ax8.set_ylim(top=4)
f_ax9.set_xticks(np.arange(0,30,6))
f_ax9.set_ylim(top=4)
f_ax10.set_xticks(np.arange(0,30,6))
f_ax10.set_ylim(top=4)

# Subplot titles
f_ax1.set_title('a.',x=0.01,y=0.85, weight='bold',fontsize=14)
f_ax2.set_title('b.',x=0.01,y=0.85, weight='bold',fontsize=14)

f_ax3.set_title('c.',x=0.04,y=0.85, weight='bold',fontsize=14)
f_ax3.text(s='2020-05-11 (Lady Face Falls), Day 26',x=-2,y=390,fontsize=13)

f_ax4.set_title('e.',x=0.04,y=0.85, weight='bold',fontsize=14)
f_ax4.text(s='2020-05-24 (Shoshone/Twin), Day 39',x=-2,y=390,fontsize=13)

f_ax5.text(s='2020-05-19 (Thunder), Day 34',x=-2,y=390,fontsize=13)
f_ax5.set_title('g.',x=0.04,y=0.85, weight='bold',fontsize=14)


f_ax6.set_title('i.',x=0.04,y=0.85, weight='bold',fontsize=14)
f_ax6.text(s='2020-05-22 (Earthquakes), Day 37',x=-2,y=390,fontsize=13)

f_ax7.set_title('d.',x=0.04,y=0.85, weight='bold',fontsize=14)
f_ax8.set_title('f.',x=0.04,y=0.85, weight='bold',fontsize=14)
f_ax9.set_title('h.',x=0.04,y=0.85, weight='bold',fontsize=14)
f_ax10.set_title('j.',x=0.04,y=0.85, weight='bold',fontsize=14)

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
f_ax1.set_ylabel('Backazimuth '+chr(176),fontsize=20)
f_ax2.set_ylabel('Slowness (s/km)',fontsize=20)
f_ax2.set_xlabel('Days after 2020-04-15',fontsize=20)
f_ax3.set_ylabel('Backazimuth '+chr(176),fontsize=19)
f_ax7.set_ylabel('Slowness (s/km)',fontsize=19)
f_ax7.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax8.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax9.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax10.set_xlabel('Hours after 12AM MDT',fontsize=14)

# Colorbars
cb = fig.colorbar(backazimuth,ax=f_ax1)
cb.ax.tick_params(labelsize='x-large')
cb1 = fig.colorbar(slowness,ax=f_ax2)
cb1.ax.tick_params(labelsize='x-large')
cb.set_label(label='# of detections',fontsize=18)

# Title, size, and saving
plt.gcf().set_size_inches(18,10.5)
plt.suptitle('0.1 to 0.5 Hz, N=22',fontsize=22,weight='bold')
plt.savefig('Figures/Fig_S1.jpg',dpi=600)
