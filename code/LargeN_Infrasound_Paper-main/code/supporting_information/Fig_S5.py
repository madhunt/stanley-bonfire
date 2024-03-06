# Figure S5 Plotting
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import obspy
from obspy.signal.util import util_geo_km
#%% Imports infrasound data

dates = ['20200511','20200519']  
#Imports 3_element data
data_list_3 = []
for days in dates:
    with open('pickle_files/N_3/'+days+'_0600_N_3.pkl','rb') as z:
        data_list_3.append(pickle.load(z))
for d in data_list_3:
    for q in range(len(d)):
        if d[q,3] <=0:
            d[q,3] = (d[q,3] + 360)
            
#Imports 6_element data
data_list_6 = []
for days in dates:
    with open('pickle_files/N_6/'+days+'_0600_N_6.pkl','rb') as z:
        data_list_6.append(pickle.load(z))
for d in data_list_6:
    for q in range(len(d)):
        if d[q,3] <=0:
            d[q,3] = (d[q,3] + 360)
            
#Imports 9_element data
data_list_9 = []
for days in dates:
    with open('pickle_files/N_9/'+days+'_0600_N_9.pkl','rb') as z:
        data_list_9.append(pickle.load(z))
for d in data_list_9:
    for q in range(len(d)):
        if d[q,3] <=0:
            d[q,3] = (d[q,3] + 360)
            
#Imports 12_element data
data_list_12 = []
for days in dates:
    with open('pickle_files/N_12/'+days+'_0600_N_12.pkl','rb') as z:
        data_list_12.append(pickle.load(z))
for d in data_list_12:
    for q in range(len(d)):
        if d[q,3] <=0:
            d[q,3] = (d[q,3] + 360)
            
#Imports 16_element data
data_list_16 = []
for days in dates:
    with open('pickle_files/N_16/'+days+'_0600_N_16.pkl','rb') as z:
        data_list_16.append(pickle.load(z))
for d in data_list_16:
    for q in range(len(d)):
        if d[q,3] <=0:
            d[q,3] = (d[q,3] + 360)

#%%
inv = obspy.read_inventory('XP.PARK.xml') # includes coordinates

contents = inv.get_contents()['channels']
lats = [inv.get_coordinates(s)['latitude'] for s in contents]
lons = [inv.get_coordinates(s)['longitude'] for s in contents]
zz = [inv.get_coordinates(s)['elevation'] for s in contents]
xx = np.zeros(len(lats))
yy = np.zeros(len(lats))
for i, (lat, lon) in enumerate(zip(lats, lons)):
    xx[i], yy[i] = util_geo_km(np.mean(lons), np.mean(lats), lon, lat)
    coords = {'x': xx, 'y': yy, 'z': zz,
                  'network': [string.split('.')[0] for string in contents],
                  'station': [string.split('.')[1] for string in contents],
                  'location': [string.split('.')[2] for string in contents]}


coords_full = pd.DataFrame(coords)
sub_array_3 = np.array([17, 9, 14])-1 # Sub-array uses HDF.09,14,17
sub_array_6 = np.array([1, 7, 12, 17, 19, 22])-1 # Sub-array uses HDF.01,07,12,17,19,22
sub_array_9 = np.array([1, 4, 7, 9, 12, 15, 17, 19, 22])-1 # Sub-array uses HDF.01,04,07,09,12,15,17,19,22
sub_array_12 = np.array([1, 2, 4, 7, 9, 10, 12, 15, 17, 18, 19, 22])-1 # Sub-array uses HDF.01,02,04,07,09,10,12,15,17,18,19,22
sub_array_16 = np.array([1, 2, 4, 5, 7, 9, 10, 12, 14, 15, 16, 17, 18, 19, 21, 22])-1 # Sub-array uses HDF.01,02,04,05,07,09,10,12,14,15,16,17,18,19,21,22

coords_sub_3 = coords_full.iloc[sub_array_3,:]
coords_sub_6 = coords_full.iloc[sub_array_6,:]
coords_sub_9 = coords_full.iloc[sub_array_9,:]
coords_sub_12 = coords_full.iloc[sub_array_12,:]
coords_sub_16 = coords_full.iloc[sub_array_16,:]
#%% Final plotting:
time = np.arange(17279)/720

waterfall_3 = data_list_3[0]
thunder_3 = data_list_3[1]
waterfall_6 = data_list_6[0]
thunder_6 = data_list_6[1]
waterfall_9 = data_list_9[0]
thunder_9 = data_list_9[1]
waterfall_12 = data_list_12[0]
thunder_12 = data_list_12[1]
waterfall_16 = data_list_16[0]
thunder_16 = data_list_16[1]

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(5,5) # defines gridspace and initializes subplot locations

# Waterfall Days 
f_ax1 = fig.add_subplot(gs[1,0]) # N=3
f_ax2 = fig.add_subplot(gs[2,0]) # N=3
f_ax3 = fig.add_subplot(gs[1,1]) # 6
f_ax4 = fig.add_subplot(gs[2,1]) # 6
f_ax5 = fig.add_subplot(gs[1,2]) # 10
f_ax6 = fig.add_subplot(gs[2,2]) # 10
f_ax7 = fig.add_subplot(gs[1,3]) # 15
f_ax8 = fig.add_subplot(gs[2,3]) # 15
f_ax9 = fig.add_subplot(gs[1,4]) # Full
f_ax10 = fig.add_subplot(gs[2,4]) # Full

# Thunder days
f_ax11 = fig.add_subplot(gs[3,0]) # N=3
f_ax12 = fig.add_subplot(gs[4,0]) # N=3
f_ax13 = fig.add_subplot(gs[3,1]) # N=6
f_ax14 = fig.add_subplot(gs[4,1]) # N=6
f_ax15 = fig.add_subplot(gs[3,2])
f_ax16 = fig.add_subplot(gs[4,2])
f_ax17 = fig.add_subplot(gs[3,3])
f_ax18 = fig.add_subplot(gs[4,3])
f_ax19 = fig.add_subplot(gs[3,4])
f_ax20 = fig.add_subplot(gs[4,4])

# Maps
f_ax21 = fig.add_subplot(gs[0,0]) # N=3
f_ax22 = fig.add_subplot(gs[0,1]) # N=6
f_ax23 = fig.add_subplot(gs[0,2]) # N=10
f_ax24 = fig.add_subplot(gs[0,3]) # N=15
f_ax25 = fig.add_subplot(gs[0,4]) # N=22


# map plotting
f_ax21.scatter(1000*coords_sub_3.x,1000*coords_sub_3.y,color='black',marker='^',s=35)
f_ax22.scatter(1000*coords_sub_6.x,1000*coords_sub_6.y,color='black',marker='^',s=35)
f_ax23.scatter(1000*coords_sub_9.x,1000*coords_sub_9.y,color='black',marker='^',s=35)
f_ax24.scatter(1000*coords_sub_12.x,1000*coords_sub_12.y,color='black',marker='^',s=35)
f_ax25.scatter(1000*coords_sub_16.x,1000*coords_sub_16.y,color='black',marker='^',s=35)

# setting x and y limits
f_ax21.set_xlim(-85,78)
f_ax21.set_ylim(-60,60)
f_ax22.set_xlim(-85,78)
f_ax22.set_ylim(-60,60)
f_ax23.set_xlim(-85,78)
f_ax23.set_ylim(-60,60)
f_ax24.set_xlim(-85,78)
f_ax24.set_ylim(-60,60)
f_ax25.set_xlim(-85,78)
f_ax25.set_ylim(-60,60)

# Backazimuth and slowness plotting
f_ax1.plot(time,waterfall_3[:,3],'k.',markersize=.7) # backaz
f_ax2.plot(time,waterfall_3[:,4],'k.',markersize=.7) # slowness
f_ax11.plot(time,thunder_3[:,3],'k.',markersize=.7) # backaz
f_ax12.plot(time,thunder_3[:,4],'k.',markersize=.7) # slowness

f_ax3.plot(time,waterfall_6[:,3],'k.',markersize=.7) # backaz
f_ax4.plot(time,waterfall_6[:,4],'k.',markersize=.7) # slowness
f_ax13.plot(time,thunder_6[:,3],'k.',markersize=.7) # backaz
f_ax14.plot(time,thunder_6[:,4],'k.',markersize=.7) # slowness

f_ax5.plot(time,waterfall_9[:,3],'k.',markersize=.7) # backaz
f_ax6.plot(time,waterfall_9[:,4],'k.',markersize=.7) # slowness
f_ax15.plot(time,thunder_9[:,3],'k.',markersize=.7) # backaz
f_ax16.plot(time,thunder_9[:,4],'k.',markersize=.7) # slowness

f_ax7.plot(time,waterfall_12[:,3],'k.',markersize=.7) # backaz
f_ax8.plot(time,waterfall_12[:,4],'k.',markersize=.7) # slowness
f_ax17.plot(time,thunder_12[:,3],'k.',markersize=.7) # backaz
f_ax18.plot(time,thunder_12[:,4],'k.',markersize=.7) # slowness

f_ax9.plot(time,waterfall_16[:,3],'k.',markersize=.7) # backaz
f_ax10.plot(time,waterfall_16[:,4],'k.',markersize=.7) # slowness
f_ax19.plot(time,thunder_16[:,3],'k.',markersize=.7) # backaz
f_ax20.plot(time,thunder_16[:,4],'k.',markersize=.7) # slowness

f_ax1.set_yticks(np.arange(0,420,60))
f_ax2.set_yticks(np.arange(0,5,1))
f_ax11.set_yticks(np.arange(0,420,60))
f_ax12.set_yticks(np.arange(0,5,1))

f_ax12.set_xticks(np.arange(0,28,4))
f_ax14.set_xticks(np.arange(0,28,4))
f_ax16.set_xticks(np.arange(0,28,4))
f_ax18.set_xticks(np.arange(0,28,4))
f_ax20.set_xticks(np.arange(0,28,4))

# Emptying axis to conserve space
f_ax3.set_yticks([])
f_ax4.set_yticks([])
f_ax5.set_yticks([])
f_ax6.set_yticks([])
f_ax7.set_yticks([])
f_ax8.set_yticks([])
f_ax9.set_yticks([])
f_ax10.set_yticks([])

f_ax13.set_yticks([])
f_ax14.set_yticks([])
f_ax15.set_yticks([])
f_ax16.set_yticks([])
f_ax17.set_yticks([])
f_ax18.set_yticks([])
f_ax19.set_yticks([])
f_ax20.set_yticks([])

f_ax1.set_xticks([])
f_ax3.set_xticks([])
f_ax5.set_xticks([])
f_ax7.set_xticks([])
f_ax9.set_xticks([])

f_ax2.set_xticks([])
f_ax4.set_xticks([])
f_ax6.set_xticks([])
f_ax8.set_xticks([])
f_ax10.set_xticks([])

f_ax11.set_xticks([])
f_ax13.set_xticks([])
f_ax15.set_xticks([])
f_ax17.set_xticks([])
f_ax19.set_xticks([])

f_ax22.set_yticks([])

f_ax23.set_yticks([])

f_ax24.set_yticks([])

f_ax25.set_yticks([])
# y axis limits
f_ax2.set_ylim(top=4)
f_ax4.set_ylim(top=4)
f_ax6.set_ylim(top=4)
f_ax8.set_ylim(top=4)
f_ax10.set_ylim(top=4)
f_ax12.set_ylim(top=4)
f_ax14.set_ylim(top=4)
f_ax16.set_ylim(top=4)
f_ax18.set_ylim(top=4)
f_ax20.set_ylim(top=4)
# x axis limits
f_ax1.set_xlim(left=-2)
f_ax2.set_xlim(left=-2)
f_ax3.set_xlim(left=-2)
f_ax4.set_xlim(left=-2)
f_ax5.set_xlim(left=-2)
f_ax6.set_xlim(left=-2)
f_ax7.set_xlim(left=-2)
f_ax8.set_xlim(left=-2)
f_ax9.set_xlim(left=-2)
f_ax10.set_xlim(left=-2)
f_ax11.set_xlim(left=-2)
f_ax12.set_xlim(left=-2)
f_ax13.set_xlim(left=-2)
f_ax14.set_xlim(left=-2)
f_ax15.set_xlim(left=-2)
f_ax16.set_xlim(left=-2)
f_ax17.set_xlim(left=-2)
f_ax18.set_xlim(left=-2)
f_ax19.set_xlim(left=-2)
f_ax20.set_xlim(left=-2)


# Subplot titles
f_ax21.set_title('a.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax22.set_title('b.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax23.set_title('c.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax24.set_title('d.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax25.set_title('e.',x=0.035,y=0.85, weight='bold',fontsize=14)

f_ax1.set_title('f.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax2.set_title('g.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax3.set_title('h.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax4.set_title('i.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax5.set_title('j.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax6.set_title('k.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax7.set_title('l.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax8.set_title('m.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax9.set_title('n.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax10.set_title('o.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax11.set_title('p.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax12.set_title('q.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax13.set_title('r.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax14.set_title('s.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax15.set_title('t.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax16.set_title('u.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax17.set_title('v.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax18.set_title('w.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax19.set_title('x.',x=0.035,y=0.85, weight='bold',fontsize=14)
f_ax20.set_title('y.',x=0.035,y=0.85, weight='bold',fontsize=14)

f_ax1.text(s='2020-05-11 (Lady Face Falls),',x=-2,y=390,fontsize=14)
f_ax1.text(s='N=3',x=17,y=390,fontsize=16,weight='bold')
f_ax3.text(s='2020-05-11 (Lady Face Falls),',x=-2,y=390,fontsize=14)
f_ax3.text(s='N=6',x=17,y=390,fontsize=16,weight='bold')
f_ax5.text(s='2020-05-11 (Lady Face Falls),',x=-2,y=390,fontsize=14)
f_ax5.text(s='N=9',x=17,y=390,fontsize=16,weight='bold')
f_ax7.text(s='2020-05-11 (Lady Face Falls),',x=-2,y=390,fontsize=14)
f_ax7.text(s='N=12',x=17,y=390,fontsize=16,weight='bold')
f_ax9.text(s='2020-05-11 (Lady Face Falls),',x=-2,y=390,fontsize=14)
f_ax9.text(s='N=16',x=17,y=390,fontsize=16,weight='bold')

f_ax11.text(s='2020-05-19 (Thunder),',x=-2,y=390,fontsize=14)
f_ax11.text(s='N=3',x=13,y=390,fontsize=16,weight='bold')
f_ax13.text(s='2020-05-19 (Thunder),',x=-2,y=390,fontsize=14)
f_ax13.text(s='N=6',x=13,y=390,fontsize=16,weight='bold')
f_ax15.text(s='2020-05-19 (Thunder),',x=-2,y=390,fontsize=14)
f_ax15.text(s='N=9',x=13,y=390,fontsize=16,weight='bold')
f_ax17.text(s='2020-05-19 (Thunder),',x=-2,y=390,fontsize=14)
f_ax17.text(s='N=12',x=13,y=390,fontsize=16,weight='bold')
f_ax19.text(s='2020-05-19 (Thunder),',x=-2,y=390,fontsize=14)
f_ax19.text(s='N=16',x=13,y=390,fontsize=16,weight='bold')

f_ax21.text(s='Array Geometry,',x=-80,y=65,fontsize=14)
f_ax21.text(s='N=3',x=-13,y=65,fontsize=16,weight='bold')
f_ax22.text(s='Array Geometry,',x=-80,y=65,fontsize=14)
f_ax22.text(s='N=6',x=-13,y=65,fontsize=16,weight='bold')
f_ax23.text(s='Array Geometry,',x=-80,y=65,fontsize=14)
f_ax23.text(s='N=9',x=-13,y=65,fontsize=16,weight='bold')
f_ax24.text(s='Array Geometry,',x=-80,y=65,fontsize=14)
f_ax24.text(s='N=12',x=-13,y=65,fontsize=16,weight='bold')
f_ax25.text(s='Array Geometry,',x=-80,y=65,fontsize=14)
f_ax25.text(s='N=16',x=-13,y=65,fontsize=16,weight='bold')

# X and Y labels
f_ax1.set_ylabel('Backazimuth '+chr(176),fontsize=21)
f_ax2.set_ylabel('Slowness (s/km)',fontsize=21)
f_ax11.set_ylabel('Backazimuth '+chr(176),fontsize=21)
f_ax12.set_ylabel('Slowness (s/km)',fontsize=21)

f_ax21.set_ylabel('meters',fontsize=18)
f_ax21.set_xlabel('meters',fontsize=18)
f_ax22.set_xlabel('meters',fontsize=18)
f_ax23.set_xlabel('meters',fontsize=18)
f_ax24.set_xlabel('meters',fontsize=18)
f_ax25.set_xlabel('meters',fontsize=18)

# Bottom x axis labels
f_ax12.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax14.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax16.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax18.set_xlabel('Hours after 12AM MDT',fontsize=14)
f_ax20.set_xlabel('Hours after 12AM MDT',fontsize=14)
# Title, size, and saving
plt.gcf().set_size_inches(22,18)

plt.savefig('Figures/Fig_S5.jpg',dpi=600)
