import obspy
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
#%% Imports infrasound data

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
    with open('pickle_files/'+ date + '_0600all_day.pkl','rb') as f:
        data_list.append(pickle.load(f))
#%% Imports and processes Lightning Data (5/19)
file_1 = 'A20200519.USNW.loc'
file_2 = 'A20200520.USNW.loc'
lat_array = 44.274900
lon_array = -115.019100

wwlln_df_519 = pd.read_csv('imported_data/'+file_1, names = ['date', 'time', 'lat', 'lon', 'elevation', 'x'])
t_string = [wwlln_df_519.iloc[i,0] + 'T' + wwlln_df_519.iloc[i,1] for i in range(wwlln_df_519.shape[0])]
t_float_sec = [round((obspy.UTCDateTime(t) - obspy.UTCDateTime('2020-05-19'))/5)-1 for t in t_string]
wwlln_df_519['time_float_interval'] = t_float_sec

lat1 = np.array(wwlln_df_519['lat'])
lon1 = np.array(wwlln_df_519['lon'])

distance_list = []
thunder_backazimuth_list = []

for index in range(len(lat1)):
    latitude = lat1[index]
    longitude = lon1[index]
    thunder_distance,thunder_backazimuth,array_backazimuth = gps2dist_azimuth(lat_array, lon_array, latitude,longitude)
    distance_list.append(thunder_distance/1000) # converting meters to km
    thunder_backazimuth_list.append(thunder_backazimuth)
wwlln_df_519['thunder_distance_km'] = distance_list
wwlln_df_519['thunder_backazimuth'] = thunder_backazimuth_list
#%% Imports and processes Lightning Data (5/20)
lat_array = 44.274900
lon_array = -115.019100

wwlln_df_520 = pd.read_csv('imported_data/'+file_2, names = ['date', 'time', 'lat', 'lon', 'elevation', 'x'])
t_string = [wwlln_df_520.iloc[i,0] + 'T' + wwlln_df_520.iloc[i,1] for i in range(wwlln_df_520.shape[0])]
t_float_sec = [round((obspy.UTCDateTime(t) - obspy.UTCDateTime('2020-05-19'))/5)-1 for t in t_string]
wwlln_df_520['time_float_interval'] = t_float_sec

lat1 = np.array(wwlln_df_520['lat'])
lon1 = np.array(wwlln_df_520['lon'])

distance_list = []
thunder_backazimuth_list = []

for index in range(len(lat1)):
    latitude = lat1[index]
    longitude = lon1[index]
    thunder_distance,thunder_backazimuth,array_backazimuth = gps2dist_azimuth(lat_array, lon_array, latitude,longitude)
    distance_list.append(thunder_distance/1000) # converting meters to km
    thunder_backazimuth_list.append(thunder_backazimuth)
wwlln_df_520['thunder_distance_km'] = distance_list
wwlln_df_520['thunder_backazimuth'] = thunder_backazimuth_list

wwlln_df_5_19_20 = pd.concat([wwlln_df_519,wwlln_df_520],axis=0, ignore_index=True)
float_interval = np.array(wwlln_df_5_19_20['time_float_interval']) - (6*720) # corrects for 6:00UTC = 12:00AM MDT (changes to local)

wwlln_df_5_19_20['time_float_interval'] = float_interval

wwlln_df_5_19_20.sort_values('thunder_distance_km',inplace=True,ascending=False,ignore_index=True)
#%% Plotting Test:
for data in data_list: # converts data from -180-180 range to 0-360
    for i in range(17279):
        if data[i,3] <= -2.5:
            data[i,3] = data[i,3] + 360
            
time = np.arange(17279*2)
thunder1 = data_list[34] # 5/19
thunder1_1 = data_list[35] # 5/20
thunder1 = np.concatenate((thunder1,thunder1_1), axis=0)
w = np.where(wwlln_df_5_19_20['thunder_distance_km'] < 900)[0]
fig = plt.figure()
plt.scatter(wwlln_df_5_19_20['time_float_interval'][w]/720,wwlln_df_5_19_20['thunder_backazimuth'][w],s=16,label='WWLLN detection backazimuth',c=wwlln_df_5_19_20['thunder_distance_km'][w],cmap='spring_r')
cb = plt.colorbar()
cb.set_label(label='Thunderstorm Distance from PARK (km)',fontsize=18)
cb.ax.tick_params(labelsize='x-large')
plt.scatter(time/720,thunder1[:,3],color='k',s=0.2,label='PARK detection backazimuth')
plt.title('b.',x=0.02,y=0.96,weight='bold',fontsize=16)
plt.xlim([0,((17279*2)-4320)/720])
plt.legend(loc='upper right',fontsize='x-large',markerscale=4)
plt.xticks(np.arange(0,44,2),size='large')
plt.yticks(np.arange(0,380,20),size='large')
plt.xlabel('Hours after 2020-05-19 12AM MDT',fontsize=22)
plt.ylabel('Backazimuth '+chr(176),fontsize=24)
def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
# Title, size, and saving
plt.gcf().set_size_inches(18,10.5)
plt.savefig('Figures/5_19_20_Fig_5.jpg',dpi=800)
    