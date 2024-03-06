# Fig.4 Plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
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
    with open('pickle_files/N_full/'+ date + '_0600all_day.pkl','rb') as f:
        data_list.append(pickle.load(f))

#%% Shoshone Falls backazimuth detection number
s_detections_list = []

for data in data_list:
    data = np.array_split(data,6,axis=0) # split into 4-hour segments
    
    for arrays in data:
        shoshone_detections = np.sum((arrays[:,3] >= 161.125) & (arrays[:,3] <=168.625)) # Between 161.125 and 168.625 backazimuth
        s_detections_list.append(shoshone_detections)


snake_discharge = pd.read_csv('imported_data/Discharge_Snake_below.csv')
snake_discharge['Discharge'] = snake_discharge['Discharge'] / 35.3147  # conversion from ft^3/sec to m^3/sec
#%% Plotting Snake river discharge and shoshone detections
time_USGS = np.arange(0,len(snake_discharge))/(4*24)
minimum = np.repeat(75,len(time_USGS))
time = np.arange(0,len(s_detections_list))/6
fig, axes = plt.subplots(2,sharex=False,sharey=False)
axes[0].plot(time,s_detections_list,'k-',linewidth=1)
axes[0].set_title('b.',x=0.02,y=0.9,weight='bold',fontsize=16)
axes[1].plot(time_USGS,snake_discharge['Discharge'])
axes[1].plot(time_USGS,minimum,color='orange',label='75 m\u00b3/sec')
axes[1].set_title('c.',x=0.02,y=0.9,weight='bold',fontsize=16)

axes[0].set_ylabel('Detections per 4-hours',fontsize=24)
axes[1].set_ylabel('Snake River Discharge\n(meters\u00b3/sec)',fontsize=24)
axes[1].legend(fontsize=20)
axes[0].grid()
axes[1].grid()
axes[1].set_xticks(np.arange(0,65,5))
axes[0].set_xticks(np.arange(0,65,5))

axes[0].tick_params(labelsize=15)
axes[1].tick_params(labelsize=15)
plt.xlabel('Days after 2020-04-15',fontsize=26)
plt.gcf().set_size_inches(18,10)
plt.savefig('Figures/Fig_4.jpg')






