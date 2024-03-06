# Fig.2 Plotting
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
#%% Imports and processes Temperature Data @ Stanely Ranger Station
stanley_data_2020 = np.array(pd.read_csv('imported_data/'+'KSNT_58day.csv',skiprows=[0,1,2,3,4,5,6]))
stanley_temp = stanley_data_2020[:,2]

# Creates an empy list and fills it with each day of 24-hour periods
temp_arrays = np.array_split(stanley_temp,58,axis=0)
adjusted_temp_arrays = []

for arrays in temp_arrays:
    adjusted_temp_arrays.append(arrays[~pd.isna(arrays)]) # Removes nan values

mean_temp_list = []
for arrays in adjusted_temp_arrays:
    total_temp = sum(arrays)
    mean_temp = total_temp / len(arrays)
    mean_temp = (mean_temp - 32) *(5/9) # conversion to Celsius
    mean_temp_list.append(mean_temp)

# OUTPUT: 
# mean_temp_list: 58-day list of mean daily temperature (C)

#%% Infrasound data processing
points_list = []

for data_index, data in enumerate(data_list):
    print([data_index,len(data_list)])
    index_list = []
    for i in range(17279):
        if data[i,3] < -123 and data[i,3] > -133 and data[i,4] < 3.2 and data[i,4] > 2.8: # finds indexes for power between backazimuth -128 +/- 10 degrees and slowness between 3.2 and 2.8
            index_list.append(i)
    points_list.append(len(index_list))
    
    
# OUTPUT:
# points_list: 58-day list of number of filtered points detected each day.
#%% Plotting Lady Face Falls: # of detections, temperature
time = np.arange(0,58,1)
fig, axes = plt.subplots(2,sharex=True,sharey=False)
axes[0].plot(time,points_list,'k-',linewidth=2.4)
axes[1].plot(time,mean_temp_list,'tab:blue',linewidth=2.4)

axes[0].grid(axis='x')
axes[1].grid(axis='x')
axes[0].set_ylabel('Detections per day',fontsize=24)
axes[1].set_ylabel('Mean daily temp ('+chr(176)+'C)\n@ Stanley Ranger Station',fontsize=24)

axes[1].set_xlabel('Days after 2020-04-15',fontsize=26)

axes[1].set_xticks(np.arange(0,62,4))
axes[1].set_yticks(np.arange(-4,22,4))
axes[0].set_title('b.',x=0.02,y=0.9,weight='bold',fontsize=16)
axes[1].set_title('c.',x=0.02,y=0.9,weight='bold',fontsize=16)

axes[0].tick_params(labelsize=15)
axes[1].tick_params(labelsize=15)
plt.xlim([-2,58])
plt.subplots_adjust(hspace=0.1)
plt.gcf().set_size_inches(18,10)
plt.savefig('Figures/Fig_3.jpg',dpi=300)




