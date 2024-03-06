#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import obspy
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
from obspy.imaging.cm import obspy_sequential
import os
import pandas as pd
import scipy as sci


def main(process=False):

    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    path_data = os.path.join(path_home, "data")
    csv_file = os.path.join(path_data, "20240114_Bonfire_Gems.csv")
    mseed_files = os.path.join(path_data, "mseed", "2024-01-15*.mseed")
    

    #NOTE better way of doing this
    path_save = os.path.join(path_data, "processed", "1HP_processed_output.npy")

    if process == True:
        # fiter and beamform    
        output = process_data(mseed_files, csv_file, path_save)

    else:
        # data has already been processed
        output = np.load(path_save)
    

    # correct backaz from 0 to 360 (instead of -180 to +180)
    #output_corr = np.copy(output)
    output[:,3] = [output[i][3] if output[i][3]>=0 else output[i][3]+360 for i in range(output.shape[0])]

    # plot backaz to test
    #plt.plot(output[:,3][:100], 'bo', alpha=0.5, label='original')
    #plt.plot(output_corr[:,3][:100], 'ro', alpha=0.5, label='corrected')
    #plt.xlabel('sample number')
    #plt.ylabel('backazimuth [deg]')
    #plt.legend()
    #plt.show()
    
    time = output[:,0]

    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)
    ax[0].scatter(time, output[:,3], c=output[:,1], alpha=0.6, edgecolors='none', cmap=obspy_sequential)
    ax[0].set_ylabel("Backazimuth [$^o$]")
    ax[0].set_ylim([0, 360])
    ax[0].set_yticks(ticks=np.arange(0, 360+60, 60))
    ax[0].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax[1].scatter(time, output[:,4], c=output[:,1], alpha=0.6, edgecolors='none', cmap=obspy_sequential)
    ax[1].set_ylabel("Slowness [s/km]")
    #ax[1].set_ylim([0, 360])
    ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[1].set_xlabel("UTC Time")

    fig.autofmt_xdate()
    fig.suptitle("Stanley Bonfire")
    plt.show()


    return

def process_data(mseed_files, csv_file, path_save):
    '''
    
    INPUTS

    RETURNS
    output : np array : timestamp, relative power, absolute power, backazimuth, slowness

    '''
    # import data as obspy stream
    data = obspy.read(mseed_files)

    # filter data with 1 Hz highpass
    data = data.filter('highpass', freq=1.0)

    # import coordinates
    coords = pd.read_csv(csv_file)
    coords["Name"] = coords["Name"].astype(str)
    
    # get rid of any stations that don't have coordinates
    data_list = [trace for trace in data.traces if trace.stats['station'] in coords["Name"].to_list()]
    # convert list back to obspy stream
    data = obspy.Stream(traces=data_list)

    # assign coordinates to stations
    for idx, row in coords.iterrows():
        sn = row["Name"]
        lat = row["Latitude"]
        lon = row["Longitude"]
        elv = row["Elevation"]

        for trace in data.select(station=sn):
            trace.stats.coordinates = AttribDict({
                'latitude': lat,
                'longitude': lon,
                'elevation': elv
            }) 
    
    
    # choose gems of interest
    gem_list = ['138', '170', '155', '136', '133']  # hopefully back az towards south
    # 126, 175, 231 are closer (but not plane wave)
    data_subset = [trace for trace in data.traces if trace.stats['station'] in gem_list]
    data_subset = obspy.Stream(traces=data_subset)
    
    #NOTE this is the earliest time one of the gems shut off
    time_start = obspy.UTCDateTime("20240115T02:30:00")
    time_end = obspy.UTCDateTime("20240115T10:00:00")

    kwargs = dict(
        # slowness grid (in [s/km])
        sll_x=-4.0, slm_x=4.0, sll_y=-4.0, slm_y=4.0, sl_s=0.1,
        # sliding window
        win_len=10, win_frac = 0.50,
        # frequency
        frqlow=2.0, frqhigh=10.0, prewhiten=0,
        # output restrictions
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        stime=time_start, etime=time_end
    )

    output = array_processing(stream=data_subset, **kwargs)

    print(output)

    # save output as pkl
    np.save(path_save, output)

    return output


if __name__ == "__main__":
    main(process=False)
