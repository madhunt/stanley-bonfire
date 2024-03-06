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


def main(process=False):

    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    path_data = os.path.join(path_home, "data")
    csv_file = os.path.join(path_data, "20240114_Bonfire_Gems.csv")
    mseed_files = os.path.join(path_data, "mseed", "2024-01-15*.mseed")
    path_save = os.path.join(path_data, "processed", "processing_output.npy")

    if process == True:
        output = process_data(mseed_files, csv_file, path_save)
    else:
        # data has already been processed
        output = np.load(path_save)

    
    # try plotting with obspy example
    #TODO clean this up to make it my own code!!
    labels = ["rel pwr", "abs pwr", "backaz", "slow"]

    xlocator = mdates.AutoDateLocator()
    fig = plt.figure()
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(4, 1, i+1)
        ax.scatter(output[:,0], output[:,i+1], c=output[:,1], alpha=0.6,
                   edgecolors='none', cmap=obspy_sequential)
        ax.set_ylabel(lab)
        ax.set_xlim(output[0,0], output[-1,0])
        ax.set_ylim(output[:,i+1].min(), output[:,i+1].max())
        ax.xaxis.set_major_locator(xlocator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
    fig.suptitle("Bonfire Test")
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    plt.show()



    return

def process_data(mseed_files, csv_file, path_save):
    # import data as obspy stream
    data = obspy.read(mseed_files)

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
    
    # HP 1 Hz filter
            
    
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
