#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
import os
import pandas as pd


def main(process=False):

    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    path_data = os.path.join(path_home, "data")
    csv_file = os.path.join(path_data, "20240114_Bonfire_Gems.csv")
    mseed_files = os.path.join(path_data, "mseed", "2024-01-15*.mseed")

    if process == True:
        output = process_data()
    else:
        # data has already been processed
        output = np.load(file_save)


    return

def process_data():
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
    
    # choose gems of interest
    gem_list = ['138', '170', '155', '136', '133']
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
    path_save = os.path.join(path_data, "pkls", "processing_output.pkl")
    #output.write(path_save, format="PICKLE")
    return output


if __name__ == "__main__":
    main(process=False)
