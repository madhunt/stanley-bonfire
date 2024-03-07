#!/usr/bin/python3
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import numpy as np
import obspy
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
from obspy.imaging.cm import obspy_sequential
import os
import pandas as pd
import scipy as sci


def main(process=False, trace_plot=False):

    #FIXME path stuff
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    path_data = os.path.join(path_home, "data")
    #FIXME ugly
    path_save = os.path.join(path_data, "processed", "1HP_processed_output.npy")

    #FIXME arguments into main or better
    gem_list = ['138', '170', '155', '136', '133']  # hopefully back az towards south
    filter_type='highpass'
    filter_options = dict(freq=1.0)

    # load data
    data = load_data(path_data, gem_list=gem_list, 
                     filter_type=filter_type, **filter_options)
    
    # plot traces
    if trace_plot == True:
        plot_traces(data)

    if process == True:
        # fiter and beamform 
        output = process_data(data, path_save)

    else:
        # data has already been processed
        output = np.load(path_save)
    
    # correct backaz from 0 to 360 (instead of -180 to +180)
    output[:,3] = [output[i][3] if output[i][3]>=0 else output[i][3]+360 
                    for i in range(output.shape[0])]

    simple_plot(output, path_home)#[:100,:])
    #better_plot(output)
    


    return

def load_data(path_data, gem_list=None, filter_type=None, **filter_options):
    '''
    Loads in and pre-processes array data.
        Loads all miniseed files in a specified directory into an obspy stream. 
        Assigns coordinates to all traces. If specified, filters data. If specified, 
        only returns a subset of gems (otherwise, returns full array).
    INPUTS
        path_data : str : Path to data folder. Should contain all miniseed files 
            under 'mseed' dir, and coordinates in .csv file.
        gem_list : list of str : Optional. If specified, should list Gem SNs 
            of interest. If `None`, will return full array.
        filter_type : str : Optional. Obspy filter type. Includes 'bandpass', 
            'highpass', and 'lowpass'.
        filter_options : dict : Optional. Obspy filter arguments. For 'bandpass', 
            contains freqmin and freqmax. For low/high pass, contains freq.
    RETURNS
        data : obspy stream : Stream of data traces for full array, or specified 
            Gems. Stats include assigned coordinates.
    '''
    # paths to mseed and coordinates
    path_mseed = os.path.join(path_data, "mseed", "*.mseed")
    #TODO do this better...
    path_coords = os.path.join(path_data, "20240114_Bonfire_Gems.csv")

    # import data as obspy stream
    data = obspy.read(path_mseed)

    # import coordinates
    coords = pd.read_csv(path_coords)
    coords["Name"] = coords["Name"].astype(str) # SN of gem
    
    # get rid of any stations that don't have coordinates
    data_list = [trace for trace in data.traces 
                    if trace.stats['station'] in coords["Name"].to_list()]
    # convert list back to obspy stream
    data = obspy.Stream(traces=data_list)

    # assign coordinates to stations
    for _, row in coords.iterrows():
        sn = row["Name"]

        for trace in data.select(station=sn):
            trace.stats.coordinates = AttribDict({
                'latitude': row["Latitude"],
                'longitude': row["Longitude"],
                'elevation': row["Elevation"]
            }) 
    
    if filter_type == None:
        # don't filter data
        pass
    else:
        # filter the data
        data = data.filter(filter_type, **filter_options)
    
    if gem_list == None:
        # use full array
        return data
    else:
        # only use specified subset of gems
        data_subset = [trace for trace in data.traces if trace.stats['station'] in gem_list]
        data_subset = obspy.Stream(traces=data_subset)
        return data_subset
    

def plot_traces(data):


    return


def process_data(data, path_save):
    '''
    RETURNS
    output : np array : timestamp, relative power, absolute power, backazimuth, slowness
    '''
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

    output = array_processing(stream=data, **kwargs)

    print(output)

    # save output as pkl
    np.save(path_save, output)

    return output

def simple_plot(output, path_home):
    
    # only plot backaz with slowness near 3 s/km

    time = output[:,0]

    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)
    im0 = ax[0].scatter(time, output[:,3], c=output[:,1], alpha=0.6, 
                  vmin=output[:,1].min(), vmax=output[:,1].max(),
                  edgecolors='none', cmap='plasma')
    ax[0].set_ylabel("Backazimuth [$^o$]")
    ax[0].set_ylim([0, 360])
    ax[0].set_yticks(ticks=np.arange(0, 360+60, 60))
    cb0 = fig.colorbar(im0, ax=ax[0])
    cb0.set_label(label='Semblance')

    im1 = ax[1].scatter(time, output[:,4], c=output[:,1], alpha=0.6, 
                  vmin=output[:,1].min(), vmax=output[:,1].max(),
                  edgecolors='none', cmap='plasma')
    ax[1].set_ylabel("Slowness [s/km]")
    ax[1].set_yticks(ticks=np.arange(0, 5, 1))
    cb1 = fig.colorbar(im1, ax=ax[1])
    cb1.set_label(label='Semblance')

    for ax_i in ax:
        ax_i.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax[1].set_xlim([datetime.datetime(2024, 1, 15, 2, 30), datetime.datetime(2024, 1, 15, 4)])
    ax[1].set_xlabel("UTC Time")
    fig.autofmt_xdate()
    fig.suptitle("Stanley Bonfire")
    #FIXME path
    plt.savefig(os.path.join(path_home, "figures", f"1HP_backaz_slowness.png"), dpi=500)
    #plt.show()
    return

if __name__ == "__main__":
    main(process=False, 
         trace_plot=True)
