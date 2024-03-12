#!/usr/bin/python3
import datetime
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
import numpy as np
import obspy
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
from obspy.imaging.cm import obspy_sequential
import os
import pandas as pd


def main(process=False, trace_plot=False, backaz_plot=False):

    #FIXME path stuff
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    path_data = os.path.join(path_home, "data")
    #FIXME ugly
    path_save = os.path.join(path_data, "processed", "1HP_processed_output.npy")

    #FIXME arguments into main or better
    gem_list = ['138', '170', '155', '136', '150']#, '133']  # hopefully back az towards south
    #FIXME removed 133 because data didnt start until 01-15T02
    filter_type='highpass'
    filter_options = dict(freq=1.0)

    

    # load data
    data = load_data(path_data, gem_list=gem_list, 
                     filter_type=filter_type, **filter_options)
    
    # plot individual traces
    if trace_plot == True:
        print("Plotting Traces")
        plot_traces(data, path_home)

    if process == True:
        print("Processing Data")
        # fiter and beamform 
        output = process_data(data, path_save, time_start=None, time_end=None)
    else:
        print("Loading Data")
        # data has already been processed
        output = np.load(path_save)
    
    if backaz_plot == True:
        print("Plotting Backazimuth and Slowness")
        plot_backaz_slowness(output, path_home)#[:100,:])
    
    return

def load_data(path_data, gem_list=None, filter_type=None, **filter_options):
    '''
    Loads in and pre-processes array data.
        Loads all miniseed files in a specified directory into an obspy stream. 
        Assigns coordinates to all traces. If specified, filters data. If specified, 
        only returns a subset of gems (otherwise, returns full array).
    INPUTS
        path_data : str : Path to data folder. Should contain all miniseed files 
            under 'mseed' dir, and coordinates in .csv file(s).
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
    path_coords = glob.glob(os.path.join(path_data, "*.csv" ))#"20240114_Bonfire_Gems.csv")

    # import data as obspy stream
    data = obspy.read(path_mseed)

    # import coordinates
    coords = pd.DataFrame()
    for file in path_coords:
        coords = pd.concat([coords, pd.read_csv(file)])
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
        # filter data
        data = data.filter(filter_type, **filter_options)

    # merge dates (discard overlaps and leave gaps)
    data = data.merge(method=0)
    
    if gem_list == None:
        # use full array
        return data
    else:
        # only use specified subset of gems
        data_subset = [trace for trace in data.traces if trace.stats['station'] in gem_list]
        data_subset = obspy.Stream(traces=data_subset)
        return data_subset
    

def plot_traces(data, path_home):

    #NOTE this will need to change if ever plotting more traces**
    #NOTE can also concat different days for same gem

    # define number of traces
    n = len(data)
    
    fig, ax = plt.subplots(n, 1, sharex=True, sharey=True, tight_layout=True)
    color = cm.rainbow(np.linspace(0, 1, n))

    for i, trace in enumerate(data):
        # plot trace
        ax[i].plot(trace.times("matplotlib"), trace.data, c=color[i])
        ax[i].grid()
        ax[i].set_ylabel(trace.stats["station"])
        ax[i].xaxis_date()

        #NOTE change this
        ax[i].set_ylim([-100, 100])

    # label bottom x-axis
    ax[n-1].set_xlabel("UTC Time")
    fig.autofmt_xdate()
    fig.suptitle("Individual Gem Traces")
    plt.savefig(os.path.join(path_home, "figures", f"traces.png"), dpi=500)
    #plt.show()
    return


def process_data(data, path_save, time_start=None, time_end=None):
    '''
    RETURNS
    output : np array : timestamp, relative power, absolute power, backazimuth (0-360), slowness
    '''

    # if times are not provided, use max/min start and end times from gems
    if time_start == None:
        # specify start time
        time_start = max([trace.stats.starttime for trace in data])
    if time_end == None:
        time_end = min([trace.stats.endtime for trace in data])

    #time_start = obspy.UTCDateTime("20240114T22:30:00")
    #time_end = obspy.UTCDateTime("20240114T14:23:59")

    process_kwargs = dict(
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

    output = array_processing(stream=data, **process_kwargs)

    # correct backaz from 0 to 360 (instead of -180 to +180)
    output[:,3] = [output[i][3] if output[i][3]>=0 else output[i][3]+360 
                    for i in range(output.shape[0])]

    #print(output)

    # save output as pkl
    np.save(path_save, output)

    return output

def plot_backaz_slowness(output, path_home):
    
    #TODO have this as a func with user input slow min/max at some point
    # only plot backaz with slowness near 3 s/km
    slow_min = 2.9
    slow_max = 3.1
    output_constrain = []
    for col in output.T:
        col_constrain = [col[i] for i in range(len(col)) if slow_min < output[:,4][i] < slow_max]
        col_constrain = np.array(col_constrain)
        output_constrain.append(col_constrain)
    output_constrain = np.array(output_constrain).T

    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)
    im0 = simple_beamform_plot('backaz', output_constrain, fig, ax[0])
    im1 = simple_beamform_plot('slowness', output, fig, ax[1])

    for ax_i in ax:
        ax_i.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    #ax[1].set_xlim([datetime.datetime(2024, 1, 15, 2, 30), datetime.datetime(2024, 1, 15, 4)])
    ax[1].set_xlabel("UTC Time")
    fig.autofmt_xdate()
    fig.suptitle("Stanley Bonfire")
    #FIXME path
    plt.savefig(os.path.join(path_home, "figures", f"1HP_backaz_slowness.png"), dpi=500)
    #plt.show()
    return

#TODO think about this... there's probably a neater way to consolidate and reuse this func
# will leave for now and wait for other plotting needs
def simple_beamform_plot(plot_type, output, fig, ax):
    '''
    Plots beackazimuth or slowness on given figure, along with colorbar. Assumes output array 
    includes columns in the same order as output of array_processing().
    INPUTS
        plot_type : str : 'backaz' or 'slowness'
        output : np array : array with 5 rows containing output of array_processing() function. 
            This includes time, semblance, abs power, backazimuth, and slowness.
        fig, ax : pyplot handles(?) : handles(?) to figure and axes
    RETURNS
        im : handle(?) : handle to image
    '''
    if plot_type == 'backaz':
        yvar = output[:,3]
        ax.set_ylabel("Backazimuth [$^o$]")
        ax.set_ylim([0, 360])
        ax.set_yticks(ticks=np.arange(0, 360+60, 60))
    elif plot_type == 'slowness':
        yvar = output[:,4]
        ax.set_ylabel("Slowness [s/km]")
        ax.set_yticks(ticks=np.arange(0, int(max(output[:,4]))+1, 1))
    else:
        raise Exception("Plot type not supported!")

    im = ax.scatter(output[:,0], yvar, c=output[:,1],
                    alpha=0.6, edgecolors='none', cmap='plasma',
                    vmin=min(output[:,1]), vmax=max(output[:,1]))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Semblance")
    return im


if __name__ == "__main__":
    main(process=False, 
         trace_plot=False,
         backaz_plot=True)
