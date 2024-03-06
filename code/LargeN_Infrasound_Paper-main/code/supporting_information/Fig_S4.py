import matplotlib.pyplot as plt
import obspy
import numpy as np
import gemlog
from scipy.signal import spectrogram
from scipy.ndimage import median_filter
from scipy.stats import kurtosis
#%%
def spectrum(tr, criterion_function = 'default', runmed_radius_t = 0,
             runmed_radius_f = 0, nfft = 1024, overlap = 0.5,
             kurtosis_threshold = 0.5, window = 'hamming'):
    """Calculate spectrogram and Welch spectrum of an obspy trace, where either
    kurtosis thresholds or median filters can be used to exclude noisy times.

    Parameters
    ----------
    tr : obspy.Trace
        Data trimmed to period for which spectrum is calculated
    criterion_function : function, 'default', or None
        Function that returns True or False and determines whether a time window
        is too noisy to process (False) or acceptable (True). If 'default', test
        whether the time window's kurtosis is less than kurtosis_threshold. If 
        None, do not exclude any time windows.
    runmed_radius_t : int
        Radius of median filter in time dimension. If zero, do not apply a median
        filter over time.
    runmed_radius_f : int
        Radius of median filter in frequency dimension. If zero, do not apply a 
        median filter over frequency.
    nfft: int
        Length of fft to calculate. The minimum nonzero frequency in the output
        spectrum will be 1/(nfft * dt).
    overlap : float
        Proportion of overlap between adjacent time windows; must be >=0 and <1.
    kurtosis_threshold : float
        If using criterion_function = 'default', time windows with kurtosis 
        greater than this value are considered noisy and excluded.
    window : str
        Window function to use when calculating individual spectral estimates. 
        See help(scipy.signal.windows.getwindow) for options. Common choices
        include 'hamming', 'hann', 'blackmanharris', and 'boxcar' (rectangular).
    
    Returns
    -------
    Dictionary with following items:
    specgram: spectrogram of time period
    times: times of output spectrogram
    freqs: frequencies of output spectrum and spectrogram
    mean: mean spectrum (Welch's method)
    median: median spectrum
    stdev: standard deviation of mean spectrum

    Example
    -------
    import obspy, riversound
    import matplotlib.pyplot as plt
    st = obspy.read()
    spec_dict = riversound.spectrum(st[0])
    plt.loglog(spec_dict['freqs'], spec_dict['mean'])

    """
    if len(tr.data) == 0:
        return {'specgram':np.zeros([nfft,1]), 'freqs':np.arange(nfft), 'times':np.array([0]), 'mean':np.zeros(nfft), 'median':np.zeros(nfft), 'stdev': np.zeros(nfft)}
    
    if criterion_function == 'default':
        def criterion_function(x): return kurtosis(x) < kurtosis_threshold
    
    freqs, times, sg = spectrogram(tr.data, fs = tr.stats.sampling_rate, window = window, nperseg = nfft, noverlap = overlap * nfft, detrend = 'linear')

    ## If a criterion function is defined, apply it to all the time windows
    ## and change results for failing windows to NaN.
    if criterion_function is not None:
        for i, t in enumerate(times):
            j1 = np.round(t * tr.stats.sampling_rate - nfft/2)
            if not criterion_function(tr.data[int(j1):int(j1 + nfft)]):
                sg[:,i] = np.NaN

    ## Apply median filter if applicable. If more than half the values in a
    ## median filter window are nan, the output for that window will be too.
    if (runmed_radius_t > 0) or (runmed_radius_f > 0):
        kernel_size = (1 + 2*runmed_radius_f, 1 + 2*runmed_radius_t)
        w = ~np.isnan(sg[0,:])
        sg[:,w] = median_filter(sg[:,w], kernel_size)
        
    return {'specgram':sg, 'freqs':freqs, 'times':times, 'mean':np.nanmean(sg,1), 'median':np.nanmedian(sg,1), 'stdev': np.nanstd(sg, 1)}
#%%
plt.subplot(2,1,1)
## Shoshone Falls, recorded by #149 from 42.59333,-114.4014, about 200 m distance

## Shoshone high 2023-07-05T00_00_00..149..HDF.mseed
t1 = obspy.UTCDateTime('2023-07-05T09_00_00')
t2 = obspy.UTCDateTime('2023-07-05T10_00_00')
st = obspy.read('imported_data/2023-07-05T00_00_00..149..HDF.mseed').slice(t1, t2)
st = gemlog.deconvolve_gem_response(st)
spec_info = spectrum(st[0])
plt.loglog(spec_info['freqs'], spec_info['mean'], label = '92 m^3/s')

## Shoshone low 2023-05-08T00_00_00..149..HDF.mseed
t1 = obspy.UTCDateTime('2023-05-08T09_00_00')
t2 = obspy.UTCDateTime('2023-05-08T10_00_00')
st = obspy.read('imported_data/2023-05-08T00_00_00..149..HDF.mseed').slice(t1, t2)
st = gemlog.deconvolve_gem_response(st)
spec_info = spectrum(st[0])
plt.loglog(spec_info['freqs'], spec_info['mean'], label = '10 m^3/s')

plt.xlim([0.6, 35])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (Pa$^2$/Hz)')
plt.title('A. Shoshone Falls', loc = 'left')
plt.legend(loc = 'upper left')

plt.subplot(2,1,2)
## LFF high 2023-06-02T00_00_00..175..HDF.mseed
t1 = obspy.UTCDateTime('2023-06-02T09_00_00')
t2 = obspy.UTCDateTime('2023-06-02T10_00_00')
st = obspy.read('imported_data/2023-06-02T00_00_00..175..HDF.mseed').slice(t1, t2)
st = gemlog.deconvolve_gem_response(st)
spec_info = spectrum(st[0])
plt.loglog(spec_info['freqs'], spec_info['mean'], label = 'High Flow')

t1 = obspy.UTCDateTime('2023-04-28T09_00_00')
t2 = obspy.UTCDateTime('2023-04-28T10_00_00')
st = obspy.read('imported_data/2023-04-28T00_00_00..175..HDF.mseed').slice(t1, t2)
st = gemlog.deconvolve_gem_response(st)
spec_info = spectrum(st[0])
plt.loglog(spec_info['freqs'], spec_info['mean'], label = 'Low Flow')

plt.xlim([0.6, 35])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (Pa$^2$/Hz)')
plt.title('B. Lady Face Falls', loc = 'left')
plt.legend(loc = 'upper left')


plt.gcf().set_size_inches(6.5, 7.5)
plt.tight_layout()


plt.savefig('Figures/Fig_S4.jpg',dpi=300)




