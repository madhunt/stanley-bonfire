# Code for Scamfer & Anderson (2023): Exploring background noise with a large-N infrasound array: waterfalls, thunderstorms, and earthquakes
## Geophysical Research Letters

### Contents
This repository contains Python code and data to reproduce Fig. 2-4 in the paper (including panels of Fig. 5 should WWLLN data be obtained), and Supporting Information Fig. S1-S5.

Folder 'code/':
The Folder 'code/' contains Python code to fully reproduce Fig. 2, and can reproduce Fig. 3-4 without the maps.
Individual panels (3) of Fig. 5 can be reproduced should you obtain WWLLN data and place it in the 'imported_data/' folder.<br>
World Wide Lightning Location Network data is available at nominal cost from http://wwlln.net.
When requesting WWLLN data, search the box (50N,126W) - (36N,102.5W) for the following days:
29,30 April 2020, 
19,20 May 2020, 
5,6 June 2020.<br>
The subfolder 'code/supporting_information/' also contains code to reproduce the Supporting Information Fig. S1-S5.

The Folder 'Figures/' contains image files of the figures in the paper.<br>
The subfolder 'code/maps/' contains QGIS map files and image files with the maps for Fig. 3-4.

The Folder 'imported_data/' contains data files including Snake River discharge, Stanely Ranger station temperature data, and infrasound data for Fig. S4. Should you obtain WWLLN data this is where you should place the files.

The Folder 'pickle_files/' contains multiple data files seperated into subfolders (e.g. N_3, N_9, N_full, etc.) These are beamformed results of array_processing() in ObsPy. Array_processing parameters can be found in the 'Methods' section and Supporting Information. We have included the products of array_processing() because producing them using raw data is computationally expensive and can take very long periods of time.

The File XP.PARK is an XML file containing metadata that describes the data collected by geophysical instrumentation (PARK infrasound array).

### Before running code:
For the code to run successfully, you must have Python and dependent packages installed. An easy way to do this uses Anaconda or Miniconda (which you must install beforehand):
```
conda deactivate
conda create -y -n large_N_infrasound python=3.9.12 matplotlib=3.6.2 obspy=1.4.0 pandas=1.5.2 numpy=1.21.6 scipy=1.9.3 gemlog=1.7.5
conda activate large_N_infrasound
```
These commands should run on any python terminal or IDE. When running code, you must set the 'LargeN_Infrasound_Paper-main/' as your working directory.


