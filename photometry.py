import os
import astropy
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from photutils.datasets import make_noise_image
from astropy.stats import sigma_clip
from scipy.stats import mode
from astropy.modeling import models, fitting
from astropy.table import Table
import photutils
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.profiles import RadialProfile
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
import astroscrappy
from astroscrappy import detect_cosmics
import glob
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from astropy.time import Time

data_dir = '/Users/elliottburdett/Downloads/arcsat'
image_list = []
arcsat_list = glob.glob(data_dir + '/*')
for arcsat in arcsat_list:
    if arcsat.endswith(".fits"):
        path = arcsat
        hdul = fits.open(arcsat)
        process = hdul[0].header['PROCESS']
        if process != 'Reduced with reduce_science_frame()':
            print("File wasn't reduced with reduce_science_frame()")
            continue
        image_list.append(arcsat) # Add the path to the image list
    else:
        print("File wasn't a fits file")

times = []
fluxes = []

for image in image_list: # Use the reduced science images here.
    mean, median, std = sigma_clipped_stats(fits.getdata(image), sigma=3.0) # Get the mean, median, and std of the image
    #print(f'Mean: {mean}, Median: {median}, Std: {std}')

    dao = DAOStarFinder(fwhm=3.0, threshold=5.0*std) # Create a DAOStarFinder object
    sources = dao(fits.getdata(image)) # Find sources in the image
    sources = sources[(sources['xcentroid'] > 470) & (sources['xcentroid'] < 530) & 
                 (sources['ycentroid'] > 470) & (sources['ycentroid'] < 530)]

    x_pos = sources['xcentroid'].tolist() # Get the x positions of the sources
    y_pos = sources['ycentroid'].tolist() # Get the y positions of the sources

    radius = 15
    position = list(zip(x_pos, y_pos)) # Use the first source's position
    print(f'Position: {position}, Radius: {radius}')
    
    science, header = fits.getdata(image, header=True) # Read the data
    
    time = header.get('DATE-OBS') # Get time of observation
    
    # Make aperture
    aperture = CircularAperture(position, r=radius)
    phot_table = aperture_photometry(science, aperture)
    # Make annulus
    annulus = CircularAnnulus(position, r_in=15, r_out=25)
    ap_stats = ApertureStats(science, annulus)
    
    # Get median background
    background = ap_stats.median
    
    # Calculate flux (background * area)
    aperture_area = aperture.area_overlap(science)
    sky_flux = background * aperture_area
    
    # Subtract background
    flux = phot_table['aperture_sum'] - sky_flux
    
    times.append(time) #Add to the time table
    fluxes.append(flux[0]) #Add to the flux table

time_flux_table = Table([times, fluxes], names=('time', 'flux'))

time_flux_table.sort('time')  # Sort times

times = time_flux_table['time']
fluxes = time_flux_table['flux']

times = Time(times).mjd  # Convert string dates to MJD

flux_norm = (time_flux_table['flux'] - np.min(time_flux_table['flux'])) / \
            (np.max(time_flux_table['flux']) - np.min(time_flux_table['flux']))

time_flux_table['flux_norm'] = flux_norm # Add normalized flux as a column

time_flux_table.write('output_table.csv', format='csv', overwrite=True)