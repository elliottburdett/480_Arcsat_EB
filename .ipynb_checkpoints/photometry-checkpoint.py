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
import glob

image_list = ['/home/jovyan/ccd_data/kelt-16-b-S001-R001-C084-r.fit'] # Change
times = []
fluxes = []

for image in image_list: # Use the reduced science images here.
    
    radius = 15 # Single radius value (not a list)
    position = (612, 612)
    
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