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

def create_median_bias(bias_list, median_bias_filename):
    """This function must:

    - Accept a list of bias file paths as bias_list.
    - Read each bias file and create a list of 2D numpy arrays.
    - Use a sigma clipping algorithm to combine all the bias frames using
      the median and removing outliers outside 3-sigma for each pixel.
    - Save the resulting median bias frame to a FITS file with the name
      median_bias_filename.
    - Return the median bias frame as a 2D numpy array.

    """
    
    list_of_arrays = []
    
    for bias in bias_list:
        bias_of_the_moment = fits.open(bias) # Open each file
        data = bias_of_the_moment[0].data.astype('f4') # Get a 2d data array from each file
        list_of_arrays.append(short_data) # Add each array to the list
        
    bias_images_masked = sigma_clip(list_of_arrays, cenfunc='median', sigma=3, axis=0) #Make a masked array out of the list based on sigma-clipping
    median_bias = np.ma.median(bias_images_masked, axis=0) # Use the masked array to compute median while ignoring masked values
    median_bias = median_bias.data # Turn it back into a regular array
    
    primary = fits.PrimaryHDU(data=median_bias, header=fits.Header('This is the median bias'))
    hdul = fits.HDUList([primary])
    hdul.writeto(median_bias_filename, overwrite=True)

    return median_bias
def create_median_dark(dark_list, bias_filename, median_dark_filename):
    """This function must:

    - Accept a list of dark file paths to combine as dark_list.
    - Accept a median bias frame filename as bias_filename (the one you created using
      create_median_bias).
    - Read all the images in dark_list and create a list of 2D numpy arrays.
    - Read the bias frame.
    - Subtract the bias frame from each dark image.
    - Divide each dark image by its exposure time so that you get the dark current
      per second. The exposure time can be found in the header of the FITS file.
    - Use a sigma clipping algorithm to combine all the bias-corrected dark frames
      using the median and removing outliers outside 3-sigma for each pixel.
    - Save the resulting dark frame to a FITS file with the name median_dark_filename.
    - Return the median dark frame as a 2D numpy array.

    """

    list_of_arrays = []

    bias_frame = fits.open(bias_filename)
    bias = bias_frame[0].data.astype('f4')
    
    for dark in dark_list:
        dark_of_the_moment = fits.open(dark) # Open each file
        data = dark_of_the_moment[0].data.astype('f4') # Get a 2d data array from each file
        short_data = short_data - bias # Subtract the bias frame from each dark image
        exptime = dark_of_the_moment[0].header['EXPTIME']
        short_data = short_data - exptime #subtract the exposure time
        list_of_arrays.append(short_data) # Add each array to the list
        
    dark_images_masked = sigma_clip(list_of_arrays, cenfunc='median', sigma=3, axis=0) #Make a masked array out of the list based on sigma-clipping
    median_dark = np.ma.median(dark_images_masked, axis=0) # Use the masked array to compute median while ignoring masked values
    median_dark = median_dark.data # Turn it back into a regular array
    
    primary = fits.PrimaryHDU(data=median_dark, header=fits.Header('This is the median dark'))
    hdul = fits.HDUList([primary])
    hdul.writeto(median_dark_filename, overwrite=True)

    return median_dark
def create_median_flat(
    flat_list,
    bias_filename,
    median_flat_filename,
    dark_filename=None,
):
    """This function must:

    - Accept a list of flat file paths to combine as flat_list. Make sure all
      the flats are for the same filter.
    - Accept a median bias frame filename as bias_filename (the one you created using
      create_median_bias).
    - Read all the images in flat_list and create a list of 2D numpy arrays.
    - Read the bias frame.
    - Subtract the bias frame from each flat image.
    - Optionally you can pass a dark frame filename as dark_filename and subtract
      the dark frame from each flat image (remember to scale the dark frame by the
      exposure time of the flat frame).
    - Use a sigma clipping algorithm to combine all the bias-corrected flat frames
      using the median and removing outliers outside 3-sigma for each pixel.
    - Create a normalised flat divided by the median flat value.
    - Save the resulting median flat frame to a FITS file with the name
      median_flat_filename.
    - Return the normalised median flat frame as a 2D numpy array.

    """
    list_of_arrays = []

    bias = fits.getdata(bias_filename)

    if dark_filename is not None:
        
        dark = fits.getdata(dark_filename)
        
        for flat in flat_list:
            data = fits.getdata(flat) # Open each file
            exptime = fits.open(flat)[0].header['EXPTIME']
            #print(exptime)
            short_data = short_data - bias - (dark * exptime)
            list_of_arrays.append(short_data) # Add each array to the list

    else:
        for flat in flat_list:
            data = fits.getdata(flat) # Open each file
            #print(exptime)
            short_data = short_data - bias
            list_of_arrays.append(short_data) # Add each array to the list

    #print(list_of_arrays)
    
    flat_images_masked = sigma_clip(list_of_arrays, cenfunc='median', sigma=3, axis=0) #Make a masked array out of the list based on sigma-clipping
    median_flat = np.ma.median(flat_images_masked, axis=0) # Use the masked array to compute median while ignoring masked values
    #print(median_flat)
    median_flat = median_flat.data # Turn it back into a regular array
    median_flat[median_flat == 0] = 1

    median_flat = median_flat / np.ma.median(median_flat)
    
    primary = fits.PrimaryHDU(data=median_flat, header=fits.Header('This is the median flat'))
    hdul = fits.HDUList([primary])
    hdul.writeto(median_flat_filename, overwrite=True)

    return median_flat
    
def reduce_science_frame(
    science_filename,
    median_bias_filename,
    median_flat_filename,
    median_dark_filename,
    reduced_science_filename="reduced_science.fits",
):
    """This function must:

    - Accept a science frame filename as science_filename.
    - Accept a median bias frame filename as median_bias_filename (the one you created
      using create_median_bias).
    - Accept a median flat frame filename as median_flat_filename (the one you created
      using create_median_flat).
    - Accept a median dark frame filename as median_dark_filename (the one you created
      using create_median_dark).
    - Read all files.
    - Subtract the bias frame from the science frame.
    - Subtract the dark frame from the science frame. Remember to multiply the
      dark frame by the exposure time of the science frame. The exposure time can
      be found in the header of the FITS file.
    - Correct the science frame using the flat frame.
    - Optionally, remove cosmic rays.
    - Save the resulting reduced science frame to a FITS file with the filename
      reduced_science_filename.
    - Return the reduced science frame as a 2D numpy array.

    """
    bias_frame = fits.open(median_bias_filename)
    bias = bias_frame[0].data.astype('f4')

    dark_frame = fits.open(median_dark_filename)
    dark = dark_frame[0].data.astype('f4')

    flat_frame = fits.open(median_flat_filename)
    flat = flat_frame[0].data.astype('f4')

    science_file = fits.open(science_filename)
    science = science_file[0].data.astype('f4')

    exptime = science_file[0].header['EXPTIME']

    science -= bias

    science -= (dark * exptime)

    science /= flat

    # Generate the cosmic ray mask and a cleaned image
    mask, reduced_science = detect_cosmics(science)
    
    new_header = fits.Header()
    new_header['DATE-OBS'] = science_header.get('DATE-OBS', 'UNKNOWN') # Preserve original observation time
    new_header['EXPTIME'] = exptime  # Keep exposure time
    new_header['PROCESS'] = 'Reduced with reduce_science_frame()' # Add processing note

    # Save the reduced frame with the preserved header
    fits.writeto(reduced_science_filename, reduced_science, header=new_header, overwrite=True)
    
def calculate_gain(files):
    """This function must:

    - Accept a list of files that you need to calculate the gain
      (two files should be enough, but what kind?).
    - Read the files and calculate the gain in e-/ADU.
    - Return the gain in e-/ADU.

    """

    a,b = files
    
    flat1 = fits.getdata(a).astype('f4')
    flat2 = fits.getdata(b).astype('f4')
    
    flat1_trim = flat1[1536:2560, 1536:2560]
    flat2_trim = flat2[1536:2560, 1536:2560]
   
    flat_diff = flat1_trim - flat2_trim
    flat_diff_var = np.var(flat_diff)

    mean_signal = 0.5 * np.mean(flat1_trim + flat2_trim)
    
    gain = 2 * mean_signal / flat_diff_var
    # print(f'Gain: {gain:.2f} e-/ADU')

    return float(gain)


def calculate_readout_noise(files, gain):
    """This function must:

    - Accept a list of files that you need to calculate the readout noise
      (two files should be enough, but what kind?).
    - Accept the gain in e-/ADU as gain. This should be the one you calculated
      in calculate_gain.
    - Read the files and calculate the readout noise in e-.
    - Return the readout noise in e-.

    """
    a,b = files
    bias1 = fits.getdata(a).astype('f4')
    bias2 = fits.getdata(b).astype('f4')
    
    bias1_trim = bias1[1536:2560, 1536:2560]
    bias2_trim = bias2[1536:2560, 1536:2560]

    bias_diff = bias1_trim - bias2_trim
    bias_diff_var = np.var(bias_diff)
    
    readout_noise_adu = np.sqrt(bias_diff_var / 2)
    readout_noise_e = readout_noise_adu * gain
    
    #print(f'Readout noise (ADU): {readout_noise_adu:.2f} ADU')
    #print(f'Readout noise (e-): {readout_noise_e:.2f} e-')

    return float(readout_noise_e)

#Begin the reduction process.

#Get the path lists:
data_dir = '/home/jovyan/ccd_data' # Change

bias_list = []
dark_list = []
flat_list = []
science_list = []

path_list = glob.glob(data_dir + '/*')
for path in path_list:
    if path.endswith(".fit"):
        hdul = fits.open(path)
        imagetype = hdul[0].header['IMAGETYP']
        if imagetype == 'BIAS':
            bias_list.append(path)
        elif imagetype == 'DARK':
            dark_list.append(path)
        elif imagetype == 'FLAT':
            flat_list.append(path)
        elif imagetype == 'LIGHT':
            science_list.append(path)
        else:
            print("File wasn't a bias, dark, or flat")
    else:
        print("File wasn't a fits file")

median_bias = create_median_bias(bias_list=bias_list, median_bias_filename='Median_Bias')
median_dark = create_median_dark(dark_list=dark_list, bias_filename='Median_Bias', median_dark_filename='Median_Dark')
median_flat = create_median_flat(flat_list=flat_list, bias_filename='Median_Bias', median_flat_filename='Median_Flat', dark_filename='Median_Dark')

counter = 0
for science in science_list: #Reduce all the science files
    counter += 1
    reduced_science = reduce_science_frame(science_filename=science_list[0], median_bias_filename='Median_Bias', median_flat_filename='Median_Flat', median_dark_filename='Median_Dark', reduced_science_filename=f"reduced_science_{counter}.fits")

gain_flats = (flat_list[0], flat_list[1])
gain = calculate_gain(gain_flats)
print(f'Gain is {gain} in e-/ADU')

noise_files = (bias_list[0], bias_list[1])
readout_noise = calculate_readout_noise(noise_files, gain)
print(f'Readout noise is {readout_noise} in e-')