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

#read in csv table
time_flux_table = Table.read('output_table.csv', format='csv')

times = time_flux_table['time']
fluxes = time_flux_table['flux']
flux_norms = time_flux_table['flux_norm']

frequency, power = LombScargle(times, fluxes).autopower(minimum_frequency=1/50, maximum_frequency=1/0.5)
dominant_period = 1 / frequency[np.argmax(power)]
print(frequency)
print(power)
print(f"Detected period: {dominant_period:.2f} days")

def fourier_series(t, *params):
    """General Fourier series with n harmonics.
    Args:
        t: Time array
        *params: [a0, a1, b1, a2, b2, ..., period]
    Returns:
        The Fourier series evaluated at t.
    """
    a0 = params[0]
    period = params[-1] # Last parameter is the period
    omega = 2 * np.pi / period
    
    series = a0 * np.ones_like(t)
    for i in range(1, len(params) - 1, 2):  # Loop over the coefficients
        a = params[i] if i < len(params) - 1 else 0  # Handle odd-length params
        b = params[i + 1] if (i + 1) < len(params) - 1 else 0
        series += a * np.cos(i * omega * t) + b * np.sin(i * omega * t)
    
    return series

p0 = [np.mean(flux_norms), 0, 1, 0, .5, 0, .2, dominant_period] # Enter initial guesses

params, _ = curve_fit(
    lambda t, a0, a1, b1, a2, b2, a3, b3, period: fourier_series(t, a0, a1, b1, a2, b2, a3, b3, period),
    times - times[0],
    flux_norms,
    p0=p0,
)

a0, a1, b1, a2, b2, a3, b3, period_fit = params
print(f"Fitted period: {period_fit:.2f} days")

plt.figure(figsize=(10, 5))
plt.scatter(times, flux_norms, color='red', label='Data', alpha=0.5)

# Generate the fitted curve
t_fit = np.linspace(times[0], times[-1], 1000)
flux_fit = fourier_series(t_fit - times[0], *params)

plt.plot(t_fit, flux_fit, 'b-', label=f'Fourier fit (P={period_fit:.2f} days)')
plt.xlabel('Time (MJD)', fontfamily='serif')
plt.ylabel('Normalized Flux', fontfamily='serif')
plt.title('Light curve with Fourier series fit', fontfamily='serif', fontweight='normal')
plt.legend(prop={'family': 'serif'})
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fourier_fit.png', dpi=300, bbox_inches='tight')
plt.show()