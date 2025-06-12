import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

frequency, power = LombScargle(times, fluxes).autopower(minimum_frequency=1/50, maximum_frequency=1/0.5)
dominant_period = 1 / frequency[np.argmax(power)]
print(frequency)
print(power)
print(f"Detected period: {dominant_period:.2f} days")

def fourier_series(t, a0, a1, b1, period):
    """Fourier series with 1 harmonic (sin + cos)."""
    omega = 2 * np.pi / period
    return a0 + a1 * np.cos(omega * t) + b1 * np.sin(omega * t)

p0 = [np.mean(fluxes), 0, amplitude, dominant_period]

params, _ = curve_fit(fourier_series, times - times[0], fluxes, p0=p0) # Make a curve fit
a0, a1, b1, period_fit = params

print(f"Fitted period: {period_fit:.2f} days")

plt.figure(figsize=(10, 5))

plt.scatter(times, fluxes, color='red', label='Data') # Add the data

# Fourier fit
t_fit = np.linspace(times[0], times[-1], 1000)
flux_fit = fourier_series(t_fit - times[0], a0, a1, b1, period_fit)
plt.plot(t_fit, flux_fit, 'b-', label=f'Fourier fit (P={period_fit:.2f} days)')

plt.xlabel('Time (MJD)')
plt.ylabel('Flux')
plt.legend()
plt.grid(alpha=0.3)
plt.show()