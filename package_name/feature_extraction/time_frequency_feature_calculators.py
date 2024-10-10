import pywt
import numpy as np

from scipy.signal import spectrogram, stft, chirp

from package_name.utils.utils import extract_features
from package_name.utils.decorators import name, exclude

# @exclude()
def teager_kaiser_energy_operator(signal):
    # https://doi.org/10.1016/j.dsp.2018.03.010
    # Calculate the TKEO
    # [x(n)] = x(n)^2 − x(n −k)x(n +k)        
    tkeo = np.roll(signal, -1) * np.roll(signal, 1) - signal ** 2
    # The first and last elements are not valid due to roll operation
    tkeo[0] = 0
    tkeo[-1] = 0
    return tkeo

@name("tkeo")
def calculate_tkeo_features(signal, sf_params, **kwargs):
    signal_tkeo = teager_kaiser_energy_operator(signal)
    return extract_features({"signal": signal_tkeo}, ["statistical"], sf_params)

@exclude()
@name("wave_coeffs_lvl_{}", "decomposition_level")
def calculate_wavelet_features(signal, wavelet, decomposition_level, sf_params, **kwargs):
    wavelet_coefficients = pywt.wavedec(signal, wavelet, level=decomposition_level)

    return extract_features({"signal": wavelet_coefficients}, ["statistical"], sf_params)

@exclude()
@name("spectrogram")
def calculate_spectrogram_features(signal, stft_window, nperseg, sf_params, **kwargs):
    f, t, Sxx = spectrogram(signal, window=stft_window, nperseg=nperseg)
    Sxx_flat = Sxx.flatten()

    return extract_features({"signal": Sxx_flat}, ["statistical"], sf_params)

# TODO: Make independent parameters
@name("stft")
def calculate_stft_features(signal, stft_window, nperseg, sf_params, **kwargs):
    f, t, Zxx = stft(signal, window=stft_window, nperseg=nperseg)
    Zxx_magnitude = np.abs(Zxx).flatten()
    print(Zxx_magnitude.shape)

    return extract_features({"signal": Zxx_magnitude}, ["statistical"], sf_params)
