import pywt
import numpy as np

from scipy.signal import spectrogram, stft, chirp

from package_name.feature_extraction import statistical_feature_calculators
from package_name.utils.utils import name, calculate_features


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
def extract_tkeo_features(signal, sf_params):
    signal_tkeo = teager_kaiser_energy_operator(signal)
    return calculate_features(("", signal_tkeo), ["statistical"], sf_params.get_settings_as_dict())

@name("wave_coeffs_lvl_{}", "decomposition_level")
def extract_wavelet_features(signal, wavelet, decomposition_level, sf_params):
    wavelet_coefficients = pywt.wavedec(signal, wavelet, level=decomposition_level)

    return calculate_features(("", wavelet_coefficients), ["statistical"], sf_params.get_settings_as_dict())

@name("spectrogram")
def extract_spectrogram_features(signal, stft_window, nperseg, sf_params):
    f, t, Sxx = spectrogram(signal, window=stft_window, nperseg=nperseg)
    Sxx_flat = Sxx.flatten()

    return calculate_features(("", Sxx_flat), ["statistical"], sf_params.get_settings_as_dict())

# TODO: Make independent parameters
@name("stft")
def extract_stft_features(signal, stft_window, nperseg, sf_params):
    f, t, Zxx = stft(signal, window=stft_window, nperseg=nperseg)
    Zxx_magnitude = np.abs(Zxx).flatten()
    print(Zxx_magnitude.shape)
    
    return calculate_features(("", Zxx_magnitude), ["statistical"], sf_params.get_settings_as_dict())
