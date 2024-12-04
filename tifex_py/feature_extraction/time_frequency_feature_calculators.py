import pywt
import numpy as np

from scipy.signal import spectrogram, stft, chirp

from tifex_py.utils.extraction_utils import extract_features
from tifex_py.utils.decorators import name, exclude

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
def calculate_tkeo_features(signal, tkeo_sf_params, **kwargs):
    signal_tkeo = teager_kaiser_energy_operator(signal)
    return extract_features({"signal": signal_tkeo}, "statistical", tkeo_sf_params)

@name("wave_coeffs_lvl_{}", "decomposition_level")
def calculate_wavelet_features(signal, wavelet, decomposition_level, wavelet_sf_params, **kwargs):
    wavelet_coefficients = pywt.wavedec(signal, wavelet, level=decomposition_level)

    features = []
    for i in range(len(wavelet_coefficients)):
        features.append(extract_features({"signal": wavelet_coefficients[i]}, "statistical", wavelet_sf_params))

    return features

@name("spectrogram")
def calculate_spectrogram_features(signal, stft_window, nperseg, spectogram_sf_params, **kwargs):
    f, t, Sxx = spectrogram(signal, window=stft_window, nperseg=nperseg)
    Sxx_flat = Sxx.flatten()

    return extract_features({"signal": Sxx_flat}, "statistical", spectogram_sf_params)

# TODO: Make independent parameters
@name("stft")
def calculate_stft_features(signal, stft_window, nperseg, stft_sf_params, **kwargs):
    f, t, Zxx = stft(signal, window=stft_window, nperseg=nperseg)
    Zxx_magnitude = np.abs(Zxx).flatten()

    return extract_features({"signal": Zxx_magnitude}, "statistical", stft_sf_params)
