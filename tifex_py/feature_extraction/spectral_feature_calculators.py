import numpy as np
from scipy.signal import welch, find_peaks
from scipy.integrate import simpson
from scipy.stats import entropy, skew, kurtosis, trim_mean, mode
import librosa

from tifex_py.utils.decorators import name, exclude

@name("spectral_centroid_order_{}", "centroid_orders")
def calculate_spectral_centroid(freqs, magnitudes, centroid_orders=1, **kwargs):
    """
    Calculates the spectral centroid of the given spectrum.

    The spectral centroid is a measure that indicates where the center of mass of the spectrum is located.
    It is often associated with the perceived brightness of a sound. This function computes the spectral
    centroid by taking the weighted mean of the frequencies, with the magnitudes as weights.

    Parameters:
    ----------
    freqs : np.array
        An array of frequencies corresponding to the spectrum bins.
    magnitudes : np.array
        An array of magnitude values of the spectrum at the corresponding frequencies.
    order : int, optional
        The order of the centroid calculation. Default is 1, which calculates the standard spectral centroid (mean frequency).
        Higher orders can be used for other types of spectral centroid calculations.

    Returns:
    -------
    np.array
        An array containing the calculated spectral centroid. The array is of length 1 for consistency in return type.
        
    Reference:
    ---------
        - Kulkarni, N., & Bairagi, V. (2018). Use of Complexity Features for Diagnosis of Alzheimer Disease. EEG-Based Diagnosis 
        of Alzheimer Disease, 47–59. https://doi.org/10.1016/B978-0-12-815392-5.00004-6
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    if isinstance(centroid_orders, int):
        return np.sum(magnitudes * (freqs ** centroid_orders)) / np.sum(magnitudes)
    else:
        centroids = []
        for order in centroid_orders:
            centroids.append(np.sum(magnitudes * (freqs ** order)) / np.sum(magnitudes))
        return np.array(centroids)

@name("spectral_variance")
def calculate_spectral_variance(freqs, magnitudes, **kwargs):
    """
    Calculates the spectral variance (one form of spectral spread) of the given spectrum.

    The spectral variance is a measure of the spread of the spectrum around its centroid.
    It quantifies how much the frequencies in the spectrum deviate from the spectral centroid.

    Parameters:
    ----------
    freqs : np.array
        An array of frequencies corresponding to the spectrum bins.
    magnitudes : np.array
        An array of magnitude values of the spectrum at the corresponding frequencies.

    Returns:
    -------
    float
        The calculated spectral variance.
    
    Reference:
    ---------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    mean_frequency = calculate_spectral_centroid(freqs, magnitudes)
    spectral_variance = np.sum(((freqs - mean_frequency) ** 2) * magnitudes) / np.sum(magnitudes)
    return spectral_variance


@name("spectral_skewness")
def calculate_spectral_skewness(freqs, magnitudes, **kwargs):
    """
    Calculates the spectral skewness of the given spectrum.

    Spectral skewness is a measure of the asymmetry of the distribution of frequencies in the spectrum
    around the spectral centroid. It indicates whether the spectrum is skewed towards higher or lower
    frequencies.

    Parameters:
    ----------
    freqs : np.array
        An array of frequencies corresponding to the spectrum bins.
    magnitudes : np.array
        An array of magnitude values of the spectrum at the corresponding frequencies.

    Returns:
    -------
    float
        The calculated spectral skewness.
    
    Reference:
    ---------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    mu1 = calculate_spectral_centroid(freqs, magnitudes, centroid_orders=1)
    mu2 = calculate_spectral_bandwidth(freqs, magnitudes, bandwidth_orders=2)
    spectral_skewness = np.sum(magnitudes * (freqs - mu1) ** 3) / (np.sum(magnitudes) * mu2 ** 3)
    return spectral_skewness

@name("spectral_kurtosis")
def calculate_spectral_kurtosis(freqs, magnitudes, **kwargs):
    """
    Calculate the spectral kurtosis of the given spectrum.

    Spectral kurtosis measures the "tailedness" or peakiness of the frequency distribution around the spectral centroid.
    It quantifies how outlier-prone the spectrum is and reflects the degree of concentration of the spectral energy.
    A higher kurtosis value indicates a more peaked distribution with heavy tails.

    Parameters:
    ----------
    freqs : np.array
        An array of frequencies corresponding to the spectrum bins.
    magnitudes : np.array
        An array of magnitude values of the spectrum at the corresponding frequencies.

    Returns:
    -------
    float
        The calculated spectral kurtosis.
    
    Reference:
    ---------
        - Antoni, J. (2006). The spectral kurtosis: a useful tool for characterising non-stationary signals. Mechanical Systems 
        and Signal Processing, 20(2), 282–307. https://doi.org/10.1016/J.YMSSP.2004.09.001
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    mu1 = calculate_spectral_centroid(freqs, magnitudes, centroid_orders=1)
    mu2 = calculate_spectral_bandwidth(freqs, magnitudes, bandwidth_orders=2)
    spectral_kurtosis = np.sum(magnitudes * (freqs - mu1) ** 4) / (np.sum(magnitudes) * mu2 ** 4)
    return spectral_kurtosis

@name("median_frequency")
def calculate_median_frequency(freqs_psd, psd, **kwargs):
    """
    Calculate the cumulative distribution function (CDF) of the PSD

    Parameters:
    ----------
    freqs_psd : np.array
        An array of frequencies corresponding to the PSD bins.
    psd : np.array
        An array of power spectral density values at the corresponding frequencies.

    Returns:
    -------
    float
        The calculated median frequency.
        
    Reference:
    ---------
        - Chung, W. Y., Purwar, A., & Sharma, A. (2008). Frequency domain approach for activity classification 
        using accelerometer. Proceedings of the 30th Annual International Conference of the IEEE Engineering in 
        Medicine and Biology Society, EMBS’08 - “Personalized Healthcare through Technology,” 1120–1123. 
        https://doi.org/10.1109/IEMBS.2008.4649357
    """
    cdf = np.cumsum(psd)
    median_freq = freqs_psd[np.searchsorted(cdf, cdf[-1] / 2)]
    return median_freq

@name("spectral_flatness")
def calculate_spectral_flatness(magnitudes, **kwargs):
    """
    Calculate the spectral flatness of a given spectrum.

    Spectral flatness measures the uniformity of signal energy in the frequency domain.
    It is often used to distinguish between noise and tonal signals. It can also be referred to as Wiener's entropy.
    The spectral flatness is defined as the geometric mean of the FT of a signal normalized by its arithmetic mean

    Parameters:
    ----------
    magnitudes : np.array
        An array of magnitude values of the spectrum.

    Returns:
    -------
    float
        The calculated spectral flatness.

    Reference:
    --------
        - Sayeed, A. M., Papandreou-Suppappola, A., Suppappola, S. B., Xia, X. G., Hlawatsch, F., Matz, G., Boashash, B., 
        Azemi, G., & Khan, N. A. (2016). Detection, Classification, and Estimation in the (t,f) Domain. Time-Frequency 
        Signal Analysis and Processing: A Comprehensive Reference, 693–743. https://doi.org/10.1016/B978-0-12-398499-9.00012-1
    """
    spectral_flatness = np.exp(np.mean(np.log(magnitudes))) / np.mean(magnitudes)
    return spectral_flatness

@name("spectral_slope_logarithmic")
def calculate_spectral_slope_logarithmic(freqs, magnitudes, **kwargs):
    """
    Calculate the logarithmic spectral slope of the given spectrum.

    The logarithmic spectral slope provides a measure of the rate at which the spectrum's magnitude changes
    across frequencies on a logarithmic scale.

    Parameters:
    ----------
    freqs : np.array
        An array of frequencies corresponding to the magnitude spectrum bins.
    magnitudes : np.array
        An array of magnitude values of the spectrum at the corresponding frequencies.

    Returns:
    -------
    float
        The calculated logarithmic spectral slope. The array is of length 1 for consistency in return type.

    Reference:
    ---------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    slope = np.polyfit(freqs, np.log(magnitudes), 1)[0]
    return slope

@name("spectral_slope_logarithmic_psd")
def calculate_spectral_slope_logarithmic_psd(freqs_psd, psd, **kwargs):
    """
    Calculate the logarithmic spectral slope of the given Power Spectral Density (PSD) values.

    The logarithmic spectral slope provides a measure of the rate at which the PSD changes
    across frequencies on a logarithmic scale.

    Parameters:
    ----------
    freqs_psd : np.array
        An array of frequency values.
    psd : np.array
        An array of Power Spectral Density (PSD) values corresponding to the frequencies.

    Returns:
    -------
    float
        The slope of the linear fit.
        
    Reference:
    ---------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    return calculate_spectral_slope_logarithmic(freqs_psd, psd)

# TODO: Figure out how to handle difference parameter inputs
@name("spectral_slope_linear")
def calculate_spectral_slope_linear(freqs, magnitudes, **kwargs):
    """
    Calculate the spectral slope of a signal given its frequencies and magnitudes.

    The spectral slope is determined by fitting a linear regression line to the 
    frequency-magnitude data. The slope of this line indicates how the magnitudes
    change with respect to frequency.

    Parameters:
    ---------
    freqs: np.array 
        An array of frequency values.
    magnitudes: np.array
        An array of magnitude values corresponding to the frequencies.

    Returns:
    -------
    float
        The slope of the linear fit.
        
    Reference:
    ---------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    slope = np.polyfit(freqs, magnitudes, 1)[0]
    return slope

@name("spectral_slope_linear_psd")
def calculate_spectral_slope_linear_psd(freqs_psd, psd, **kwargs):
    """
    TODO: Update
    """
    return calculate_spectral_slope_linear(freqs_psd, psd)

@name("peak_freq_{}", "n_dom_freqs")
def calculate_peak_frequencies(freqs_psd, psd, n_dom_freqs, **kwargs):
    """
    Identifies the peak frequencies from the given Power Spectral Density (PSD) values.

    Parameters:
    ----------
    freqs_psd: np.array
        An array of frequency values.
    psd: np.array
        An array of Power Spectral Density (PSD) values corresponding to the frequencies.
    n_dom_freqs: int
        The number of dominant frequencies to identify.

    Returns:
    -------
    np.array
        A numpy array containing the peak frequencies.
    
    Reference:
    ---------
        - Murthy, V. K., Julian Haywood, L., Richardson, J., Kalaba, R., Salzberg, S., Harvey, G., 
        & Vereeke, D. (1971). Analysis of power spectral densities of electrocardiograms. Mathematical 
        Biosciences, 12(1–2), 41–51. https://doi.org/10.1016/0025-5564(71)90072-1
    """
        # Validate inputs
    if n_dom_freqs <= 0:
        raise ValueError("n_dom_freqs must be greater than 0.")
    if n_dom_freqs > len(psd):
        raise ValueError(f"n_dom_freqs ({n_dom_freqs}) cannot exceed the length of the PSD array ({len(psd)}).")

    peak_frequencies = freqs_psd[np.argsort(psd)[-n_dom_freqs:][::-1]]
    return np.array(peak_frequencies)

@name("edge_freq_thresh_{}", "cumulative_power_thresholds")
def calculate_spectral_edge_frequency(freqs_psd, psd, cumulative_power_thresholds, **kwargs):
    """
    Calculate the spectral edge frequencies for given cumulative power thresholds.

    The spectral edge frequency is the frequency below which a certain percentage 
    of the total power of the signal is contained. This function calculates the 
    spectral edge frequencies for multiple thresholds provided in 
    `cumulative_power_thresholds`.

    Parameters:
    ----------
    freqs_psd: np.array
        An array of frequency values.
    psd: np.array
        An array of Power Spectral Density (PSD) values corresponding to the frequencies.

    Returns:
    -------
    np.array
        A numpy array containing the spectral edge frequencies for each threshold.
    
    Reference:
    ---------
        - Drummond, J. C., Brann, C. A., Perkins, D. E., & Wolfe, D. E. (1991). A comparison of median frequency, 
        spectral edge frequency, a frequency band power ratio, total power, and dominance shift in the determination of 
        depth of anesthesia [Article]. Acta Anaesthesiologica Scandinavica., 35(8), 693–699. https://doi.org/10.1111/j.1399-6576.1991.tb03374.x
    """
    # A special case would be roll-off frequency (threshold = .85)
    feats = []
    cumulative_power = np.cumsum(psd) / np.sum(psd)
    for threshold in cumulative_power_thresholds:
        feats.append(freqs_psd[np.argmax(cumulative_power >= threshold)])
    return np.array(feats)

@name(["total_band_power", "absolute_band_power_{}", "relative_band_power_{}"], [0, "f_bands", "f_bands"])
def calculate_band_power(freqs_psd, psd, f_bands, **kwargs):
    """
    Calculates the total power, band absolute powers and band relative powers in specified frequency bands.
    
    Parameters:
    ----------
    freqs_psd: np.array
        An array of frequency values.
    psd: np.array
        An array of Power Spectral Density (PSD) values corresponding to the frequencies.

    Returns:
    -------
    np.array
        An array containing the total power, followed by the absolute and relative
        power for each specified frequency band.

    Reference:
    ---------
        - https://www.mathworks.com/help/signal/ref/bandpower.html
        - https://raphaelvallat.com/bandpower.html
    """
    # The features array for storing the total power, band absolute powers, and band relative powers
    feats = []
    freq_res = freqs_psd[1] - freqs_psd[0]  # Frequency resolution
    # Calculate the total power of the signal
    try:
        feats.append(simpson(psd, dx=freq_res))
    except:
        feats.append(np.nan)
    # Calculate band absolute and relative power
    for f_band in f_bands:
        try:
            # Keeping the frequencies within the band
            idx_band = np.logical_and(freqs_psd >= f_band[0], freqs_psd < f_band[1])
            # Absolute band power by integrating PSD over frequency range of interest
            feats.append(simpson(psd[idx_band], dx=freq_res))
            # Relative band power
            feats.append(feats[-1] / feats[0])
        except:
            feats.extend([np.nan, np.nan])
    return np.array(feats)

@name("spectral_entropy")
def calculate_spectral_entropy(psd, **kwargs):
    """
    Calculate the spectral entropy of a Power Spectral Density (PSD) array.

    Spectral entropy is a measure of the disorder or complexity of a signal's 
    frequency distribution. It is calculated by normalizing the PSD values to 
    form a probability distribution and then computing the entropy of this 
    distribution.
    
    Parameters:
    -----------
    psd: np.array
        An array of Power Spectral Density (PSD) values corresponding to the frequencies.
        
    Returns:
    --------
    float
        The spectral entropy value.
        
    Reference:
    ---------
        - Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S., Iyama, A., Katsuda, Y., & Hirano, M. (1991). Quantification
        of EEG irregularity by use of the entropy of the power spectrum. Electroencephalography and Clinical Neurophysiology, 79(3),
        204–210. https://doi.org/10.1016/0013-4694(91)90138-T
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    try:
        # Formula from Matlab doc
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
    except:
        spectral_entropy = np.nan
    return spectral_entropy

@name("spectral_contrast_band_{}", "f_bands")
def calculate_spectral_contrast(freqs_psd, psd, f_bands, **kwargs):
    """
    Calculate the spectral contrast of a Power Spectral Density (PSD) array.

    Spectral contrast measures the difference between peaks and valleys in the spectrum
    for specified frequency bands. It reflects the dynamic range of the frequency content
    of the signal.
    
    Parameters:
    ----------
    freqs_psd: np.array
        An array of frequency values.
    psd: np.array
        An array of Power Spectral Density (PSD) values corresponding to the frequencies.
    f_bands: list
        A list of tuples specifying the frequency bands for which to calculate the spectral contrast.

    Returns:
    -------
    np.array:
        An array containing the spectral contrast values for each specified frequency band.

    Reference:
    ----------
        - Music type classification by spectral contrast feature. (n.d.). Retrieved September 16,
        2024, from https://www.researchgate.net/publication/313484983_Music_type_classification_by_spectral_contrast_feature
        - McFee, B., Matt McVicar, Daniel Faronbi, Iran Roman, Matan Gover, Stefan Balke, Scott Seyfarth, Ayoub Malek, 
        Colin Raffel, Vincent Lostanlen, Benjamin van Niekirk, Dana Lee, Frank Cwitkowitz, Frank Zalkow, Oriol Nieto, 
        Dan Ellis, Jack Mason, Kyungyun Lee, Bea Steers, … Waldir Pimenta. (2024). librosa/librosa: 0.10.2.post1 (0.10.2.post1). 
        Zenodo. https://doi.org/10.5281/zenodo.11192913
    """
    feats = []
    for f_band in f_bands:
        try:
            idx_band = np.logical_and(freqs_psd >= f_band[0], freqs_psd < f_band[1])
            peak = np.max(psd[idx_band])
            valley = np.min(psd[idx_band])
            contrast = peak - valley
            feats.append(contrast)
        except:
            feats.append(np.nan)
    return np.array(feats)

@name("spectral_bandwidth_order_{}", "bandwidth_orders")
def calculate_spectral_bandwidth(freqs, magnitudes, bandwidth_orders, **kwargs):
    """
    Calculate the spectral bandwidth of a given frequency spectrum.
    
    Spectral bandwidth is a measure of the width of the spectrum, indicating the spread 
    of the magnitudes across frequencies. This function computes the spectral bandwidth 
    based on the specified order, where:
    - The 1st order spectral bandwidth corresponds to the spectral mean deviation.
    - The 2nd order spectral bandwidth corresponds to the spectral standard deviation.
    
    Parameters:
    ----------
    freqs : np.array
        An array of frequencies corresponding to the magnitude spectrum bins.
    magnitudes : np.array
        An array of magnitude values of the spectrum at the corresponding frequencies.

    bandwidth_orders : int
    The order of the spectral bandwidth calculation. The order defines the type of 
    deviation being measured:
    - 1 for spectral mean deviation.
    - 2 for spectral standard deviation.
    This calculates up to the 4th order
    
    Returns:
    --------
    np.array
        A 1D numpy array containing the calculated spectral bandwidth value.
    
    References:
    -----------
        - McFee, B., Matt McVicar, Daniel Faronbi, Iran Roman, Matan Gover, Stefan Balke, Scott Seyfarth, Ayoub Malek, 
        Colin Raffel, Vincent Lostanlen, Benjamin van Niekirk, Dana Lee, Frank Cwitkowitz, Frank Zalkow, Oriol Nieto, 
        Dan Ellis, Jack Mason, Kyungyun Lee, Bea Steers, … Waldir Pimenta. (2024). librosa/librosa: 0.10.2.post1 (0.10.2.post1). 
        Zenodo. https://doi.org/10.5281/zenodo.11192913
        - Giannakopoulos, T., & Pikrakis, A. (2014). Audio Features. Introduction to Audio Analysis, 
        59–103. https://doi.org/10.1016/B978-0-08-099388-1.00004-2
    """
    normalized_magnitudes = magnitudes / np.sum(magnitudes)
    mean_frequency = calculate_spectral_centroid(freqs, magnitudes)
    if isinstance(bandwidth_orders, int):
        return ((np.sum(((freqs - mean_frequency) ** bandwidth_orders) * normalized_magnitudes)) ** (1 / bandwidth_orders))
    else:
        spectral_bandwidth = []
        for order in bandwidth_orders:
            spectral_bandwidth.append(((np.sum(((freqs - mean_frequency) ** order) * normalized_magnitudes)) ** (1 / order)))
        return np.array(spectral_bandwidth)

#TODO: Check what is going on here
@name("spectral_absolute_deviation_order_{}", "abs_dev_orders")
def calculate_spectral_absolute_deviation(freqs, magnitudes, abs_dev_orders=1, **kwargs):
    """
    Calculate the spectral absolute deviation of a given frequency spectrum.

    Spectral absolute deviation measures the average deviation of frequency components 
    from the spectral centroid (mean frequency) weighted by their magnitudes. This 
    function generalizes the concept for any given order, with the even order 
    spectral absolute deviation being equivalent to the spectral bandwidth of the 
    same order.

    Parameters:
    -----------
    freqs : np.array
        An array of frequencies corresponding to the magnitude spectrum bins.
    magnitudes : np.array
        An array of magnitude values of the spectrum at the corresponding frequencies.
    order : int, optional default=1)
        The order of the deviation calculation. When `order=2`, the result is equivalent 
        to the spectral bandwidth (standard deviation) of the spectrum.

    Returns:
    --------
    np.array
        A 1D numpy array containing the calculated spectral absolute deviation.
        
    References:
    -----------
        - McFee, B., Matt McVicar, Daniel Faronbi, Iran Roman, Matan Gover, Stefan Balke, Scott Seyfarth, Ayoub Malek, 
        Colin Raffel, Vincent Lostanlen, Benjamin van Niekirk, Dana Lee, Frank Cwitkowitz, Frank Zalkow, Oriol Nieto, 
        Dan Ellis, Jack Mason, Kyungyun Lee, Bea Steers, … Waldir Pimenta. (2024). librosa/librosa: 0.10.2.post1 (0.10.2.post1). 
        Zenodo. https://doi.org/10.5281/zenodo.11192913
    """
    # The even order spectral absolute deviation is the same as spectral bandwidth of the same order
    normalized_magnitudes = magnitudes / np.sum(magnitudes)
    mean_frequency = calculate_spectral_centroid(freqs, magnitudes)
    if isinstance(abs_dev_orders, int):
        return ((np.sum((np.abs(freqs - mean_frequency) ** abs_dev_orders) * normalized_magnitudes)) ** (1 / abs_dev_order))
    else:
        spectral_absolute_deviation = []
        for order in abs_dev_orders:
            spectral_absolute_deviation.append(((np.sum((np.abs(freqs - mean_frequency) ** order) * normalized_magnitudes)) ** (1 / order)))
        return np.array(spectral_absolute_deviation)

@name("spectral_covariance")
def calculate_spectral_cov(freqs, magnitudes, **kwargs):
    """
    Calculate the spectral coefficient of variation (CoV) for a given frequency spectrum.

    The spectral CoV provides a normalized measure of the dispersion of frequencies 
    around the spectral centroid (mean frequency). It is calculated as the ratio of 
    the spectral standard deviation (second-order spectral bandwidth) to the spectral 
    centroid, multiplied by 100 to express it as a percentage.

    Parameters:
    -----------
    freqs : array-like
        The frequency values of the spectrum. This is a 1D array representing the 
        frequency bins of the spectrum.

    magnitudes : array-like
        The magnitude values of the spectrum corresponding to each frequency bin. 
        This is a 1D array representing the magnitude of the signal at each frequency.

    Returns:
    --------
    float
        The spectral coefficient of variation (CoV) expressed as a percentage.

    References:
    -----------
        - Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for 
        neural digital signal processing. Journal of Open Source Software, 4(36), 1272. 
        https://doi.org/10.21105/JOSS.01272
    """
    mean_frequency = calculate_spectral_centroid(freqs, magnitudes)
    frequency_std = calculate_spectral_bandwidth(freqs, magnitudes, 2)
    coefficient_of_variation = (frequency_std / mean_frequency) * 100
    return coefficient_of_variation

@name("spectral_flux_order_{}", "flux_orders")
def calculate_spectral_flux(magnitudes, flux_orders=2, **kwargs):
    """
    Calculates the flux of the spectrum.

    Spectral flux is a measure of the variability of the spectrum over time.

    Parameters:
    -----------
    magnitudes : array-like
        The magnitude values of the spectrum corresponding to each frequency bin. 
        This is a 1D array representing the magnitude of the signal at each frequency.

    Returns:
    --------
    float
        The spectral flux of the spectrum

    Reference:
    ----------
    - Wang, W., Yu, X., Wang, Y. H., & Swaminathan, R. (2012). Audio fingerprint based on spectral flux 
    for audio retrieval. ICALIP 2012 - 2012 International Conference on Audio, Language and Image Processing, 
    Proceedings, 1104–1107. https://doi.org/10.1109/ICALIP.2012.6376781
    """
    if isinstance(flux_orders, int):
        spectral_flux = (np.sum(np.abs(np.diff(magnitudes)) ** flux_orders)) ** (1 / flux_orders)
    else:
        spectral_flux = []
        for order in flux_orders:
            spectral_flux.append((np.sum(np.abs(np.diff(magnitudes)) ** order)) ** (1 / order))
        return np.array(spectral_flux)

@name("spectral_rolloff")
def calculate_spectral_rolloff(freqs, magnitudes, roll_percent=0.85, **kwargs):
    """
    Calculate the spectral rolloff point of a signal.

    The spectral rolloff is the frequency below which a specified percentage 
    (default is 85%) of the total spectral energy is contained.

    Parameters:
    -----------
    freqs : np.array
        Array of frequencies corresponding to the frequency components of the signal.
    magnitudes : np.array
        Array of magnitudes (or power) corresponding to the frequencies.
    roll_percent : float, optional
        The percentage of total spectral energy below the rolloff point. Default is 0.85 (85%).

    Returns:
    --------
    np.array
        The frequency at which the spectral rolloff occurs.

    References:
    -----------
        - Giannakopoulos, T., & Pikrakis, A. (2014). Audio Features. Introduction to Audio Analysis, 
        59–103. https://doi.org/10.1016/B978-0-08-099388-1.00004-2
    """
    cumulative_magnitudes = np.cumsum(magnitudes)
    rolloff_frequency = np.min(freqs[np.where(cumulative_magnitudes >= roll_percent * cumulative_magnitudes[-1])])
    return rolloff_frequency

@name("harmonic_ratio")
def calculate_harmonic_ratio(signal, **kwargs):
    """
    Calculate the harmonic ratio of a given signal.

    The harmonic ratio is a measure of the amount of harmonic content in the signal 
    compared to the total energy of the signal. It is typically used in audio analysis 
    to quantify how much of the signal consists of harmonic components.

    Parameters:
    -----------
    signal : np.array
        The input audio signal as a 1D numpy array.

    Returns:
    --------
    np.array
        The harmonic ratio of the signal as a 1D numpy array.

    References:
    -----------
        This function is based on the harmonic ratio calculation methodology as described in:
            https://www.mathworks.com/help/audio/ref/harmonicratio.html
    """
    harmonic_component = librosa.effects.harmonic(signal)
    
    # Calculate mean absolute amplitudes
    mean_abs_harmonic = np.mean(np.abs(harmonic_component))
    mean_abs_signal = np.mean(np.abs(signal))
    
    # Avoid division by zero
    if mean_abs_signal == 0:
        return np.array([0.0])
    harmonic_ratio = mean_abs_harmonic / mean_abs_signal
    return harmonic_ratio

@name("fundamental_frequency")
def calculate_fundamental_frequency(signal, **kwargs):
    """
    Calculate the fundamental frequency (F0) of a given audio signal.

    The fundamental frequency, often referred to as F0, is the lowest frequency 
    of a periodic waveform, corresponding to the perceived pitch of the sound.
    This function uses the YIN algorithm to estimate the fundamental frequency.

    Parameters:
    -----------
    signal : np.array
        The input audio signal as a 1D numpy array.

    Returns:
    --------
    np.array
        The average fundamental frequency (F0) of the signal as a 1D numpy array.
        
    Reference:
    ----------
        - Hee Lee, J., & Humes, L. E. (2012). Effect of fundamental-frequency and sentence-onset 
        differences on speech-identification performance of young and older adults in a competing-talker background. 
        The Journal of the Acoustical Society of America, 132(3), 1700–1717. https://doi.org/10.1121/1.4740482
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, 
        H. (2020). TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    return np.mean(f0)

@name("spectral_crest_factor")
def calculate_spectral_crest_factor(magnitudes, **kwargs):
    """
    Calculate the spectral crest factor of a signal's magnitude spectrum.

    The spectral crest factor is a measure of how peaky the spectrum is. It is defined as the ratio 
    of the maximum value of the magnitude spectrum to the mean value of the magnitude spectrum.

    Parameters:
    -----------
    magnitudes : np.array
        Array of magnitudes (or power) corresponding to the frequencies in the signal's spectrum.

    Returns:
    --------
    np.array
        The spectral crest factor of the signal as a 1D numpy array.

    References:
    -----------
        The spectral crest factor computation follows the description from:
            https://www.mathworks.com/help/signal/ref/spectralcrest.html#d126e220002
    """
    crest_factor = np.max(magnitudes) / np.mean(magnitudes)
    return crest_factor

@name("spectral_decrease")
def calculate_spectral_decrease(freqs, magnitudes, **kwargs):
    """
    Calculate the spectral decrease of a signal.

    Spectral decrease is a measure of the amount of energy reduction or attenuation 
    in the spectrum as the frequency increases. It is calculated as the weighted sum 
    of the differences between the spectral magnitudes and the first magnitude value, 
    normalized by the bin index.

    Parameters:
    -----------
    freqs : np.array
        Array of frequencies corresponding to the frequency components of the signal.
    magnitudes : np.array
        Array of magnitudes corresponding to the frequencies in the signal's spectrum.

    Returns:
    --------
    np.array
        The spectral decrease value of the signal as a 1D numpy array.

    References:
    -----------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, 
        H. (2020). TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    k = np.arange(1, len(magnitudes) + 1)
    spectral_decrease = np.sum((magnitudes[1:] - magnitudes[0]) / k[1:])
    return spectral_decrease

@name("spectral_irregularity")
def calculate_spectral_irregularity(magnitudes, **kwargs):
    """
    Calculate the spectral irregularity of a signal's magnitude spectrum.

    Spectral irregularity measures the degree of fluctuation between consecutive 
    magnitudes in the spectrum, indicating how smooth or rough the spectrum is.

    Parameters:
    -----------
    magnitudes : np.array
        Array of magnitudes corresponding to the frequencies in the signal's spectrum.

    Returns:
    --------
    float
        The spectral irregularity value.
        
    Reference:
    ----------
        - Spectral features (spectralFeaturesProc.m) — The Two!Ears Auditory Model <unknown> documentation. (n.d.). 
        Retrieved September 17, 2024, from https://docs.twoears.eu/en/latest/afe/available-processors/spectral-features/
    """
    irregularity = np.sum(np.abs(magnitudes[1:] - magnitudes[:-1])) / (len(magnitudes) - 1)
    return irregularity

@name("spectral_winsorized_mean")
def calculate_spectral_winsorized_mean(freqs, magnitudes, limits=(0.05, 0.95), **kwargs):
    """
    Calculate the winsorized mean of frequencies, trimming the magnitude values at the specified limits.
    
    Parameters:
    -----------
    freqs : np.array
        Array of frequency values corresponding to the magnitudes.
    magnitudes : np.array
        Array of magnitude values corresponding to the frequencies.
    limits : tuple, optional
        A tuple specifying the lower and upper percentage of magnitudes to be trimmed 
        (default is (0.05, 0.95), meaning the lowest and highest 5% are excluded).
    
    Returns:
    --------
    np.array
        A numpy array containing the winsorized mean of the trimmed frequency values.    
        
    Reference:
    ----------
        - Onoz, B., & Oguz, B. (2003). Assessment of Outliers in Statistical Data Analysis. Integrated Technologies for 
        Environmental Monitoring and Information Production, 173–180. https://doi.org/10.1007/978-94-010-0231-8_13
    """
    # Ensure magnitudes and freqs are numpy arrays
    freqs = np.asarray(freqs)
    magnitudes = np.asarray(magnitudes)

    sorted_indices = np.argsort(magnitudes)
    
    # Calculate the lower and upper limits for trimming
    lower_limit = int(limits[0] * len(magnitudes))
    upper_limit = int(limits[1] * len(magnitudes))
    # Select the trimmed indices
    trimmed_indices = sorted_indices[lower_limit:upper_limit]
    winsorized_mean = np.mean(freqs[trimmed_indices])
    
    return winsorized_mean

@name("total_harmonic_distortion")
def calculate_total_harmonic_distortion(signal, fs, harmonics=5, **kwargs):
    """
    Calculate the Total Harmonic Distortion (THD) of a signal.
    
    THD is a measure of the distortion in a signal caused by the presence of 
    harmonics (integer multiples of the fundamental frequency). It is calculated 
    as the square root of the ratio of harmonic power to the fundamental power.
    
    Parameters:
    -----------
    signal : array-like
        The input signal for which the THD is being calculated.
        
    fs : int or float
        The sampling frequency of the input signal, in Hz.
        
    harmonics : int, optional
        The number of harmonic frequencies to consider for THD calculation.
        Default is 5.
    
    Returns:
    --------
    thd : float
    The total harmonic distortion of the input signal, as a ratio of harmonic 
    power to the fundamental power.
    References
    ----------
        - Blagouchine, I. v., & Moreau, E. (2011). Analytic method for the computation 
        of the total harmonic distortion by the cauchy method of residues. IEEE Transactions 
        on Communications, 59(9), 2478–2491. https://doi.org/10.1109/TCOMM.2011.061511.100749
        - McFee, B., Matt McVicar, Daniel Faronbi, Iran Roman, Matan Gover, Stefan Balke, Scott Seyfarth, Ayoub Malek, 
        Colin Raffel, Vincent Lostanlen, Benjamin van Niekirk, Dana Lee, Frank Cwitkowitz, Frank Zalkow, Oriol Nieto, 
        Dan Ellis, Jack Mason, Kyungyun Lee, Bea Steers, … Waldir Pimenta. (2024). librosa/librosa: 0.10.2.post1 (0.10.2.post1). 
        Zenodo. https://doi.org/10.5281/zenodo.11192913
    """
    f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    fundamental_freq = np.mean(f0)
    fft_spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fundamental_power = np.abs(fft_spectrum[fundamental_idx])**2 # power of the fundamental frequency
    
    # Harmonic power (sum of harmonics' power starting from the 2nd harmonic)
    harmonic_power = 0
    for i in range(2, harmonics+1):  # Start from the second harmonic 
        harmonic_idx = np.argmin(np.abs(freqs - i * fundamental_freq))
        harmonic_power += np.abs(fft_spectrum[harmonic_idx])**2
    thd = np.sqrt(harmonic_power / fundamental_power)
    return thd

@name("inharmonicity")
def calculate_inharmonicity(signal, freqs, magnitudes, **kwargs):
    """
    Calculate inharmonicity coefficient from audio signal. Inharmonicity is the measure of 
    how partial tones deviate from harmonics.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input audio signal
    freqs : np.ndarray
        Array of frequency values corresponding to the magnitudes.
    magnitudes : np.ndarray
        Array of magnitude values corresponding to the frequencies.
        
    Returns:
    --------
    np.ndarray
        Returns np.nan if calculation fails
        
    References:
    -----------
        - Rigaud, F., David, B., & Daudet, L. (2013). A parametric model and estimation techniques 
        for the inharmonicity and tuning of the piano. The Journal of the Acoustical Society of America, 
        133(5), 3107–3118. https://doi.org/10.1121/1.4799806
    """
    try:
        # First try pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'), frame_length=2048)
        
        # If pYIN fails, fallback to YIN
        if np.all(np.isnan(f0)):
            f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'),fmax=librosa.note_to_hz('C8'))
        
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return np.nan
            
        fundamental_freq = np.median(f0_clean)
        
        peak_indices, _ = find_peaks(magnitudes, height=np.max(magnitudes)*0.1)
        harmonic_freqs = freqs[peak_indices]
        harmonics = [f for f in harmonic_freqs if np.isclose(f % fundamental_freq, 0, atol=1)]
        
        if len(harmonics) == 0:
            return np.nan
            

        inharmonicity_sum = 0
        for i, harmonic in enumerate(harmonics):
            ideal_harmonic = (i+1) * fundamental_freq
            deviation = np.abs(harmonic - ideal_harmonic) / ideal_harmonic
            inharmonicity_sum += deviation
        
        inharmonicity = inharmonicity_sum / len(harmonics) if harmonics else np.nan
        
    except Exception as e:
        inharmonicity = np.nan
    return inharmonicity

@name(["tristimulus_T1", "tristimulus_T2", "tristimulus_T3"])
def calculate_tristimulus(magnitudes, **kwargs):
    """
    Calculate the tristimulus values from the magnitudes of a frequency spectrum.

    Tristimulus values are used in timbral analysis of audio signals, providing a measure of how 
    energy is distributed across different parts of the frequency spectrum. These are calculated as:
    - T1: The proportion of energy in the first harmonic.
    - T2: The proportion of energy in the second harmonic.
    - T3: The proportion of energy in the remaining harmonics.

    Parameters:
    ----------
    magnitudes : np.ndarray
        Array of magnitude values corresponding to the frequencies.

    Returns:
    -------
    np.array
        A numpy array containing the three tristimulus values [T1, T2, T3].
        If the input `magnitudes` array has fewer than 3 elements, the function returns [np.nan, np.nan, np.nan].
        
    References:
    -----------
        - Du, X., Teng, G., Wang, C., Carpentier, L., & Norton, T. (2021). A tristimulus-formant model for automatic 
        recognition of call types of laying hens. Computers and Electronics in Agriculture, 187, 106221. 
        https://doi.org/10.1016/J.COMPAG.2021.106221
    """
    if len(magnitudes) < 3:
        return np.array([np.nan, np.nan, np.nan])
    t1 = magnitudes[0] / np.sum(magnitudes)
    t2 = magnitudes[1] / np.sum(magnitudes)
    t3 = np.sum(magnitudes[2:]) / np.sum(magnitudes)
    return np.array([t1, t2, t3])

@name("spectral_rollon")
def calculate_spectral_rollon(freqs, magnitudes, roll_percent=0.05, **kwargs):
    """
    Calculate the spectral roll-on point of a signal.
    
    The spectral roll-on is the frequency below which a specified percentage 
    (e.g., 5%) of the total spectral magnitude is contained. It is used to 
    determine the low-frequency cutoff for a given signal's spectrum.
    
    Parameters:
    -----------
    freqs : array-like
        Array of frequencies corresponding to the magnitude spectrum of the signal.
        
    magnitudes : array-like
        Array of magnitudes corresponding to the frequency spectrum of the signal.
        
    roll_percent : float, optional
        The percentage of the total spectral energy below which the roll-on 
        frequency is calculated. The default is 0.05 (5%).
    
    Returns:
    --------
    rollon_frequency : float
        The frequency at which the cumulative magnitude reaches the specified 
        percentage of the total energy (the roll-on point).
    
    Reference:
    ----------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, 
        H. (2020). TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    cumulative_magnitudes = np.cumsum(magnitudes)
    rollon_frequency = np.min(freqs[np.where(cumulative_magnitudes >= roll_percent * cumulative_magnitudes[-1])])
    return rollon_frequency

@name("spectral_hole_count")
def calculate_spectral_hole_count(magnitudes, threshold=0.05, **kwargs):
    # https://doi.org/10.1103/PhysRevA.104.063111
    peaks, _ = find_peaks(magnitudes, height=threshold)
    dips, _ = find_peaks(-magnitudes, height=-threshold)
    return len(dips)


@name("spectral_variability")
def calculate_spectral_variability(magnitudes, **kwargs):
    # https://doi.org/10.1016/j.dsp.2015.10.011
    variability = np.var(magnitudes)
    return variability

@name("spectral_spread_ratio")
def calculate_spectral_spread_ratio(freqs, magnitudes, reference_value=1.0, **kwargs):
    """
    Calculate the spectral spread ratio by normalizing the spectral bandwidth (spread) by a reference value.

    Parameters:
    -----------
    freqs : array-like
        Array of frequencies corresponding to the magnitude spectrum of the signal.
        
    magnitudes : array-like
        Array of magnitudes corresponding to the frequency spectrum of the signal.
        
    reference_value: float, optional (default=1.0)
        The reference value to normalize the spectral spread (bandwidth).
        
    Return:
    ------
    spread_ratio: np.ndarray
        Normalized spectral spread ratio.
    """
    # https://doi.org/10.1016/j.softx.2020.100456
    spread = calculate_spectral_bandwidth(freqs, magnitudes, bandwidth_orders = 2)
    spread_ratio = spread / reference_value
    return spread_ratio

@name("spectral_skewness_ratio")
def calculate_spectral_skewness_ratio(freqs, magnitudes, reference_value=1.0, **kwargs):
    """
    Calculate the spectral skewness ratio by normalizing the spectral bandwidth (spread) by a reference value.

    Parameters:
    -----------
    freqs : array-like
        Array of frequencies corresponding to the magnitude spectrum of the signal.
        
    magnitudes : array-like
        Array of magnitudes corresponding to the frequency spectrum of the signal.
        
    reference_value: float, optional (default=1.0)
        The reference value to normalize the spectral skewness.
        
    Return:
    ------
    skewness_ratio: np.ndarray
        Normalized spectral skewness ratio.
    """ 
    # https://doi.org/10.1016/j.softx.2020.100456
    skewness = calculate_spectral_skewness(freqs, magnitudes)
    skewness_ratio = skewness / reference_value
    return skewness_ratio

@name("spectral_kurtosis_ratio")
def calculate_spectral_kurtosis_ratio(freqs, magnitudes, reference_value=1.0, **kwargs):
    """
    Calculate the spectral kurtosis ratio by normalizing the spectral bandwidth (spread) by a reference value.

    Parameters:
    -----------
    freqs : array-like
        Array of frequencies corresponding to the magnitude spectrum of the signal.
        
    magnitudes : array-like
        Array of magnitudes corresponding to the frequency spectrum of the signal.
        
    reference_value: float, optional (default=1.0)
        The reference value to normalize the spectral kurtosis.
        
    Return:
    ------
    kurtosis_ratio: np.ndarray
        Normalized spectral kurtosis ratio.
    """
    # https://doi.org/10.1016/j.softx.2020.100456
    kurtosis = calculate_spectral_kurtosis(freqs, magnitudes)
    kurtosis_ratio = kurtosis / reference_value
    return kurtosis_ratio

@name("spectral_tonal_power_ratio")
def calculate_spectral_tonal_power_ratio(signal, **kwargs):
    """
    Calculate the Spectral Tonal Power Ratio (STPR) of a signal.

    The Spectral Tonal Power Ratio is a measure of the proportion of harmonic (or tonal) content 
    in a signal relative to the total power. It provides insight into the harmonic structure of the signal, 
    which is useful in applications such as audio analysis, music processing, and signal classification.

    Parameters:
    ----------
    signal : array-like
        The input audio signal in the time domain. This should be a one-dimensional array representing
        the signal waveform.

    Returns:
    -------
    np.array
        A NumPy array containing the Spectral Tonal Power Ratio (STPR) as a single element.
        If the total power of the signal is zero (i.e., an all-zero or silent signal), the function returns [0.0].

    Reference:
    ----------
        - McFee, B., Matt McVicar, Daniel Faronbi, Iran Roman, Matan Gover, Stefan Balke, Scott Seyfarth, Ayoub Malek, 
        Colin Raffel, Vincent Lostanlen, Benjamin van Niekirk, Dana Lee, Frank Cwitkowitz, Frank Zalkow, Oriol Nieto, 
        Dan Ellis, Jack Mason, Kyungyun Lee, Bea Steers, … Waldir Pimenta. (2024). librosa/librosa: 0.10.2.post1 (0.10.2.post1). 
        Zenodo. https://doi.org/10.5281/zenodo.11192913
    """
    total_power = np.sum(signal**2)
    if total_power == 0:
        return np.array([0.0])
        
    harmonic_power = np.sum(librosa.effects.harmonic(signal)**2) 
    tonal_power_ratio = harmonic_power / total_power
    return tonal_power_ratio

@name("spectral_harmonics_to_noise_ratio")
def calculate_spectral_harmonics_to_noise_ratio(signal, **kwargs):
    """
    Calculate the ratio of harmonic energy to noise energy in an audio signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input audio signal. Should be a 1D numpy array.
    
    Returns
    -------
    np.ndarray
        Array containing the harmonics-to-noise ratio(s).
        Returns a single value for the entire signal by default
        
    Notes
    -----
    The Harmonics-to-Noise Ratio (HNR) is calculated as:
    HNR = 10 * log10(harmonic_energy / noise_energy)
    
    A higher HNR indicates a more harmonic signal (e.g., clean speech, musical tones)
    while a lower HNR indicates more noise content.
    
    References
    ----------
        - Boersma, P. (n.d.). Accurate Short-Term Analysis Of The Fundamental Frequency And The 
        Harmonics-To-Noise Ratio Of A Sampled Sound. https://www.researchgate.net/publication/2326829
    """    
    harmonic_part = librosa.effects.harmonic(signal)
    noise_part = signal - harmonic_part
        
    harmonic_energy = np.sum(harmonic_part**2)
    noise_energy = np.sum(noise_part**2)
        
    if noise_energy == 0:
        return np.array([100.0])

    hnr_db = 10 * np.log10(harmonic_energy / noise_energy)
        
    return hnr_db

@name("spectral_noise_to_harmonics_ratio")
def calculate_spectral_noise_to_harmonics_ratio(signal, **kwargs):
    """
    Calculate the ratio of noise energy to harmonic energy in an audio signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input audio signal
        
    Returns
    -------
    signal : np.ndarray
        Single-element array containing the noise-to-harmonics ratio
        
    Reference
    ----------
        - Pereira Jotz, G., Cervantes, O., Abrahão, M., Parente Settanni, F. A., & Carrara de Angelis, E. C. (2002). 
        Noise-to-harmonics ratio as an acoustic measure of voice disorders in boys. Journal of Voice : Official 
        Journal of the Voice Foundation, 16(1), 28–31. https://doi.org/10.1016/S0892-1997(02)00068-1
    """        
    harmonic_part = librosa.effects.harmonic(signal)
    noise_part = signal - harmonic_part
    
    harmonic_energy = np.sum(harmonic_part**2)
    if harmonic_energy == 0:
        return np.array([float('inf')])  
        
    noise_energy = np.sum(noise_part**2)
    nhr_db = 10 * np.log10(noise_energy / harmonic_energy)

    return nhr_db

@name("spectral_even_to_odd_harmonic_energy_ratio")
def calculate_spectral_even_to_odd_harmonic_energy_ratio(signal, fs, **kwargs):
    """
    Calculate the ratio of spectral energy between even and odd harmonics in the signal.

    The even-to-odd harmonic energy ratio measures how the energy is distributed between 
    even and odd harmonics in a signal, which is important in the timbral analysis of audio.
    A high ratio indicates more energy in the even harmonics, while a low ratio indicates more 
    energy in the odd harmonics.

    Parameters:
    ----------
    signal : array-like
        The input audio signal in the time domain. Should be a one-dimensional array.
    
    fs : int
        The sampling frequency of the signal, in Hz.

    Returns:
    -------
    np.array
        A NumPy array containing the ratio of the energy in the even harmonics to the energy 
        in the odd harmonics. If either energy is zero, the function returns np.inf or 0 accordingly.

    Reference:
    ----------
    Zenodo citation: https://zenodo.org/badge/latestdoi/6309729
    """
    # Estimate the fundamental frequency (f0)
    f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    fundamental_freq = np.median(f0)

    # Generate lists of even and odd harmonics
    even_harmonics = [(2 * i + 2) * fundamental_freq for i in range(int(fs / (2 * fundamental_freq)))]
    odd_harmonics = [(2 * i + 1) * fundamental_freq for i in range(int(fs / (2 * fundamental_freq)))]

    # Calculate the energy of even and odd harmonics
    even_energy = sum([np.sum(np.abs(np.fft.rfft(signal * np.sin(2 * np.pi * harmonic * np.arange(len(signal)) / fs)))) for harmonic in even_harmonics])
    odd_energy = sum([np.sum(np.abs(np.fft.rfft(signal * np.sin(2 * np.pi * harmonic * np.arange(len(signal)) / fs)))) for harmonic in odd_harmonics])
    even_to_odd_ratio = 10 * np.log10(even_energy / odd_energy) if odd_energy != 0 else np.inf  # Avoid division by zero

    return even_to_odd_ratio


@name("spectral_strongest_frequency_phase")
def calculate_spectral_strongest_frequency_phase(spectrum, **kwargs):
    """
    Calculates the phase of the strongest frequency component in a given spectrum.

    This function identifies the frequency with the highest magnitude in the provided
    frequency spectrum and returns the phase angle associated with that frequency.

    Parameters
    ----------
    spectrum : array_like
        A complex-valued array representing the frequency spectrum, where each element
        contains the magnitude and phase of the corresponding frequency component.
    
    Returns
    -------
    np.array
        A single-element array containing the phase (in radians) of the strongest frequency component.
    
    Reference:
    ----------
        - https://mriquestions.com/phase-v-frequency.html
    """
    strongest_frequency_index = np.argmax(np.abs(spectrum))
    phase = np.angle(spectrum[strongest_frequency_index])
    return phase

@name("spectral_frequency_below_peak")
def calculate_spectral_frequency_below_peak(freqs, magnitudes, **kwargs):
    """
    Calculates the frequency just below the peak frequency in a spectrum.

    This function identifies the frequency component with the highest magnitude
    and returns the frequency immediately below that peak.

    Parameters
    ----------
    freqs : array_like
        A 1D array representing the frequency components of the spectrum.
    magnitudes : array_like
        A 1D array representing the magnitudes corresponding to each frequency component.
    
    Returns
    -------
    np.array
        A single-element array containing the frequency just below the peak frequency.
        If the peak is at the first frequency, returns the first frequency itself.
    
    Reference:
    ----------
        - Sörnmo, L., & Laguna, P. (2005). EEG Signal Processing. Bioelectrical Signal Processing in Cardiac 
        and Neurological Applications, 55–179. https://doi.org/10.1016/B978-012437552-9/50003-9    
    """
    peak_index = np.argmax(magnitudes)
    frequency_below_peak = freqs[max(0, peak_index - 1)]
    return frequency_below_peak

@name("spectral_frequency_above_peak")
def calculate_spectral_frequency_above_peak(freqs, magnitudes, **kwargs):
    """
    Calculates the frequency just above the peak frequency in a spectrum.

    This function identifies the frequency component with the highest magnitude
    and returns the frequency immediately above that peak.

    Parameters
    ----------
    freqs : array_like
        A 1D array representing the frequency components of the spectrum.
    magnitudes : array_like
        A 1D array representing the magnitudes corresponding to each frequency component.
    
    Returns
    -------
    np.array
        A single-element array containing the frequency just above the peak frequency.
        If the peak is at the last frequency, returns the last frequency itself.
    
    Reference:
    ----------
        - Sörnmo, L., & Laguna, P. (2005). EEG Signal Processing. Bioelectrical Signal Processing in Cardiac 
        and Neurological Applications, 55–179. https://doi.org/10.1016/B978-012437552-9/50003-9    
    """
    peak_index = np.argmax(magnitudes)
    frequency_above_peak = freqs[min(len(freqs) - 1, peak_index + 1)]
    return frequency_above_peak

@name("spectral_cumulative_frequency_above_threshold_{}", "thresholds_freq_above")
def calculate_spectral_cumulative_frequency_above(freqs, magnitudes, thresholds_freq_above, **kwargs):
    """
    Calculate the frequency at which the cumulative power reaches or exceeds a given threshold.

    This function computes the cumulative sum of magnitudes (spectral power) normalized 
    to the total power. It identifies the first frequency where the cumulative power reaches 
    or surpasses the specified threshold.

    Parameters:
    -----------
    freqs : np.ndarray
        Array of frequencies corresponding to the spectral data.
    magnitudes : np.ndarray
        Array of magnitudes (power) at each frequency.
    thresholds_freq_above : float
        A value between 0 and 1 indicating the cumulative power threshold (e.g., 0.5 for 50%).

    Returns:
    --------
    np.ndarray
        An array containing the frequency at which the cumulative power meets or exceeds the threshold.
    
    Reference:
    ---------
        - Lee, S.-C., & Peters, R. D. (2009). A New Look at an Old Tool-the Cumulative Spectral Power 
        of Fast-Fourier Transform Analysis. https://arxiv.org/abs/0901.3708v1
    """
    cumulative_power = np.cumsum(magnitudes) / np.sum(magnitudes)
    if isinstance(thresholds_freq_above, float):
        return freqs[np.where(cumulative_power >= thresholds_freq_above)[0][0]]
    else:  
        frequency = []
        for threshold in thresholds_freq_above:
            frequency.append(freqs[np.where(cumulative_power >= threshold)[0][0]])
        return np.array(frequency)

@name("spectral_cumulative_frequency_below_threshold_{}", "thresholds_freq_below")
def calculate_spectral_cumulative_frequency_below(freqs, magnitudes, thresholds_freq_below, **kwargs):
    """
    Calculate the frequency below which the cumulative power stays within a given threshold.

    This function computes the cumulative sum of magnitudes (spectral power) normalized 
    to the total power. It identifies the last frequency where the cumulative power remains 
    below or equal to the specified threshold.

    Parameters:
    -----------
    freqs : np.ndarray
        Array of frequencies corresponding to the spectral data.
    magnitudes : np.ndarray
        Array of magnitudes (power) at each frequency.
    thresholds_freq_below : float
        A value between 0 and 1 indicating the cumulative power threshold (e.g., 0.8 for 80%).

    Returns:
    --------
    np.ndarray
        An array containing the frequency at which the cumulative power stays below or equals the threshold.

    Reference:
    ---------
        - Lee, S.-C., & Peters, R. D. (2009). A New Look at an Old Tool-the Cumulative Spectral Power 
        of Fast-Fourier Transform Analysis. https://arxiv.org/abs/0901.3708v1
    """
    cumulative_power = np.cumsum(magnitudes) / np.sum(magnitudes)
    if isinstance(thresholds_freq_below, float):
        return freqs[np.where(cumulative_power <= thresholds_freq_below)[-1][-1]]
    else:
        frequency = []
        for threshold in thresholds_freq_below:
            frequency.append(freqs[np.where(cumulative_power <= threshold)[-1][-1]])
        return np.array(frequency)

@name("spectral_change_vector_magnitude")
def calculate_spectral_change_vector_magnitude(magnitudes, **kwargs):
    """
    Calculate the magnitude of the spectral change vector 
    based on consecutive differences in magnitudes.

    Parameters
    ----------
    magnitudes : array_like
        A 1D array representing the magnitudes corresponding to each frequency component.
    
    Returns
    -------
    np.array
        An array containing the spectral change vector magnitude.
        
    Reference:
    ----------
        - Carvalho Júnior, O. A., Guimarães, R. F., Gillespie, A. R., Silva, N. C., & Gomes, R. A. T. (2011). 
        A New Approach to Change Vector Analysis Using Distance and Similarity Measures. Remote Sensing 2011, 
        Vol. 3, Pages 2473-2493, 3(11), 2473–2493. https://doi.org/10.3390/RS3112473
    """
    change_vector_magnitude = np.linalg.norm(np.diff(magnitudes))
    return change_vector_magnitude

@name("spectral_low_frequency_content")
def calculate_spectral_low_frequency_content(freqs, magnitudes, psd, method="fixed", low_freq_threshold=300, percentile=50, **kwargs):
    """
    Calculate the low-frequency content in the spectral data using a dynamic or fixed threshold.
    
    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequencies corresponding to the spectral data.
    magnitudes : np.ndarray
        Array of magnitudes (power) at each frequency.
    psd: np.array
            An array of Power Spectral Density (PSD) values corresponding to the frequencies.
    method: string
        Method to determine the threshold ('fixed', 'mean', 'median', 'percentile').
    threshold: float 
        Fixed frequency threshold for 'fixed' method (default is 300 Hz).
    percentile: float
        Percentile value for 'percentile' method (default is 50, which is the median).
    
    Returns:
    --------
    low_freq_content: float
        Sum of the magnitudes for frequencies below the dynamically chosen threshold.
    """    
    # https://resources.pcb.cadence.com/blog/2022-an-overview-of-frequency-bands-and-their-applications
    # Determine the threshold based on the selected method
    if method == 'fixed':
        dynamic_threshold = low_freq_threshold
    elif method == 'mean':
        dynamic_threshold = np.sum(freqs * magnitudes) / np.sum(magnitudes)  # Mean frequency
    elif method == 'median':
        cumulative_psd = np.cumsum(psd)
        dynamic_threshold = freqs[np.searchsorted(cumulative_psd, cumulative_psd[-1] / 2)]  # Median frequency
    elif method == 'percentile':
        dynamic_threshold = np.percentile(freqs, percentile)  # Percentile-based threshold
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'fixed', 'mean', 'median', or 'percentile'.")

    low_freq_content = np.sum(magnitudes[freqs < dynamic_threshold])
    
    return low_freq_content

@name("spectral_mid_frequency_content")
def calculate_spectral_mid_frequency_content(freqs, magnitudes, psd, method="fixed", mid_freq_range=(300, 3000), percentile=50, **kwargs):
    """
    Calculate the mid-frequency content in the spectral data by summing the magnitudes 
    in the specified or dynamically determined mid-frequency range.
    
    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequencies corresponding to the spectral data.
    magnitudes : np.ndarray
        Array of magnitudes (power) at each frequency.
    psd: np.array
            An array of Power Spectral Density (PSD) values corresponding to the frequencies.
    method: string
        Method to determine the threshold ('fixed', 'mean', 'median', 'percentile').
    mid_freq_range: tuple
        A tuple specifying the lower and upper bound of the mid-frequency range (default is 300 Hz to 3000 Hz).
    percentile: float
        Percentile value for 'percentile' method (default is 50, which is the median).
    
    Returns:
    --------
    mid_freq_content: float
        Sum of the magnitudes for frequencies within the dynamically or fixed mid-frequency range.
    """    
    # Determine the mid-frequency range based on the selected method
    if method == 'fixed':
        dynamic_mid_freq_range = mid_freq_range
    elif method == 'mean':
        mean_freq = np.sum(freqs * magnitudes) / np.sum(magnitudes)  # Mean frequency
        dynamic_mid_freq_range = (mean_freq * 0.5, mean_freq * 1.5)  # Use a factor around the mean frequency
    elif method == 'median':
        cumulative_psd = np.cumsum(psd)
        median_freq = freqs[np.searchsorted(cumulative_psd, cumulative_psd[-1] / 2)]  # Median frequency
        dynamic_mid_freq_range = (median_freq * 0.5, median_freq * 1.5)  # Factor around the median frequency
    elif method == 'percentile':
        lower_bound = np.percentile(freqs, percentile)
        upper_bound = np.percentile(freqs, 100 - percentile)
        dynamic_mid_freq_range = (lower_bound, upper_bound)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'fixed', 'mean', 'median', or 'percentile'.")
    
    # Calculate the sum of magnitudes within the mid-frequency range
    mid_freq_content = np.sum(magnitudes[(freqs >= dynamic_mid_freq_range[0]) & (freqs <= dynamic_mid_freq_range[1])])
    
    return mid_freq_content

@name("spectral_peak_to_valley_ratio")
def calculate_spectral_peak_to_valley_ratio(magnitudes, **kwargs):
    """
    Calculate the spectral peak-to-valley ratio from a given array of magnitudes.

    The peak-to-valley ratio is defined as the ratio of the maximum peak magnitude 
    to the minimum valley magnitude in the given signal spectrum. Peaks are local 
    maxima, and valleys are local minima of the magnitude spectrum. 

    Parameters
    ----------
    magnitudes : array_like
            A 1D array representing the magnitudes corresponding to each frequency component.

    Returns
    -------
    np.array
        A single-element array containing the peak-to-valley ratio. If no peaks 
        or valleys are found, returns an array with NaN to indicate invalid output.

    References
    ----------
        - Biberger, T., & Ewert, S. D. (2022). Binaural detection thresholds and 
        audio quality of speech and music signals in complex acoustic environments. 
        Frontiers in Psychology, 13, 994047. https://doi.org/10.3389/FPSYG.2022.994047/BIBTEX
        - https://openlab.help.agilent.com/en/index.htm#t=mergedProjects/DataAnalysis/27021601168830603.htm
    """
    peaks, _ = find_peaks(magnitudes)
    valleys, _ = find_peaks(-magnitudes)
    if len(peaks) == 0 or len(valleys) == 0:
        return np.nan
    
    peak_to_valley_ratio = np.max(magnitudes[peaks]) / np.min(magnitudes[valleys])
    return peak_to_valley_ratio

@name("spectral_valley_depth_mean")
def calculate_spectral_valley_depth_mean(magnitudes, **kwargs):
    """
    Calculate the mean of the spectral valley depths.

    This function identifies valleys in the magnitude spectrum by finding peaks
    in the negative of the magnitudes (indicating valleys) and computes the mean 
    depth of these valleys. If no valleys are found, it returns NaN.

    Parameters:
    ---------
    magnitudes : array-like
        A 1D array representing the magnitudes corresponding to each frequency component.

    Returns:
    -------
    np.array
        Mean of the valley depths or NaN if no valleys are found.
        
    Reference:
    ---------
        - Ananthapadmanabha, T. v, Ramakrishnan, A. G., Sharma, S., & Anantha, J. (2015). 
        Significance of the levels of spectral valleys with application to front/back 
        distinction of vowel sounds. 2. https://arxiv.org/abs/1506.04828v2
    """
    valleys, _ = find_peaks(-magnitudes)
    if len(valleys) == 0:
        return np.nan
    valley_depth_mean = np.mean(magnitudes[valleys])
    return valley_depth_mean

@name("spectral_valley_depth_std")
def calculate_spectral_valley_depth_std(magnitudes, **kwargs):
    """
    Calculate the standard deviation of the spectral valley depths.

    This function identifies valleys in the magnitude spectrum by finding peaks
    in the negative of the magnitudes (indicating valleys) and computes the standard 
    deviation of the valley depths. If no valleys are found, it returns NaN.

    Parameters:
    ---------
    magnitudes : array-like
        A 1D array representing the magnitudes corresponding to each frequency component.

    Returns:
    --------
    np.array
        Standard deviation of the valley depths or NaN if no valleys are found.
            
    Reference:
    ---------
        - Ananthapadmanabha, T. v, Ramakrishnan, A. G., Sharma, S., & Anantha, J. (2015). 
        Significance of the levels of spectral valleys with application to front/back 
        distinction of vowel sounds. 2. https://arxiv.org/abs/1506.04828v2
    """
    valleys, _ = find_peaks(-magnitudes)
    if len(valleys) == 0:
        return np.nan
    valley_depth_std = np.std(magnitudes[valleys])
    return valley_depth_std

@name("spectral_valley_depth_variance")
def calculate_spectral_valley_depth_variance(magnitudes, **kwargs):
    """
    Calculate the variance of the spectral valley depths.

    This function identifies valleys in the magnitude spectrum by finding peaks
    in the negative of the magnitudes (indicating valleys) and computes the variance 
    of these valleys' depths. If no valleys are found, it returns NaN.

    Parameters:
    ---------
    magnitudes : array-like
        A 1D array representing the magnitudes corresponding to each frequency component.

    Returns:
    --------
    np.array
        Variance of the valley depths or NaN if no valleys are found.
            
    Reference:
    ---------
        - Ananthapadmanabha, T. v, Ramakrishnan, A. G., Sharma, S., & Anantha, J. (2015). 
        Significance of the levels of spectral valleys with application to front/back 
        distinction of vowel sounds. 2. https://arxiv.org/abs/1506.04828v2
    """
    valleys, _ = find_peaks(-magnitudes)
    if len(valleys) == 0:
        return np.nan
    valley_depth_variance = np.var(magnitudes[valleys])
    return valley_depth_variance

@name("spectral_valley_width_mode")
def calculate_spectral_valley_width_mode(magnitudes, **kwargs):
    """
    Calculate the mode of the spectral valley widths.

    This function identifies valleys in the magnitude spectrum by finding peaks
    in the negative of the magnitudes. It then calculates the mode of the widths 
    between consecutive valleys. If fewer than two valleys are found, it returns NaN.

    Parameters:
    ---------
    magnitudes : array-like
        A 1D array representing the magnitudes corresponding to each frequency component.

    Returns:
    --------
    np.array
        Mode of the valley widths or NaN if fewer than two valleys are found.
    
    Reference:
    ---------
        - Ananthapadmanabha, T. v, Ramakrishnan, A. G., Sharma, S., & Anantha, J. (2015). 
        Significance of the levels of spectral valleys with application to front/back 
        distinction of vowel sounds. 2. https://arxiv.org/abs/1506.04828v2
    """
    valleys, _ = find_peaks(-magnitudes)
    if len(valleys) < 2:
        return np.nan
    
    valley_widths = np.diff(valleys)
    if len(valley_widths) == 0:
        return np.nan
    elif len(valley_widths) == 1:
        return float(valley_widths[0])
    
    
    valley_width_mode = mode(valley_widths)[0]
    
    try:
        valley_width_mode = mode(valley_widths)[0]
        if hasattr(valley_width_mode, 'mode'):
            return float(valley_width_mode.mode[0])
        else:
            return float(valley_width_mode[0][0])
    except Exception as e:
        return np.nan

@name("spectral_valley_width_std")
def calculate_spectral_valley_width_std(magnitudes, **kwargs):
    """
    Calculate the standard deviation of the spectral valley widths.

    This function identifies valleys in the magnitude spectrum by finding peaks
    in the negative of the magnitudes and calculates the standard deviation of the 
    widths between consecutive valleys. If fewer than two valleys are found, it returns NaN.

    Parameters:
    ----------
    magnitudes : array-like
        A 1D array representing the magnitudes corresponding to each frequency component.

    Returns:
    --------
    np.array
        Standard deviation of the valley widths or NaN if fewer than two valleys are found.
            
    Reference:
    ---------
        - Ananthapadmanabha, T. v, Ramakrishnan, A. G., Sharma, S., & Anantha, J. (2015). 
        Significance of the levels of spectral valleys with application to front/back 
        distinction of vowel sounds. 2. https://arxiv.org/abs/1506.04828v2
    """
    valleys, _ = find_peaks(-magnitudes)
    if len(valleys) < 2:
        return np.nan
    valley_widths = np.diff(valleys)
    valley_width_std = np.std(valley_widths)
    return valley_width_std

@name("spectral_subdominant_valley")
def calculate_spectral_subdominant_valley(magnitudes, **kwargs):
    """
    Calculate the second-largest valley in the magnitude spectrum.

    Parameters:
    ----------
    magnitudes : array-like
        A 1D array representing the magnitudes corresponding to each frequency component.

    Returns:
    ---------
    np.array
        An array containing the second-largest valley value or NaN if there are fewer than two valleys.
    """
    valleys, _ = find_peaks(-magnitudes)
    if len(valleys) < 2:
        return np.nan
    sorted_valleys = np.sort(magnitudes[valleys])
    subdominant_valley = sorted_valleys[-2] if len(sorted_valleys) >= 2 else np.nan
    return subdominant_valley

@name("spectral_valley_count")
def calculate_spectral_valley_count(magnitudes, **kwargs):
    """
    Calculate the number of valleys in the magnitude spectrum.

    Parameters:
    ----------
    magnitudes : array-like
        Array of magnitude values from the spectrum.

    Returns:
    np.array
        An array containing the number of valleys in the magnitude spectrum.
    
    Reference:
    ---------
        - Ananthapadmanabha, T. v, Ramakrishnan, A. G., Sharma, S., & Anantha, J. (2015). 
        Significance of the levels of spectral valleys with application to front/back 
        distinction of vowel sounds. 2. https://arxiv.org/abs/1506.04828v2
    """
    valleys, _ = find_peaks(-magnitudes)
    return len(valleys)

@name("spectral_peak_broadness")
def calculate_spectral_peak_broadness(freqs, magnitudes, **kwargs):
    """
    Calculate the average distance between peaks in the magnitude spectrum, indicating peak broadness.

    Parameters:
    ----------
    freqs : array-like
        Array of frequency values.
    magnitudes : array-like
        Array of magnitude values from the spectrum.

    Returns:
    --------
    np.array
        An array containing the average distance between peaks or NaN if there are fewer than two peaks.
                
        
    Reference:
    ----------
        - https://terpconnect.umd.edu/~toh/spectrum/PeakFindingandMeasurement.htm
    """
    peaks, _ = find_peaks(magnitudes)
    if len(peaks) < 2:
        return np.nan
    peak_widths = np.diff(peaks)
    peak_broadness = np.mean(peak_widths)
    return peak_broadness

@name("spectral_valley_broadness")
def calculate_spectral_valley_broadness(freqs, magnitudes, **kwargs):
    """
    Calculate the average distance between valleys in the magnitude spectrum, indicating valley broadness.

    Parameters:
    -----------
    freqs : array-like
        Array of frequency values.
    magnitudes : array-like
        Array of magnitude values from the spectrum.

    Returns:
    --------
    np.array
        An array containing the average distance between valleys or NaN if there are fewer than two valleys.
        
    Reference:
    ---------
        - Ananthapadmanabha, T. v, Ramakrishnan, A. G., Sharma, S., & Anantha, J. (2015). 
        Significance of the levels of spectral valleys with application to front/back 
        distinction of vowel sounds. 2. https://arxiv.org/abs/1506.04828v2
    """
    valleys, _ = find_peaks(-magnitudes)
    if len(valleys) < 2:
        return np.nan
    valley_widths = np.diff(valleys)
    valley_broadness = np.mean(valley_widths)
    return valley_broadness

@name("spectral_range")
def calculate_spectral_range(freqs, **kwargs):
    """
    Calculate the range of frequencies in the spectrum.

    Parameters:
    -----------
    freqs : array-like
        Array of frequency values.

    Returns:
    --------
    np.array
        An array containing the range of frequencies.
            
    Reference
    ---------
        - Galar, D., & Kumar, U. (2017). Preprocessing and Features. EMaintenance, 129–177. 
        https://doi.org/10.1016/B978-0-12-811153-6.00003-8
    """
    freq_range = np.max(freqs) - np.min(freqs)
    return freq_range

@name("spectral_trimmed_mean")
def calculate_spectral_trimmed_mean(freqs, magnitudes, trim_percent=0.1, **kwargs):
    """
    Calculate the spectral trimmed mean of a set of frequencies.
    
    The trimmed mean is calculated by sorting the magnitudes, removing the top and bottom 
    `trim_percent` of the values, and then taking the mean of the remaining frequencies.
    
    Parameters:
    -----------
    freqs : array-like
        The input frequencies.
    magnitudes : array-like
        The corresponding magnitudes of the frequencies.
    trim_percent : float, optional (default=0.1)
        The percentage of the top and bottom magnitudes to trim.
        
    Returns:
    --------
    float
        The trimmed mean of the input frequencies.
        
    References:
    ----------
        - Galar, D., & Kumar, U. (2017). Preprocessing and Features. EMaintenance, 129–177. 
        https://doi.org/10.1016/B978-0-12-811153-6.00003-8
    """
    sorted_indices = np.argsort(magnitudes)
    lower_limit = int(trim_percent * len(magnitudes))
    upper_limit = int((1 - trim_percent) * len(magnitudes))
    trimmed_indices = sorted_indices[lower_limit:upper_limit]
    trimmed_mean = np.mean(freqs[trimmed_indices])
    return trimmed_mean

@name("harmonic_product_spectrum")
def calculate_harmonic_product_spectrum(magnitudes, **kwargs):
    """
    Calculate the sum of the Harmonic Product Spectrum (HPS) from the given magnitudes.

    This function computes the HPS and returns its sum as a single scalar feature. It is often 
    used as part of feature extraction pipelines in audio signal processing, 
    particularly for pitch detection, speech/music classification, or speaker recognition.

    Parameters:
    -----------
    magnitudes : numpy.array
            An array of magnitude values of the spectrum at the corresponding frequencies.

    Returns:
    --------
    float
        A single scalar value representing the sum of the Harmonic Product Spectrum. 
        This value reflects the overall strength of harmonic content in the input signal.
        
    References:
    -----------
        - Nhu, T. V., & Sawada, H. (2018). Intoning Speech Performance of the Talking Robot 
        for Vietnamese Language Case. MHS 2018 - 2018 29th International Symposium on 
        Micro-NanoMechatronics and Human Science. https://doi.org/10.1109/MHS.2018.8886911

    Notes:
    ------
    - This function returns only the sum of the HPS, which is useful as a compact 
    feature for tasks like classification or clustering.
    - For more detailed analysis, you may want to retain the full HPS spectrum 
    instead of just the sum.
    """
    hps = np.copy(magnitudes)
    for h in range(2, 5):
        decimated = magnitudes[::h]
        hps[:len(decimated)] *= decimated
    return np.sum(hps)

@name("smoothness")
def calculate_smoothness(magnitudes, **kwargs):
    """
    Calculate the smoothness of a signal based on the magnitude values.

    Smoothness is determined as the sum of the squared differences 
    between consecutive magnitudes. This provides a measure of how much 
    the signal changes between successive points, which can be used to 
    quantify the stability or fluctuation of the signal over time.

    Parameters:
    ----------
    magnitudes (numpy.ndarray):
        A 1D array of magnitudes (e.g., signal amplitude) from which smoothness is to be calculated.

    Returns:
    -------
    numpy.ndarray:
        A 1D array containing the calculated smoothness value.
        
    References:
    -----------
        - Liu, W., He, C., & Sun, L. (2021). Spectral-Smoothness and Non-Local Self-Similarity Regularized
        Subspace Low-Rank Learning Method for Hyperspectral Mixed Denoising. Remote Sensing 2021, Vol. 13, 
        Page 3196, 13(16), 3196. https://doi.org/10.3390/RS13163196
    """
    smoothness = np.sum(np.diff(magnitudes)**2)
    return smoothness

@name("roughness")
def calculate_roughness(magnitudes, **kwargs):
    """
    Calculate the roughness of a signal based on the magnitude values.

    Roughness is determined as the sum of the absolute differences 
    between consecutive magnitudes. This provides a measure of the 
    overall variation or "roughness" of the signal, which reflects how 
    irregular or jagged the signal is over time.

    Parameters:
    -----------
    magnitudes (numpy.ndarray): 
        A 1D array of magnitudes (e.g., signal amplitude) from which roughness is to be calculated.

    Returns:
    --------
    numpy.ndarray: 
        A 1D array containing the calculated roughness value.
    """
    roughness = np.sum(np.abs(np.diff(magnitudes)))
    return roughness


# features haven't been implemented yet and cannot find reference
# https://musicinformationretrieval.com/spectral_features.html
# Spectral Strongest Frequency Magnitude: Magnitude of the strongest frequency component.
# Spectral Strongest Frequency: The strongest frequency component.
# Spectral Frequency at Median Power: The frequency at which the power is median.
# Spectral Cumulative Frequency Below 25% Power: The frequency below which 25% of the spectral power is contained.
# Spectral Cumulative Frequency Above 25% Power: The frequency above which 25% of the spectral power is contained.
# Spectral Cumulative Frequency Above 50% Power: The frequency above which 50% of the spectral power is contained.
# Spectral Power Ratio (between different bands, **kwargs): The ratio of power between different frequency bands.
# Spectral Centroid Shift: The change in spectral centroid over time.
# Spectral Flux Shift: The change in spectral flux over time.
# Spectral Rolloff Shift: The change in spectral rolloff over time.
# Spectral Energy Vector: Vector of spectral energy values.
# Spectral High Frequency Content: The amount of energy in the high-frequency band.
# Spectral Peak Distribution: The distribution of peaks in the spectrum.
# Spectral Valley Distribution: The distribution of valleys in the spectrum.
# Spectral Valley Depth Median: The median depth of valleys in the spectrum.
# Spectral Valley Depth Mode: The most frequent valley depth.
# Spectral Valley Width Mean: The mean width of valleys.
# Spectral Valley Width Median: The median width of valleys.
# Spectral Valley Width Variance: The variance of valley widths.
# Spectral Dominant Valley: The most prominent valley.
# Spectral Peak Sharpness: The sharpness of the spectral peaks.
# Spectral Valley Sharpness: The sharpness of the spectral valleys.
# Spectral Dominant Peak: The most prominent peak.
# Spectral Subdominant Peak: The second most prominent peak.