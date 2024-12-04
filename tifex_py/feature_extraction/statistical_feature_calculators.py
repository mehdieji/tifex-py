import numpy as np
import math
from scipy.stats import skew, kurtosis, moment, gmean, hmean, trim_mean, entropy, linregress, mode, pearsonr
from statsmodels.tsa.stattools import acf, adfuller
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import detrend, argrelextrema, find_peaks, find_peaks_cwt, ricker
from itertools import groupby
from scipy.ndimage.filters import convolve
from scipy import stats
import warnings

from tifex_py.utils.decorators import name, exclude

# def extract_all_features

@name("mean")
def calculate_mean(signal, **kwargs):
    """
    Calculates the mean of the given signal.

    Parameters:
    ---------
    signal: np.array
        An array of values corresponding to the signal.

    Returns:
    -------
    float
        The mean of the signal.
    
    References:
    ----------
        - Nanopoulos, A., Alcock, R., & Manolopoulos, Y. (2001). Feature-based 
        Classification of Time-series Data. https://www.researchgate.net/publication/234800113_Feature-based_Classification_of_Time-series_Data
    """
    return np.mean(signal)

@name("geometric_mean")
def calculate_geometric_mean(signal, **kwargs):
    """
    Calculates the geometric mean of the given signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.

    Returns:
    -------
    float
        The geometric mean of the signal.
    
    References:
    ----------
        - Chaddad, A., & M.D., R. R. C. (2014). Statistical feature selection for enhanced detection of brain tumor. 
        Https://Doi.Org/10.1117/12.2062143, 9217, 260–267. https://doi.org/10.1117/12.2062143
    """  
    positive_signal = signal[signal > 0]
    
    if len(positive_signal) == 0:
        return np.nan
        
    try:
        return gmean(positive_signal)
    except Exception as e:
        return np.nan

@name("harmonic_mean")
def calculate_harmonic_mean(signal, **kwargs):
    """
    Calculates the harmonic mean of the given signal.
    Only positive values in the signal are considered.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.

    Returns:
    -------
    float
        The harmonic mean of the signal.
    
    References:
    ----------
        - Chaddad, A., & M.D., R. R. C. (2014). Statistical feature selection for enhanced detection of brain tumor. 
        Https://Doi.Org/10.1117/12.2062143, 9217, 2should60–267. https://doi.org/10.1117/12.2062143
    """
    # Filter out non-positive values
    positive_signal = signal[signal > 0]
    
    if len(positive_signal) == 0:
        return np.nan
    try:
        return hmean(positive_signal)
    except Exception as e:
        return np.nan

@name("trimmed_mean_{}", "trimmed_mean_thresholds")
def calculate_trimmed_mean(signal, trimmed_mean_thresholds, **kwargs):
    """
    Calculate the trimmed mean of the given signal for different proportions.
    
    The trimmed mean excludes a fraction of the smallest and largest values 
    from the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the signal.
    trimmed_mean_thresholds (list): 
        List of proportions to cut from each end of the signal.

    Returns:
    -------
    float
        The trimmed means for each proportion.
    
    References:
    ----------
        - Chaddad, A., & M.D., R. R. C. (2014). Statistical feature selection for enhanced detection of brain tumor. 
        Https://Doi.Org/10.1117/12.2062143, 9217, 260–267. https://doi.org/10.1117/12.2062143
    """
    feats = []
    for proportiontocut in trimmed_mean_thresholds:
        feats.append(trim_mean(signal, proportiontocut=proportiontocut))
    return np.array(feats)

@name("mean_of_abs")
def calculate_mean_abs(signal, **kwargs):
    """
    Calculate the mean of the absolute values of the given signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.

    Returns:
    -------
    float
        The mean of the absolute values of the signal.
    
    References:
    ----------
        - Myroniv, B., Wu, C.-W., Ren, Y., Christian, A., Bajo, E., & Tseng, 
        Y.-C. (2017). Analyzing User Emotions via Physiology Signals. 
        https://www.researchgate.net/publication/323935725
        - Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction 
        and selection for EMG signal classification. Expert Systems with Applications, 39(8), 
        7420–7431. https://doi.org/10.1016/J.ESWA.2012.01.102
        - Purushothaman, G., & Vikas, · Raunak. (2018). Identification of a feature selection
        based pattern recognition scheme for finger movement recognition from multichannel 
        EMG signals. Australasian Physical & Engineering Sciences in Medicine, 41, 549–559. 
        https://doi.org/10.1007/s13246-018-0646-7
    """
    return np.mean(np.abs(signal))

@name("geometric_mean_of_abs")
def calculate_geometric_mean_abs(signal, **kwargs):
    """
    Calculate the geometric mean of the absolute values of the given signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.

    Returns:
    -------
    float
        An array containing the geometric mean of the absolute values of the signal.

    References:
    ----------
        - Gil’mutdinov, A. K., & Ushakov, P. A. (2017). Physical implementation of elements with fractal 
        impedance: State of the art and prospects. Journal of Communications Technology and Electronics, 
        62(5), 441–453. https://doi.org/10.1134/S1064226917050060/METRICS
    """
    return gmean(np.abs(signal))

@name("harmonic_mean_of_abs")
def calculate_harmonic_mean_abs(signal, **kwargs):
    """
    Calculate the harmonic mean of the absolute values of the given signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.

    Returns:
    -------
        float
            An array containing the harmonic mean of the absolute values of the signal.
    """
    return hmean(np.abs(signal))

@name("trimmed_mean_abs_{}", "trimmed_mean_thresholds")
def calculate_trimmed_mean_abs(signal, trimmed_mean_thresholds, **kwargs):
    """
    Calculate the trimmed mean of the absolute values of the given signal for 
    different proportions.
    
    The trimmed mean excludes a fraction of the smallest and largest values 
    from the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the signal.
    trimmed_mean_thresholds: list
        List of proportions to cut from each end of the absolute values of the signal.

    Returns:
    -------
    np.array
        An array containing the trimmed means for each proportion.
    """
    feats = []
    for proportiontocut in trimmed_mean_thresholds:
        feats.append(trim_mean(np.abs(signal), proportiontocut=proportiontocut))
    return np.array(feats)

@name("std")
def calculate_std(signal, **kwargs):
    """
    Calculates the standard deviation of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        An array containing the standard deviation of the signal.
        
    References:
    ---------
        - Nanopoulos, A., Alcock, R., & Manolopoulos, Y. (2001). Feature-based Classification 
        of Time-series Data. https://www.researchgate.net/publication/234800113_Feature-based_Classification_of_Time-series_Data
    """
    return np.std(signal)

@name("std_of_abs")
def calculate_std_abs(signal, **kwargs):
    """
    Calculates the standard deviation of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        An array containing the standard deviation of the absolute values of the signal.
    """
    return np.std(np.abs(signal))

@name("skewness")
def calculate_skewness(signal, **kwargs):
    """
    Calculates the skewness of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        An array containing the skewness of the signal.
        
    References:
    ----------
        - Nanopoulos, A., Alcock, R., & Manolopoulos, Y. (2001). Feature-based Classification 
        of Time-series Data. https://www.researchgate.net/publication/234800113_Feature-based_Classification_of_Time-series_Data
        - Whitaker, B. M., Suresha, P. B., Liu, C., -,  al, Wang, G., Huang, J., Zhang -, F., Mohd Khan, S., Ali Khan, A., 
        Farooq -, O., Khorshidtalab, A., E Salami, M. J., & Hamedi, M. (2013). Robust classification of motor imagery EEG signals 
        using statistical time–domain features. Physiological Measurement, 34(11), 1563. https://doi.org/10.1088/0967-3334/34/11/1563
    """ 
    return skew(signal)

@name("skewness_of_abs")
def calculate_skewness_abs(signal, **kwargs):
    """
    Calculates the skewness of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        An array containing the skewness of the absolute values of the signal.
    """
    return skew(np.abs(signal))

@name("kurtosis")
def calculate_kurtosis(signal, **kwargs):
    """
    Calculates the kurtosis of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The kurtosis of the signal.
        
    References:
    ----------
        - Nanopoulos, A., Alcock, R., & Manolopoulos, Y. (2001). Feature-based Classification 
        of Time-series Data. https://www.researchgate.net/publication/234800113_Feature-based_Classification_of_Time-series_Data
        - Whitaker, B. M., Suresha, P. B., Liu, C., -,  al, Wang, G., Huang, J., Zhang -, F., Mohd Khan, S., Ali Khan, A., 
        Farooq -, O., Khorshidtalab, A., E Salami, M. J., & Hamedi, M. (2013). Robust classification of motor imagery EEG signals 
        using statistical time–domain features. Physiological Measurement, 34(11), 1563. https://doi.org/10.1088/0967-3334/34/11/1563
    """
    return kurtosis(signal)

@name("kurtosis_of_abs")
def calculate_kurtosis_abs(signal, **kwargs):
    """
    Calculates the kurtosis of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The kurtosis of the absolute values of the signal.
    """
    return kurtosis(np.abs(signal))

@name("median")
def calculate_median(signal, **kwargs):
    """
    Calculates the median of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The median  of the signal.
        
    References:
    ---------
        - Banos, O., Damas, M., Pomares, H., Prieto, A., & Rojas, I. (2012). Daily living activity recognition 
        based on statistical feature quality group selection. Expert Systems with Applications, 39(9), 8013–8021.
        https://doi.org/10.1016/J.ESWA.2012.01.164
    """
    return np.median(signal)

@name("median_of_abs")
def calculate_median_abs(signal, **kwargs):
    """
    Calculates the median of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The median of the absolute values of the signal.
    """
    return np.median(np.abs(signal))

@name("min")
def calculate_min(signal, **kwargs):
    """
    Calculates the minimum value of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The minimum value of the signal.
        
    References:
    ---------
        -Quanz, B., Fei, H., Huan, J., Evans, J., Frost, V., Minden, G., Deavours, D., Searl, L., 
        Depardo, D., Kuehnhausen, M., Fokum, D., Zeets, M., & Oguna, A. (2009). Anomaly detection 
        with sensor data for distributed security. Proceedings - International Conference on Computer 
        Communications and Networks, ICCCN. https://doi.org/10.1109/ICCCN.2009.5235262
    """
    min_val = np.min(signal)
    return min_val

@name("min_of_abs")
def calculate_min_abs(signal, **kwargs):
    """
    Calculates the minimum value of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The minimum value of the absolute values of the signal.
    """
    min_abs_val = np.min(np.abs(signal))
    return min_abs_val

@name("max")
def calculate_max(signal, **kwargs):
    """
    Calculates the maximum value of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The maximum value of the signal.
        
    References:
    ---------
        - Whitaker, B. M., Suresha, P. B., Liu, C., -,  al, Wang, G., Huang, J., Zhang -, F., Mohd Khan, S., Ali Khan, A., 
        Farooq -, O., Khorshidtalab, A., E Salami, M. J., & Hamedi, M. (2013). Robust classification of motor imagery EEG signals 
        using statistical time–domain features. Physiological Measurement, 34(11), 1563. https://doi.org/10.1088/0967-3334/34/11/1563
    """
    max_val = np.max(signal)
    return max_val

@name("max_of_abs")
def calculate_max_abs(signal, **kwargs):
    """
    Calculates the maximum value of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The maximum value of the absolute values of the signal.
        
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    max_abs_val = np.max(np.abs(signal))
    return max_abs_val

@name("range")
def calculate_range(signal, **kwargs):
    """
    Calculates the range of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    np.array
        The range of the signal.
        
    References:
    ---------
        - Wan, X., Wang, W., Liu, J., & Tong, T. (2014). Estimating the sample mean and standard deviation from 
        the sample size, median, range and/or interquartile range. BMC Medical Research Methodology, 14(1), 
        1–13. https://doi.org/10.1186/1471-2288-14-135/TABLES/3
    """
    return np.max(signal) - np.min(signal)

@name("range_of_abs")
def calculate_range_abs(signal, **kwargs):
    """
    Calculates the range of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
        
    Returns:
    -------
    float
        The range of the absolute values of the signal.
    """
    abs_signal = np.abs(signal)
    return np.max(abs_signal) - np.min(abs_signal)

@name("var")
def calculate_variance(signal, **kwargs):
    """
    Calculates the variance of the signal.

    Parameters:
    ----------
    signal : np.array
        The input time series.
        
    Returns:
    -------
    float
        The variance of the signal.
        
    References:
    ---------
        - Whitaker, B. M., Suresha, P. B., Liu, C., -,  al, Wang, G., Huang, J., Zhang -, F., Mohd Khan, S., Ali Khan, A., 
        Farooq -, O., Khorshidtalab, A., E Salami, M. J., & Hamedi, M. (2013). Robust classification of motor imagery EEG signals 
        using statistical time–domain features. Physiological Measurement, 34(11), 1563. https://doi.org/10.1088/0967-3334/34/11/1563
    """
    return np.var(signal)

@name("var_of_abs")
def calculate_variance_abs(signal, **kwargs):
    """
    Calculates the variance of the absolute values of the signal.

    Parameters:
    ----------
    signal : np.array
        The input time series.
        
    Returns:
    -------
    float
        The variance of the absolute values of the signal.
    """
    return np.var(np.abs(signal))

@name("variance_larger_than_standard_deviation")
def calculate_variance_larger_than_standard_deviation(signal,**kwargs):
    """
    Determine if the variance of a signal is greater than its standard deviation.

    Parameters:
    ----------
    signal : np.array
        The input time series.
    
    Returns:
    -------
    bool
        True if the variance of `signal` is greater than its standard deviation, 
        False otherwise.
    
    Notes:
    -----
    Since variance is the square of the standard deviation, the result will always 
    be True for non-zero signals, as long as the standard deviation is a positive 
    non-zero number.
    
    References:
    -----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.var(signal) > np.std(signal)

@name("iqr")
def calculate_interquartile_range(signal, **kwargs):
    """
    Calculate the interquartile range (IQR) of the given signal.

    The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the signal.

    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        An array containing the interquartile range of the signal.
    
    References:
    ----------
        - Bedeeuzzaman, M., Farooq, O., & U Khan, Y. (2012). Automatic Seizure Detection using Inter Quartile Range. 
        International Journal of Computer Applications, 44(11), 1–5. https://doi.org/10.5120/6304-8614
    """
    return np.percentile(signal, 75) - np.percentile(signal, 25)

# TODO: FIX THIS 
@name("quantile_{}", "q")
def calculate_quantile(signal, q, **kwargs):
    """
    Calculate the quantile of the time series. The first and third quartiles are highlighted.
    
    Parameters:
    -----------
    signal : np.array
        The input time series
    q : array of float
            
    Returns:
    -------
    np.array
        An array containing the quantile values of the signal
    """
    if len(signal) == 0:
        return np.NaN
    return np.quantile(signal, q)

@name("mean_abs_deviation")
def calculate_mean_absolute_deviation(signal, **kwargs):
    """
    Calculate the mean absolute deviation (MAD) of the given signal.

    The MAD is the average of the absolute deviations from the mean of the signal.

    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The mean absolute deviation of the signal.
    
    References:
    ----------
        - Pham-Gia, T., & Hung, T. L. (2001). The mean and median absolute deviations. Mathematical and Computer Modelling,
        34(7–8), 921–936. https://doi.org/10.1016/S0895-7177(01)00109-1
    """
    return np.mean(np.abs(signal - np.mean(signal)))

@name("rms")
def calculate_root_mean_square(signal, **kwargs):
    """
    Calculate the root mean square (RMS) of the given signal.

    The RMS is the square root of the mean of the squares of the signal values.

    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The root mean square of the signal.
    
    References:
    ----------
        - Whitaker, B. M., Suresha, P. B., Liu, C., -,  al, Wang, G., Huang, J., Zhang -, F., 
        Mohd Khan, S., Ali Khan, A., Farooq -, O., Khorshidtalab, A., E Salami, M. J., & 
        Hamedi, M. (2013). Robust classification of motor imagery EEG signals using statistical 
        time–domain features. Physiological Measurement, 34(11), 1563. https://doi.org/10.1088/0967-3334/34/11/1563
    """
    return np.sqrt(np.mean(signal**2))

@name("energy")
def calculate_signal_energy(signal, **kwargs):
    """
    Calculate the energy of the given signal.
    
    The energy of the signal is the sum of the squares of its values.

    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The energy of the signal.
    
    References:
    -----------
        - Rafiuddin, N., Khan, Y. U., & Farooq, O. (2011). Feature extraction and classification of EEG for automatic 
        seizure detection. 2011 International Conference on Multimedia, Signal Processing and Communication Technologies, 
        IMPACT 2011, 184–187. https://doi.org/10.1109/MSPCT.2011.6150470
    """
    return np.sum(signal**2)

@name("log_of_energy")
def calculate_log_energy(signal, **kwargs):
    """
    Calculate the logarithm of the energy of the given signal.
    
    The log energy is the natural logarithm of the sum of the squares of the signal values.

    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The logarithm of the energy of the signal.
        
    References:
    -----------
    Implementation based on:
        https://mathworks.com/help/audio/ref/mfcc.html
    """
    return np.log(np.sum(signal**2))

@name("entropy")
def calculate_entropy(signal, window_size, **kwargs):
    """
    Calculate the entropy of the given signal.
    
    The entropy is a measure of the uncertainty or randomness in the signal,
    calculated from its histogram.

    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The entropy of the signal.
    
    References:
    ----------
        - Guido, R. C. (2018). A tutorial review on entropy-based handcrafted feature extraction for information fusion.
        Information Fusion, 41, 161–175. https://doi.org/10.1016/J.INFFUS.2017.09.006
    """
    try:
        # Calculate the histogram
        hist, _ = np.histogram(signal, bins=window_size//2, density=True)
        # Replace zero values with a small epsilon
        epsilon = 1e-10
        hist = np.where(hist > 0, hist, epsilon)
        entropy = -np.sum(hist * np.log2(hist))
    except ValueError as e:
        entropy = np.nan
    except Exception as e:
        entropy = np.nan
    return entropy

@name("sample_entropy")
def calculate_sample_entropy(signal, **kwargs):
    """
    Calculate the sample entropy of a signal.

    Sample entropy is a measure of the complexity or irregularity in a time series. It quantifies 
    the likelihood that similar patterns of data points in a time series will remain similar on 
    the next incremental comparison.

    Parameters:
    -----------
    signal : array-like
        The input time series data for which sample entropy is to be calculated.

    Returns:
    --------
    float
        The calculated sample entropy of the input signal.

    References:
    -----------
        - antropy.sample_entropy — antropy 0.1.6 documentation. (n.d.). Retrieved July 22, 2024, from 
        https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html
        - Kumar, Y., Dewal, M. L., & Anand, R. S. (2012). Features extraction of EEG signals using approximate and sample entropy. 
        2012 IEEE Students’ Conference on Electrical, Electronics and Computer Science: Innovation for Humanity, SCEECS 2012. 
        https://doi.org/10.1109/SCEECS.2012.6184830
        - Delgado-Bonal, A., & Marshak, A. (2019). Approximate Entropy and Sample Entropy: A Comprehensive Tutorial. 
        Entropy 2019, Vol. 21, Page 541, 21(6), 541. https://doi.org/10.3390/E21060541
    """
    N = len(signal)
    m = 2
    r = 0.2 * np.std(signal)  # r known as the tolerance is typically set as a fraction of the standard deviation

    def _phi(m):
        X = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)

    A = _phi(m + 1)
    B = _phi(m)

    return -np.log(A / B)

@name("differential_entropy")
def calculate_differential_entropy(signal, **kwargs):
    """
    Calculate the differential entropy of a signal.

    Differential entropy is a measure of the continuous entropy of a random variable. It is 
    an extension of the concept of entropy to continuous probability distributions, which 
    provides a measure of uncertainty or randomness in a signal.

    Parameters:
    -----------
    signal : array-like
        The input time series data for which differential entropy is to be calculated.

    Returns:
    --------
    float
        The calculated differential entropy of the input signal.

    References:
    -----------
        - Zhu, Y., & Zhong, Q. (2021). Differential Entropy Feature Signal Extraction Based on 
        Activation Mode and Its Recognition in Convolutional Gated Recurrent Unit Network. Frontiers 
        in Physics, 8, 629620. https://doi.org/10.3389/FPHY.2020.629620/BIBTEX
    """
    probability, _ = np.histogram(signal, bins=10, density=True)
    probability = probability[probability > 0]
    return entropy(probability)

@name("approximate_entropy_{}", "r")
def calculate_approximate_entropy(signal, m, r, **kwargs):
    """
    Calculate the approximate entropy of a signal.

    Approximate entropy is a statistic used to quantify the amount of regularity and the 
    unpredictability of fluctuations in a time series. Lower values indicate a more 
    predictable series, while higher values indicate more complexity or randomness.
    
    Parameters:
    -----------
    signal : array-like
        The input time series data for which differential entropy is to be calculated.
    m: int
        Length of compared run of data.
    r: list of floats
        Filtering levels.

    Returns:
    --------
    float
        The calculated differential entropy of the input signal.

    References:
    -----------
        - Delgado-Bonal, A., & Marshak, A. (2019). Approximate Entropy and Sample Entropy: A Comprehensive Tutorial. 
        Entropy 2019, Vol. 21, Page 541, 21(6), 541. https://doi.org/10.3390/E21060541
        - Kumar, Y., Dewal, M. L., & Anand, R. S. (2012). Features extraction of EEG signals using approximate and sample entropy. 
        2012 IEEE Students’ Conference on Electrical, Electronics and Computer Science: Innovation for Humanity, SCEECS 2012. 
        https://doi.org/10.1109/SCEECS.2012.6184830
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    N = len(signal)
    feats = []
    def _phi(m, r):
            X = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0)
            return np.sum(np.log(C / (N - m + 1))) / (N - m + 1)
        
    for r in r:
        r *= np.std(signal)  # r known as the tolerance is typically set as a fraction of the standard deviation
        feats.append(_phi(m, r) - _phi(m + 1, r))
        
    return feats

@name("renyi_entropy")
def calculate_renyi_entropy(signal, window_size, renyi_alpha_parameter, **kwargs):
    """
    Calculate the Rényi entropy of a signal.

    Rényi entropy is a generalization of Shannon entropy that introduces a parameter 
    alpha, which controls the sensitivity to different parts of the probability 
    distribution. For alpha = 1, Rényi entropy reduces to Shannon entropy, and 
    for other values of alpha, it adjusts the focus on the distribution's tails 
    (either giving more weight to rare or frequent events).

    Parameters:
    -----------
    signal : array-like
        The input time series data or signal for which the Rényi entropy is to be calculated.
        
    window_size : int
        The size of the window used for histogram calculation. It determines the number 
        of bins in the histogram as window_size // 2.
        
    renyi_alpha_parameter : float
        The order alpha of the Rényi entropy. If alpha = 1, the function 
        returns the Shannon entropy.

    Returns:
    --------
    float
        The calculated Rényi entropy of the signal. If an error 
        occurs during the calculation, the function returns NaN.

    References:
    -----------
        - Beadle, E., Schroeder, J., Moran, B., & Suvorova, S. (2008). An overview of renyi entropy and some potential applications. 
        Conference Record - Asilomar Conference on Signals, Systems and Computers, 1698–1704. https://doi.org/10.1109/ACSSC.2008.5074715
    """
    try:
        # Calculate the histogram
        hist, _ = np.histogram(signal, bins=window_size//2, density=True)
        # Replace zero values with a small epsilon
        epsilon = 1e-10
        hist = np.where(hist > 0, hist, epsilon)

        if renyi_alpha_parameter == 1:
            # Return the Shannon entropy
            return -sum([p * (0 if p == 0 else np.log(p)) for p in hist])
        else:
            return (1 / (1 - renyi_alpha_parameter)) * np.log(sum([p ** renyi_alpha_parameter for p in hist]))
    except:
        return np.nan

@name("tsallis_entropy")
def calculate_tsallis_entropy(signal, window_size, tsallis_q_parameter, **kwargs):
    """
    Calculates the tsallis entropy of the signal.
    Tsallis entropy generalizes the Shannon entropy parameterized by a parameter 𝑞
    q, which introduces non-extensivity (a form of non-linearity) into the entropy measure.

    Parameters:
    ----------
    signal : array-like
        The input time series.
    window_size : int
        The size of the window used for histogram calculation. It determines the 
        number of bins in the histogram as window_size // 2.
    tsallis_q_parameter : float
        The q-parameter of the Tsallis entropy. If q = 1, the function
        returns the Boltzmann–Gibbs entropy.

    Returns:
    -------
    float
        The Tsallis entropy of the signal.

    References:
    ---------
        - Sneddon, R. (2007). The Tsallis entropy of natural information. Physica A: Statistical Mechanics 
        and Its Applications, 386(1), 101–118. https://doi.org/10.1016/J.PHYSA.2007.05.065
    """
    try:
        # Calculate the histogram
        hist, _ = np.histogram(signal, bins=window_size//2, density=True)
        # Replace zero values with a small epsilon
        epsilon = 1e-10
        hist = np.where(hist > 0, hist, epsilon)
        if tsallis_q_parameter == 1:
            # Return the Boltzmann–Gibbs entropy
            return -sum([p * (0 if p == 0 else np.log(p)) for p in hist])
        else:
            return (1 - sum([p ** tsallis_q_parameter for p in hist])) / (tsallis_q_parameter - 1)
    except:
        return np.nan

@name("svd_entropy")  
def calculate_svd_entropy(signal, svd_entropy_order, svd_entropy_delay, **kwargs):
    """
    Calculate the Singular Value Decomposition (SVD) Entropy of a given signal.

    SVD entropy is a measure of the complexity or randomness of a signal, calculated based on 
    the singular values of an embedding matrix constructed from the signal. The embedding matrix 
    is created using time-delay embedding, and the entropy is then computed from the normalized 
    singular values obtained from Singular Value Decomposition.

    Parameters:
    -----------
    signal : array-like
        The input time series data (1D signal) for which SVD entropy is to be calculated.

    svd_entropy_order : int
        The embedding dimension or order used to construct the embedding matrix. This determines 
        the number of rows in the embedding matrix.
        
    svd_entropy_delay : int
        The time delay used in the embedding process. This determines the step size between 
        consecutive points in the embedding matrix.

    Returns:
    --------
    float
        The calculated SVD entropy value. If an error occurs during the 
        calculation, the function returns NaN.

    References:
    -----------
        - Banerjee, M., & Pal, N. R. (2014). Feature selection with SVD entropy. Information Sciences—Informatics 
        and Computer Science, Intelligent Systems, Applications: An International Journal, 264, 118–134. 
        https://doi.org/10.1016/J.INS.2013.12.029
        - Strydom, T., Dalla Riva, G. v., & Poisot, T. (2021). SVD Entropy Reveals the High Complexity of Ecological 
        Networks. Frontiers in Ecology and Evolution, 9, 623141. https://doi.org/10.3389/FEVO.2021.623141/BIBTEX
    """
    try:
        order = svd_entropy_order
        delay = svd_entropy_delay
        # Embedding function integrated directly for 1D signals
        signal_length = len(signal)
        if order * delay > signal_length:
            raise ValueError("Error: order * delay should be lower than signal length.")
        if delay < 1:
            raise ValueError("Delay has to be at least 1.")
        if order < 2:
            raise ValueError("Order has to be at least 2.")
        embedding_matrix = np.zeros((order, signal_length - (order - 1) * delay))
        for i in range(order):
            embedding_matrix[i] = signal[(i * delay): (i * delay + embedding_matrix.shape[1])]
        embedded_signal = embedding_matrix.T

        # Singular Value Decomposition
        singular_values = np.linalg.svd(embedded_signal, compute_uv=False)

        # Normalize the singular values
        normalized_singular_values = singular_values / sum(singular_values)

        base = 2
        log_values = np.zeros(normalized_singular_values.shape)
        log_values[normalized_singular_values < 0] = np.nan
        valid = normalized_singular_values > 0
        log_values[valid] = normalized_singular_values[valid] * np.log(normalized_singular_values[valid]) / np.log(
            base)
        svd_entropy_value = -log_values.sum()
    except:
        svd_entropy_value = np.nan
    return svd_entropy_value

# TODO: I think there is a problem with this implementation
@name("permutation_entropy")
def calculate_permutation_entropy(signal, window_size, permutation_entropy_order, permutation_entropy_delay, **kwargs):
    """
    Calculate the permutation entropy of a time series signal.

    Permutation entropy is a measure of complexity or randomness in a time series, based on the order patterns (permutations)
    of values within a sliding window. This method captures the temporal order of the values, making it robust against noise 
    and suitable for detecting changes in the dynamics of nonlinear systems.

    Parameters:
    -----------
    signal : array-like
        The input time series data (1D signal) for which the permutation entropy is to be calculated.

    window_size : int
        The length of the window in which the permutation entropy is calculated. This should be less than or equal to 
        the length of the signal.

    permutation_entropy_order : int
        The embedding dimension (order), which determines the length of the ordinal patterns (permutations) considered.

    permutation_entropy_delay : int or list/array of int
        The time delay between consecutive values used to form the ordinal patterns. If multiple delays are passed as a 
        list or array, the function will return the average permutation entropy across all delays.
    
    Returns:
    --------
    float
        The calculated permutation entropy value. If an error occurs during the 
        calculation, the function returns NaN.

    References:
    -----------
        - Bandt, C., & Pompe, B. (2002). Permutation Entropy: A Natural Complexity Measure for Time Series. 
        https://doi.org/10.1103/PhysRevLett.88.174102
        - Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). Permutation Entropy and Its Main Biomedical 
        and Econophysics Applications: A Review. Entropy 2012, Vol. 14, Pages 1553-1577, 14(8), 1553–1577.
        https://doi.org/10.3390/E14081553
    """
    try:
        order = permutation_entropy_order
        delay = permutation_entropy_delay

        if order * delay > window_size:
            raise ValueError("Error: order * delay should be lower than signal length.")
        if delay < 1:
            raise ValueError("Delay has to be at least 1.")
        if order < 2:
            raise ValueError("Order has to be at least 2.")
        embedding_matrix = np.zeros((order, window_size - (order - 1) * delay))
        for i in range(order):
            embedding_matrix[i] = signal[(i * delay): (i * delay + embedding_matrix.shape[1])]
        embedded_signal = embedding_matrix.T

        # Continue with the permutation entropy calculation. If multiple delay are passed, return the average across all
        if isinstance(delay, (list, np.ndarray, range)):
            return np.mean([calculate_permutation_entropy(signal) for d in delay])

        order_range = range(order)
        hash_multiplier = np.power(order, order_range)

        sorted_indices = embedded_signal.argsort(kind="quicksort")
        hash_values = (np.multiply(sorted_indices, hash_multiplier)).sum(1)
        _, counts = np.unique(hash_values, return_counts=True)
        probabilities = np.true_divide(counts, counts.sum())

        base = 2
        log_values = np.zeros(probabilities.shape)
        log_values[probabilities < 0] = np.nan
        valid = probabilities > 0
        log_values[valid] = probabilities[valid] * np.log(probabilities[valid]) / np.log(base)
        entropy_value = -log_values.sum()
    except:
        entropy_value = np.nan
    return entropy_value

@name("binned_entropy")
def calculate_binned_entropy(signal, entropy_bins, **kwargs):
    """
    Calculate the entropy of a signal based on the distribution of its values across specified bins.
    
    Parameters:
    ----------
    signal : array-like
        The input time series data.
        
    entropy_bins : int or sequence of scalars
        If `entropy_bins` is an integer, it defines the number of equal-width bins in the histogram.
        If `entropy_bins` is a sequence, it defines the bin edges, including the rightmost edge.
    
    Returns:
    -------
    float
        The entropy of the binned values in `signal`. 
        
    Notes:
    ------
    Binned entropy provides a measure of the randomness of values in `signal` over the specified bins.
    Higher entropy indicates a more uniform distribution across bins, while lower entropy indicates 
    that values are concentrated in fewer bins.
        
    References:
    -----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    hist, _ = np.histogram(signal, entropy_bins)
    probs = hist / len(signal)
    probs[probs == 0] = 1.0
    return -np.sum(probs * np.log(probs))

@name("no._of_zero_crossings")
def calculate_zero_crossings(signal, **kwargs):
    """
    Calculates the number of times the signal crosses zero.
    
    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    int
        The number of times (integer) the signal crosses zero
    
    References:
    ----------
        - Myroniv, B., Wu, C.-W., Ren, Y., Christian, A., Bajo, E., & Tseng, 
        Y.-C. (2017). Analyzing User Emotions via Physiology Signals. https://www.researchgate.net/publication/323935725
        - Sharma, G., Umapathy, K., & Krishnan, S. (2020). Trends in audio signal feature extraction methods.
        Applied Acoustics, 158, 107020. https://doi.org/10.1016/J.APACOUST.2019.107020
        - Purushothaman, G., & Vikas, · Raunak. (2018). Identification of a feature selection
        based pattern recognition scheme for finger movement recognition from multichannel EMG
        signals. Australasian Physical & Engineering Sciences in Medicine, 41, 549–559. 
        https://doi.org/10.1007/s13246-018-0646-7
    """
    # Compute the difference in signbit (True if number is negative)
    zero_cross_diff = np.diff(np.signbit(signal))
    # Sum the differences to get the number of zero-crossings
    num_zero_crossings = zero_cross_diff.sum()
    return num_zero_crossings

@name("crest_factor")
def calculate_crest_factor(signal, **kwargs):
    """
    Calculate the crest factor of the given signal.
    
    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The crest factor(float) of the signal.
    
    References:
    ----------
        - Cempel, C. (1980). Diagnostically oriented measures of vibroacoustical processes. 
        Journal of Sound and Vibration, 73(4), 547–561. https://doi.org/10.1016/0022-460X(80)90667-7
        - Wang, X., Zheng, Y., Zhao, Z., & Wang, J. (2015). Bearing fault diagnosis based on statistical 
        locally linear embedding. Sensors (Switzerland), 15(7), 16225–16247. https://doi.org/10.3390/S150716225
    """
    crest_factor = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))
    return crest_factor

@name("clearance_factor")
def calculate_clearance_factor(signal, **kwargs):
    """
    Calculate the clearance factor of the given signal.
    
    The clearance factor is a measure used in signal processing to 
    quantify the peakiness of a signal. It isdefined as the ratio 
    of the maximum absolute value of the signal to the square of 
    the mean square root of the absolute values of the signal.
    
    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        Clearance factor (float) of the signal.
    
    References:
    ----------     
        Formula from The MathWorks Inc., 2022, Available: [Signal Features](https://www.mathworks.com)
        - Wang, X., Zheng, Y., Zhao, Z., & Wang, J. (2015). Bearing fault diagnosis based on statistical 
        locally linear embedding. Sensors (Switzerland), 15(7), 16225–16247. https://doi.org/10.3390/S150716225      
    """
    clearance_factor = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal))) ** 2)
    return clearance_factor

@name("shape_factor")
def calculate_shape_factor(signal, **kwargs):
    """
    Calculates the shape factor of the time series.
    
    The shape factor is a measure of the waveform shape of a signal, which is the ratio of the root mean square (RMS) 
    value to the mean absolute value of the signal.

    Parameters:
    ---------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The shape factor (float) of the signal.
        
    References:
    ----------
        - Cempel, C. (1980). Diagnostically oriented measures of vibroacoustical processes. 
        Journal of Sound and Vibration, 73(4), 547–561. https://doi.org/10.1016/0022-460X(80)90667-7
    """
    shape_factor = np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal))
    return shape_factor

@name("no._of_mean_crossings")
def calculate_mean_crossing(signal, **kwargs):
    """
    Calculate the number of mean crossings in a time series signal.

    Mean crossing refers to the number of times the signal crosses its mean value.

    Parameters:
    ---------
    signal : np.array
        The input time series.

    Returns:
    -------
    float
        The shape factor (float) of the signal.
        
    References:
    ----------
        -Myroniv, B., Wu, C.-W., Ren, Y., Christian, A., Bajo, E., & Tseng, 
        Y.-C. (2017). Analyzing User Emotions via Physiology Signals. 
        https://www.researchgate.net/publication/323935725
    """
    mean_val = np.mean(signal)
    return np.sum(np.diff(signal > mean_val))

@name("impulse_factor")
def calculate_impulse_factor(signal, **kwargs):
    """
    Calculate the impulse factor of a time series signal.

    The impulse factor is a measure of the peakiness or impulsiveness of a signal. It is defined as the ratio of the maximum 
    absolute value of the signal to the mean absolute value. This metric is useful in signal processing and vibration analysis 
    for identifying sharp peaks or transients within the signal, which could indicate faults or other significant events.

    Parameters:
    -----------
    signal : array-like
        The input time series.

    Returns:
    --------
    float
        The impulse factor (float) of the signal. Higher values indicate a signal with sharp 
        peaks relative to its average level, while lower values suggest a more uniform signal.

    References:
    -----------
        - Cempel, C. (1980). Diagnostically oriented measures of vibroacoustical processes. 
        Journal of Sound and Vibration, 73(4), 547–561. https://doi.org/10.1016/0022-460X(80)90667-7
    """
    impulse_factor = np.max(np.abs(signal)) / np.mean(np.abs(signal))
    return impulse_factor

@name("mean_of_auto_corr_lags")
def calculate_mean_auto_correlation(signal, n_lags_auto_correlation, **kwargs):
    """
    Calculate the mean of the auto-correlation values of a time series signal.

    Auto-correlation measures the similarity between a signal and a time-shifted version of itself, providing insight into the 
    signal's repeating patterns and periodicity. The mean auto-correlation value represents the average correlation of the signal 
    with its lagged versions, excluding the zero-lag correlation (which is always 1).

    Parameters:
    -----------
    signal : array-like
        The input time series.

    n_lags_auto_correlation : int
        The number of lags to include in the auto-correlation calculation.

    Returns:
    --------
    np.ndarray
        The mean auto-correlation value (float) of the signal.

    References:
    ----------
        - Fulcher, B. D. (2017). Feature-based time-series analysis. Feature Engineering for Machine 
        Learning and Data Analytics, 87–116. https://doi.org/10.1201/9781315181080-4
        - Banos, O., Damas, M., Pomares, H., Prieto, A., & Rojas, I. (2012). Daily living activity 
        recognition based on statistical feature quality group selection. Expert Systems with Applications, 
        39(9), 8013–8021. https://doi.org/10.1016/J.ESWA.2012.01.164
    
    """
    auto_correlation_values = acf(signal, nlags=n_lags_auto_correlation)[1:]
    return np.mean(auto_correlation_values)

@name("moment_order_{}", "moment_orders")
def calculate_higher_order_moments(signal, moment_orders, **kwargs):
    """
    Calculates the higher order moments of the given time series.

    Parameters:
    ---------
    signal : np.array 
        The input time series.

    moment_orders : array-like
        A list or array of integers specifying the orders of the moments to be calculated. For example, [3, 4] will calculate 
        the 3rd and 4th moments (skewness and kurtosis).
        
    Returns:
    -------
    np.array
        An array containing the higher order moments of the signal.
        
    References:
    ---------
        - de Clerk, L., & Savel’ev, S. (2022). An investigation of higher order moments of empirical financial 
        data and their implications to risk. Heliyon, 8(2), e08833. https://doi.org/10.1016/J.HELIYON.2022.E08833
    """
    feats = []
    for order in moment_orders:
        feats.append(moment(signal, moment=order))
    return np.array(feats)

@name("spectral_coefficient_of_variation")
def calculate_coefficient_of_variation(signal, **kwargs):
    """
    Calculate the coefficient of variation (CV) of a time series signal.

    The coefficient of variation is a standardized measure of the dispersion of the data points in a signal. It is defined as 
    the ratio of the standard deviation to the mean, providing a relative measure of variability regardless of the signal's 
    scale. This metric is particularly useful in comparing the degree of variation between signals with different units or 
    widely different means.
    
    Parameters:
    ---------
    signal : np.array
        The input time series.

    Returns:
    -------
    np.array
        The coefficient of variation (float) of the signal. 
        
    References:
    ---------
        - Jalilibal, Z., Amiri, A., Castagliola, P., & Khoo, M. B. C. (2021). Monitoring the 
        coefficient of variation: A literature review. Computers & Industrial Engineering, 161,
        107600. https://doi.org/10.1016/J.CIE.2021.107600
    
    """
    coefficient_of_variation = np.std(signal) / np.mean(signal)
    return coefficient_of_variation

@name("mean_abs_deviation")
def calculate_median_absolute_deviation(signal, adjusted, **kwargs):
    """
    Calculate the Median Absolute Deviation (MAD) of a time series signal.

    The Median Absolute Deviation (MAD) is a robust measure of statistical dispersion. Unlike the standard deviation, 
    which is influenced by outliers, MAD provides a more resilient measure of variability by focusing on the median rather 
    than the mean.
    
    Parameters:
    -----------
    signal : array-like
        The input time series data.
    adjusted : bool, optional (default=False)
        If True, returns the adjusted MAD, making it consistent with the standard deviation
        for normally distributed data by multiplying by 1.4826.

    Returns:
    --------
    float
        The MAD (float) of the signal. This value represents the median of the absolute 
        deviations from the median of the signal.

    References:
    -----------
        - Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute Deviation.
        Journal of the American Statistical Association, 88(424), 1273–1283. https://doi.org/10.2307/2291267
        - Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard
        deviation around the mean, use absolute deviation around the median. Journal of Experimental Social Psychology, 
        49(4), 764–766. https://doi.org/10.1016/J.JESP.2013.03.013
        - Pham-Gia, T., & Hung, T. L. (2001). The mean and median absolute deviations. Mathematical and Computer Modelling,
        34(7–8), 921–936. https://doi.org/10.1016/S0895-7177(01)00109-1
    """
    median_value = np.median(signal)
    mad = np.median(np.abs(signal - median_value))

    if adjusted:
        mad *= 1.4826
    return mad

# def calculate_signal_magnitude_area(self, signal, **kwargs):
#         # Formula from Khan et al., 2010, DOI: 10.1109/TITB.2010.2051955
#         # Formula from Rafiuddin et al., 2011, DOI: 10.1109/MSPCT.2011.6150470
#         return np.array([np.sum(np.abs(signal))])

@name("avg_amplitude_change")
def calculate_avg_amplitude_change(signal, **kwargs):
    """
    Calculates the average wavelength of the time series.
    
    Parameters:
    ----------
    signal : array-like
        The input time series data.

    Returns:
    -------
    float
        The average amplitude change (float) of the signal.
    
    References:
    ----------
        - Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. 
        Expert Systems with Applications, 39(8), 7420–7431. https://doi.org/10.1016/J.ESWA.2012.01.102
    """
    avg_amplitude_change = np.mean(np.abs(np.diff(signal)))
    return avg_amplitude_change

@name("no._of_slope_sign_changes")
def calculate_slope_sign_change(signal, ssc_threshold, **kwargs):
    """
    Calculate the Slope Sign Change (SSC) of the signal, considering a threshold.

    Parameters:
    -----------
    signal : array-like
        The input time series signal.
    ssc_threshold : float, optional
        The threshold value to determine significant slope changes (default is 0).

    Returns:
    --------
    float
        The SSC value.
        
    References:
    ----------
        - Purushothaman, G., & Vikas, · Raunak. (2018). Identification of a feature selection
        based pattern recognition scheme for finger movement recognition from multichannel EMG
        signals. Australasian Physical & Engineering Sciences in Medicine, 41, 549–559. 
        https://doi.org/10.1007/s13246-018-0646-7
    """
    # Calculate the first and second differences
    diff1 = np.diff(signal[:-1])
    diff2 = np.diff(signal[1:])

    # Compute the product of the differences
    product = diff1 * diff2

    # Apply the threshold and count the number of valid slope sign changes
    slope_sign_change = np.sum(product >= ssc_threshold)
    return slope_sign_change

@name("higuchi_fractal_dimensions_k={}", "higuchi_k_values")
def calculate_higuchi_fractal_dimensions(signal, higuchi_k_values, **kwargs):
    """
    Calculates the Higuchi Fractal Dimension for a given time series signal.
    
    Parameters:
    -----------
    signal : array-like
        The input time series data.
        
    Returns:
    --------
    np.array
        Array containing the Higuchi Fractal Dimension for each `k` in higuchi_k_values.
        
    References:
    ----------
        - Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory. 
        Physica D: Nonlinear Phenomena, 31(2), 277–283. https://doi.org/10.1016/0167-2789(88)90081-4
        - Wijayanto, I., Hartanto, R., & Nugroho, H. A. (2019). Higuchi and Katz Fractal Dimension for
        Detecting Interictal and Ictal State in Electroencephalogram Signal. 2019 11th International
        Conference on Information Technology and Electrical Engineering, ICITEE 2019. 
        https://doi.org/10.1109/ICITEED.2019.8929940
        - Wanliss, J. A., & Wanliss, G. E. (2022). Efficient calculation of fractal properties via
        the Higuchi method. Nonlinear Dynamics, 109(4), 2893–2904. 
        https://doi.org/10.1007/S11071-022-07353-2/FIGURES/8
    """

    def compute_length_for_interval(data, interval, start_time):
        data_size = data.size
        num_intervals = (data_size - start_time) // interval
        normalization_factor = (data_size - 1) / (num_intervals * interval)
        sum_difference = np.sum(np.abs(np.diff(data[start_time::interval], n=1)))
        length_for_time = (sum_difference * normalization_factor) / interval
        return length_for_time

    def compute_average_length(data, interval):
        lengths = [
            compute_length_for_interval(data, interval, start_time)
            for start_time in range(1, interval + 1)
        ]
        return np.mean(lengths)

    feats = []
    for max_interval in higuchi_k_values:
        try:
            average_lengths = np.array([
                compute_average_length(signal, interval)
                for interval in range(1, max_interval + 1)
            ])
            interval_range = np.arange(1, max_interval + 1)
            fractal_dimension, _ = -np.polyfit(np.log(interval_range), np.log(average_lengths), 1)
        except Exception as e:
            print(f"Error computing HFD for k={max_interval}: {e}")
            fractal_dimension = np.nan
        feats.append(fractal_dimension)

    return np.array(feats)

@name("katz_fractal_dimension")
def calculate_katz_fractal_dimension(signal, **kwargs):
    """
    Calculate the Katz Fractal Dimension (KFD) of a given signal.

    The Katz Fractal Dimension is a measure of the complexity of a signal, which takes into 
    account the total length of the signal and the maximum distance between the first point 
    and any other point in the signal.

    Parameters:
    -----------
    signal : array-like
        The input time series

    Returns:
    --------
    float
        The Katz Fractal Dimension of the signal.
        
    References:
    ----------
        - Li, Y. ;, Zhou, Y. ;, Jiao, S., Li, Y., Zhou, Y., & Jiao, S. (2023). Variable-Step Multiscale Katz Fractal Dimension: 
        A New Nonlinear Dynamic Metric for Ship-Radiated Noise Analysis. Fractal and Fractional 2024, Vol. 8, Page 9, 8(1), 9. 
        https://doi.org/10.3390/FRACTALFRACT8010009
    """
    N = len(signal)
    distance = np.max(np.abs(signal - signal[0]))
    length = np.sum(np.abs(np.diff(signal)))
    return np.log10(N) / (np.log10(N) + np.log10(distance / length))

@name("petrosian_fractal_dimension")
def calculate_petrosian_fractal_dimension(signal, **kwargs):
    """
    Calculate the Petrosian Fractal Dimension (PFD) of a given signal.

    The Petrosian Fractal Dimension is a measure used to quantify the complexity of a signal, 
    often used in analyzing time series data. It is particularly useful for distinguishing 
    between noise and structural patterns in the signal. The PFD is computed based on the number 
    of zero-crossings in the signal's first derivative.

    Parameters:
    -----------
    signal : array-like
        The input time series

    Returns:
    --------
    float
        The Petrosian Fractal Dimension of the signal.

    References:
    -----------
        - Petrosian, A. (1995). Kolmogorov complexity of finite sequences and recognition of different
        preictal EEG patterns. Proceedings of the IEEE Symposium on Computer-Based Medical Systems, 212–217. 
        https://doi.org/10.1109/CBMS.1995.465426
        - Alfredo, M. L., Marta, M., Alfredo, M. L., & Marta, M. (2020). Classification of low-density EEG for 
        epileptic seizures by energy and fractal features based on EMD. The Journal of Biomedical Research, 
        2020, Vol. 34,  Issue 3, Pages: 180-190, 34(3), 180–190. https://doi.org/10.7555/JBR.33.20190009
    """
    N = len(signal)
    nzc = calculate_zero_crossings(np.diff(signal))
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nzc)))

@name(["hjorth_mobility", "hjorth_complexity"])
def calculate_hjorth_mobility_and_complexity(signal, **kwargs):
    """
    Calculates mobility and complexity of the time series which are based on 
    the first and second derivatives of the time series.

    Parameters:
    ---------
    signal : array-like
        The input time series data.

    Returns:
    -------
    np.array
        Array containing the mobility and complexity of the time series.
        
    References:
    ---------
        - Hjorth, B. (1970). EEG analysis based on time domain properties. Electroencephalography
        and Clinical Neurophysiology, 29(3), 306–310. https://doi.org/10.1016/0013-4694(70)90143-4
    """
    try:
        signal = np.asarray(signal)
        # Calculate derivatives
        first_derivative = np.diff(signal)
        second_derivative = np.diff(first_derivative)
        # Calculate variance
        signal_variance = np.var(signal)  # = activity
        first_derivative_variance = np.var(first_derivative)
        second_derivative_variance = np.var(second_derivative)
        # Mobility and complexity
        mobility = np.sqrt(first_derivative_variance / signal_variance)
        complexity = np.sqrt(second_derivative_variance / first_derivative_variance) / mobility
    except:
        mobility = np.nan
        complexity = np.nan
    return np.array([mobility, complexity])

@name("cardinality")
def calculate_cardinality(signal, window_size, **kwargs):
    # Parameter
    thresh = 0.05 * np.std(signal)  # threshold
    # Sort data
    sorted_values = np.sort(signal)
    cardinality_array = np.zeros(window_size - 1)
    for i in range(window_size - 1):
        cardinality_array[i] = np.abs(sorted_values[i] - sorted_values[i + 1]) > thresh
    cardinality = np.sum(cardinality_array)
    return cardinality

@name("rms_to_mean_of_abs")
def calculate_rms_to_mean_abs(signal, **kwargs):
    """
    Calculates the ratio of the root-mean-squared value to the mean
    absolute value.

    Parameters:
    ---------
    signal : array-like
        The input time series data.

    Returns:
    -------
    float
        The ratio of the root-mean-squared value to the mean
        absolute value.
    """
    rms_val = np.sqrt(np.mean(signal ** 2))
    mean_abs_val = np.mean(np.abs(signal))
    ratio = rms_val / mean_abs_val
    return ratio

@name("area_under_curve")
def calculate_area_under_curve(signal, **kwargs):
    """
    Calculates the area under the curve of the given signal

    Parameters:
    ----------
    signal : array-like
        The input time series data.

    Returns:
    -------
    float
        Area under curve.
    
    References:
    ---------
        - Kuremoto, T., Baba, Y., Obayashi, M., Mabu, S., & Kobayashi, K. 
        (2018). Enhancing EEG Signals Recognition Using ROC Curve. Journal 
        of Robotics, Networking and Artificial Life, 4(4), 283. https://doi.org/10.2991/JRNAL.2018.4.4.5
    """
    return simpson(np.abs(signal), dx=1)

@name("area_under_squared_curve")
def calculate_area_under_squared_curve(signal, **kwargs):
    """
    Calculates the area under the curve of the given signal squared

    Parameters:
    ----------
    signal : array-like
        The input time series data.

    Returns:
    -------
    float
        Area under curve of signal squared.
    """
    return simpson(signal**2, dx=1)

@name("autoregressive_model_coefficients_{}", "ar_model_coefficients_order", 1)
def calculate_autoregressive_model_coefficients(signal, ar_model_coefficients_order, **kwargs):
    """
    Calculates the autoregressive model coefficients of the time series

    Parameters:
    ---------
    signal : array-like
        The input time series data.
    ar_model_coefficients_order : (int, optional)
        The number of lags to include in the model, defaults to 4.

    Returns:
    -------
    numpy.ndarray
        An array containing coefficients 
        
    References:
    ---------
        - Khan, A. M., Lee, Y. K., & Kim, T. S. (2008). Accelerometer signal-based human 
        activity recognition using augmented autoregressive model coefficients and artificial 
        neural nets. Proceedings of the 30th Annual International Conference of the IEEE Engineering 
        in Medicine and Biology Society, EMBS’08 - “Personalized Healthcare through Technology,” 5172–5175. 
        https://doi.org/10.1109/IEMBS.2008.4650379
    """
    model = AutoReg(signal, lags=ar_model_coefficients_order)
    model_fitted = model.fit()
    return model_fitted.params

@name("count")
def calculate_count(signal, **kwargs):
    """
    Calculates the length of the time series.

    Parameters:
    ---------
    signal : array-like
        The input time series data.

    Returns:
    --------
    int
        The length of the time series
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return len(signal)

@name("count_above_mean")
def calculate_count_above_mean(signal, **kwargs):
    """
    Calculate the number of data points in the signal that are above the mean.

    This function computes the mean of the signal and returns the count of elements
    in the signal that are strictly greater than the mean value.
    
    Parameters:
    ----------
    signal : array-like
        The input time series.
    
    Returns:
    -------
    int
        The number of data points in the signal that are above the mean.


    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    mean_val = np.mean(signal)
    return np.sum(signal > mean_val)

@name("count_below_mean")
def calculate_count_below_mean(signal, **kwargs):
    """
    Calculate the number of data points in the signal that are below the mean.

    This function computes the mean of the signal and returns the count of elements
    in the signal that are strictly less than the mean value.

    Parameters:
    ----------
    signal : array-like
        The input time series.
        
    Returns:
    -------
    int
        The number of data points in the signal that are below the mean.
        
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    mean_val = np.mean(signal)
    return np.sum(signal < mean_val)

@name("count_below")
def calculate_count_below(signal, count_below_or_above_x, **kwargs):
    """
    Calculate the percentage of signal that is lower than the specified value. 
    
    Parameters:
    -----------
    signal : array-like
        The input time series data.
    count_below_or_above_x : int
        Value of interest.
        
    Returns:
    -------
    int
        number of values lower than x
        
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.sum(signal <= count_below_or_above_x) / len(signal)

@name("count_above")
def calculate_count_above(signal, count_below_or_above_x, **kwargs):
    """
    Calculate the percentage of signal that is above than the specified value
    
    Parameters:
    -----------
    signal : array-like
        The input time series data.
    count_below_or_above_x : int
        Value of interest
        
    Returns:
    -------
    int
        number of values higher than x
        
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.sum(signal >= count_below_or_above_x) / len(signal)

@name("value_count_{}", "values")
def calculate_value_count(signal, values, **kwargs):
    """
    Calculate the count of occurrences of a specified value in a signal.

    Parameters:
    ----------
    signal : array-like
        The input time series data.
    
    values :  list of float or int or NaN
        The value to count within the signal. If `value` is NaN, the function will
        count the number of NaN entries in `signal`.

    Returns:
    -------
    lists of int
        The counts of occurrences of `value` in `signal`. If `value` is NaN, it
        returns the count of NaN values.
        
    References:
    -----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    signal = np.asarray(signal)

    feats = []
    for value in values:
        if np.isnan(value):
            return np.isnan(signal).sum()
        else:
            feats.append(signal[signal == value].size)
    return feats

# Also not suitable for calculate all
@exclude()
@name("covariance")
def calculate_covariance(signal, other_signal, **kwargs):
    """
    Calculate the covariance of one signal with another signal.

    Parameters:
    -----------
    signal : array-like
        The input time series data.
    other_signal : array-like
        A second time series to compute the covariance with.
        
    Returns:
    -------
    int
        number of values higher than x

    References:
    ----------
        https://support.ptc.com/help/mathcad/r9.0/en/index.html#page/PTC_Mathcad_Help/covariance.html
    """
    return np.cov(signal, other_signal)[0, 1]

# TODO: comments
@name("cumulative_sum")
def calculate_cumulative_sum(signal, **kwargs):
    """
    Calculate the cumulative sum of a signal.
    
    Parameters:
    ----------
    signal: array-like
        The input time series data

    Returns:
    -------
    cumulative_sum: float
        The last value of the cumulative sum of the input signal.
        
    References:
    ----------
        https://docs.amd.com/r/2020.2-English/ug1483-model-composer-sys-gen-user-guide/Cumulative-Sum
    """
    return np.cumsum(signal)[-1]

@name("energy_ratio_by_chunks_{}", "energy_ratio_chunks")
def calculate_energy_ratio_by_chunks(signal, energy_ratio_chunks, **kwargs):
    """
    Calculate the energy ratio of a signal by dividing it into chunks.

    The energy ratio by chunks is calculated by dividing the signal into a specified number of chunks. The energy of each chunk 
    is computed as the sum of the squared values of the signal within that chunk. The ratio of the energy of each chunk to the 
    total energy of the signal is then returned.

    This feature can be useful in identifying where the energy of a signal is concentrated over different segments of the signal, 
    which can help in analyzing periodicity, transient events, or the distribution of power in the signal over time.

    Parameters:
    -----------
    signal : array-like
        The input time series signal.
    energy_ratio_chunks : int, optional
        The number of chunks to divide the signal into (default is 4).

    Returns:
    --------
    float
        An array containing the energy ratio of each chunk relative to the total energy of the signal.

    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    chunk_size = len(signal) // energy_ratio_chunks
    energies = np.array([np.sum(signal[i*chunk_size:(i+1)*chunk_size]**2) for i in range(energy_ratio_chunks)])
    total_energy = np.sum(signal**2)
    return energies / total_energy

@name("moving_average")
def calculate_moving_average(signal, window_size, mode, **kwargs):
    """
    Returns the moving average of a time series signal.

    The moving average is a common technique used to smooth out short-term fluctuations and highlight longer-term trends 
    or cycles in a time series. This implementation applies a simple moving average filter to the input signal.

    Parameters:
    -----------
    signal : array-like
        The input time series signal.
    window_size : int, optional
        The size of the moving window used to calculate the average. Default is 10.
    mode : str, optional
        The mode parameter determines the type of padding applied to the input signal. 
        Options include:
            - 'valid': Returns output of length max(M, N) - min(M, N) + 1. This means no padding is applied and 
            the result is only calculated where the window fully overlaps with the signal.
            - 'same': Returns output of the same length as the input signal. Padding may be applied to ensure the output 
            has the same length as the input.
            - 'full': Returns the convolution at each point of overlap, with padding. The output length is M + N - 1.
        Default is 'valid'.

    Returns:
    -------
    np.ndarray or float
        An array containing the moving average of the signal. If the signal length is shorter than the window size, 
        NaN is returned.

    References:
    ----------
    SPTK Moving Average Filter:
        https://cyclostationary.blog/2021/05/23/sptk-the-moving-average-filter/
    """
    if len(signal) < window_size:
        return np.nan
    return np.convolve(signal, np.ones(window_size) / window_size, mode=mode)

@name("weighted_moving_average")
def calculate_weighted_moving_average(signal, weights, mode, **kwargs):
    """
    Calculate the weighted moving average of a time series signal.

    The weighted moving average assigns different weights to different points within the moving window, allowing some 
    points to contribute more heavily to the average than others. This can be useful for emphasizing more recent 
    data points in the time series.

    Parameters:
    -----------
    signal : array-like
        The input time series signal.
    weights : array-like, optional
        The weights to apply to the signal. If not provided, a linearly decreasing set of weights is used.
        The length of weights must be equal to or less than the length of the signal.
    mode : str, optional
        The mode parameter determines the type of padding applied to the input signal. 
        Options include:
            - 'valid': Returns output of length max(M, N) - min(M, N) + 1. This means no padding is applied and 
            the result is only calculated where the window fully overlaps with the signal.
            - 'same': Returns output of the same length as the input signal. Padding may be applied to ensure the output 
            has the same length as the input.
            - 'full': Returns the convolution at each point of overlap, with padding. The output length is M + N - 1.
        Default is 'valid'.

    Returns:
    --------
    np.ndarray
        The weighted moving average of the signal. If the signal is shorter than the weights, NaN is returned.
        
    References:
    -----------
        - https://www.mathworks.com/help/signal/ug/signal-smoothing.html
    """
    if weights is None:
        weights = np.linspace(1, 0, num=len(signal))
    else:
        weights = np.asarray(weights)
        if len(weights) > len(signal, **kwargs):
            raise ValueError("Length of weights must be less than or equal to the length of the signal.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    # Normalize the weights
    weights = weights / np.sum(weights)

    # If the signal is shorter than the weights, return NaN
    if len(signal) < len(weights):
        return np.nan

    # Apply the weighted moving average
    return np.convolve(signal, weights, mode=mode)

@name("exponential_moving_average")
def calculate_exponential_moving_average(signal, ema_alpha, **kwargs):
    """
    Calculates the exponential moving average of the given signal

    Parameters:
    ---------
    signal : array-like
        The input time series.
    ema_alpha : float 
        Defaults to 0.3.

    Returns:
    -------
    float
        Last value in the array
    """
    ema = np.zeros_like(signal)
    ema[0] = signal[0]
    for i in range(1, len(signal)):
        ema[i] = ema_alpha * signal[i] + (1 - ema_alpha) * ema[i - 1]
    return ema[-1]

@name("first_location_of_maximum")
def calculate_first_location_of_maximum(signal, **kwargs):
    """
    Returns the location of the first maximum value of the time series.
    
    Parameters:
    ---------
    signal : np.array
        The input time series
    
    Returns:
    -------
    float
        The location of the first maximum value of the time series.
        
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.argmax(signal)

@name("first_location_of_minimum")
def calculate_first_location_of_minimum(signal, **kwargs):
    """
    Returns the location of the first minimum value of the time series
    
    Parameters:
    ---------
    signal : array-like
        The input time series
        
    Returns:
    -------
    float
        The location of the first minimum value of the time series.
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.argmin(signal)

@name("first_order_difference")
def calculate_first_order_difference(signal, **kwargs):
    """
    Calculates the first-order difference of a given signal.
    
    Parameters:
    --------
    signal : array-like
        The input time series

    Returns:
    -------
    np.ndarray
        An array containing the first-order difference of the input signal. 
        The length of this array is one less than the original signal.
    
    References:
    ----------
        https://cran.r-project.org/web/packages/doremi/vignettes/first-order.html    
    """
    return np.diff(signal, n=1)

@name("fisher_information")
def calculate_fisher_information(signal, **kwargs):
    # https://www.researchgate.net/publication/311396939_Analysis_of_Signals_by_the_Fisher_Information_Measure
    variance = np.var(signal)
    information = 1 / variance if variance != 0 else float('inf')
    return information

# TODO: Allow integer series for naming or choose to put all in one
@name("histogram_bin_frequencies_{}", "hist_bins")
def calculate_histogram_bin_frequencies(signal, hist_bins, **kwargs):
    """
    Calculate the frequency of values in a signal within specified histogram bins.

    This function generates a histogram of the input signal by dividing the data into
    the specified number of bins (default: 10) or using specified bin edges. It returns the count of
    data points falling within each bin.

    Parameters:
    -----------
    signal : array-like
        The input time series
    
    hist_bins : int
        It defines the number of equal-width bins in the range of the signal.        

    Returns:
    --------
    np.ndarray
        An array of integers where each value represents the number of occurrences of the signal's
        values within each bin.
        
    References:
    ----------
        - Pizzi, N. J., Somorjai, R. L., & Pedrycz, W. (2006). Classifying Biomedical Spectra 
        Using Stochastic Feature Selection and Parallelized Multi-Layer Perceptrons. Modern 
        Information Processing, 383–393. https://doi.org/10.1016/B978-044452075-3/50032-7
    """
    hist, _ = np.histogram(signal, bins=hist_bins)
    return hist

@name("last_location_of_maximum")
def calculate_last_location_of_maximum(signal, **kwargs):
    """
    Returns the last location of the maximum value
    
    Parameters:
    ----------
    signal : array-like
        The input time series
        
    Returns:
    -------
    int
        Last location of the maximum value in the time series
        
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.max(np.where(signal == np.max(signal))[0])

@name("last_location_of_minimum")
def calculate_last_location_of_minimum(signal, **kwargs):
    """
    Returns the last location of the mininum value
    
    Parameters:
    ----------
    signal : array-like
        The input time series
        
    Returns:
    -------
    int
        Last location of the mininum value in the time series
        
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.max(np.where(signal == np.min(signal))[0])

@name(["lin_trend_slope", "lin_trend_intercept", "lin_trend_R2", "lin_trend_p_value", "lin_trend_std_err"])
def calculate_linear_trend_with_full_linear_regression_results(signal, **kwargs):
    """
    Calculate the linear trend of a signal and return the full set of linear regression results.

    This function performs a linear regression on the input signal, where the signal values are 
    the dependent variable and their indices are the independent variable. It returns the slope 
    of the best-fit line, the y-intercept, the coefficient of determination (R²), the p-value for 
    the slope, and the standard error of the estimated slope.

    Parameters:
    -----------
    signal : array-like
        The input time series

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - slope (float): The slope of the best-fit line.
        - intercept (float): The y-intercept of the best-fit line.
        - r_value**2 (float): The coefficient of determination (R²), which indicates the 
        goodness of fit of the model.
        - p_value (float): The p-value for the slope, indicating the significance of the slope.
        - std_err (float): The standard error of the estimated slope.

    Notes:
    ------
    - The function uses `scipy.stats.linregress` to perform the linear regression.
    - The R² value is calculated as the square of the correlation coefficient (`r_value`).
    - A low p-value suggests that the slope is significantly different from zero.

    References:
    -----------
        - "Linear Regression", MATLAB Documentation: https://www.mathworks.com/help/matlab/data_analysis/linear-regression.html
    """
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(signal)), signal)
    return slope, intercept, r_value**2, p_value, std_err

@name(["local_maxima", "local_minima"])
def calculate_local_maxima_and_minima(signal, **kwargs):
    """
    Calculate the number of local maxima and minima in a signal.
    
    Parameters:
    ----------
    signal : array-like
        The input time series
    
    Returns:
    -------
    local_max: int
        The number of local maxima in the signal.
    local_min: int
        The number of local minima in the signal.

    References:
    ----------
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    local_max = find_peaks(signal)[0]
    local_min = find_peaks(-signal)[0]
    return len(local_max), len(local_min)

@name("log_return")
def calculate_log_return(signal, **kwargs):
    """
    Calculate the logarithmic return of a time series efficiently.

    The log return is a commonly used measure in finance that represents the rate of return 
    of an asset over a period of time. It is computed as the natural logarithm of the 
    ratio between the final and initial values of the time series.

    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the time series or asset prices. 
        The signal should contain only positive values for a valid logarithmic return.

    Returns:
    --------
    float
        The logarithmic return of the time series. If the initial or final value is non-positive, 
        the function returns `NaN` to indicate an undefined log return.

    Notes:
    ------
    - This function assumes that the signal is clean and contains no non-positive values. 
    If non-positive values are present, the function will return `NaN`.
    
    References:
    ----------
        - Based on concepts from: https://pypi.org/project/stockstats/
    """    
    # Check if the initial or final value is non-positive
    if signal[0] <= 0 or signal[-1] <= 0:
        return np.nan
    
    # Calculate and return the log 
    log_return = np.log(signal[-1] / signal[0])
    return log_return

@name("longest_strike_above_mean")
def calculate_longest_strike_above_mean(signal, **kwargs):
    """
    Calculate the longest sequence (strike) of consecutive values in the time series that are above the mean.

    This function computes the mean of the given signal and identifies the longest continuous sequence of 
    values that are greater than the mean. This "strike" or sequence length is returned as the result.

    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    --------
    int
        The length of the longest sequence of consecutive values in the time series that are above the mean value.

    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on the basis 
        of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77.
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    mean_val = np.mean(signal)
    strike = max([sum(1 for i in g) for k, g in groupby(signal > mean_val) if k])
    return strike

@name("longest_strike_below_mean")
def calculate_longest_strike_below_mean(signal, **kwargs):
    """
    Calculate the longest sequence (strike) of consecutive values in the time series that are below the mean.

    This function computes the mean of the given signal and identifies the longest continuous sequence of 
    values that are less than the mean. This "strike" or sequence length is returned as the result.

    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    --------
    int
        The length of the longest sequence of consecutive values in the time series that are below the mean value.

    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on the basis 
        of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77.
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    mean_val = np.mean(signal)
    strike = max([sum(1 for i in g) for k, g in groupby(signal < mean_val) if k])
    return strike

@name("lower_complete_moment")
def calculate_lower_complete_moment(signal, order=2, **kwargs):
    mean_val = np.mean(signal)
    moment = np.mean([(x - mean_val)**order for x in signal if x < mean_val])
    return moment

@name("mean_absolute_change")
def calculate_mean_absolute_change(signal, **kwargs):
    """
    Calculate the mean absolute change of a signal.

    The mean absolute change is a measure of the average amount by which the 
    values of the signal change from one point to the next. It is calculated 
    as the mean of the absolute differences between consecutive elements in 
    the signal.

    Parameters:
    ----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    -------
    float
        The mean absolute change of the signal.

    References:
    ----------
        - Mean absolute difference: https://en.wikipedia.org/wiki/Mean_absolute_difference
    """
    return np.mean(np.abs(np.diff(signal)))

@name("sum_of_absolute_changes")
def calculate_sum_of_absolute_changes(signal, **kwargs):
    """
    Calculates the sum of absolute differences of the time series
    
    Parameters:
    ----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    -------
    float
        The mean absolute change of the signal.
        
    References:
    ---------
        - Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, T., & Gamboa, H. (2020). 
        TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. https://doi.org/10.1016/j.softx.2020.100456
    """
    return np.sum(np.abs(np.diff(signal)))

@name("mean_relative_change")
def calculate_mean_relative_change(signal, **kwargs):
    """
    Calculates the mean relative change of the time series which is calculated 
    as the absolute difference between consecutive indices divided by their mean value.
    
    Parameters:
    ----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    -------
    float
        The mean relative change of the signal.
        
    References:
    ----------
        - https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.1365-2745.2007.01281.x
    """
    epsilon = 1e-10  # Small constant to avoid division by zero
    
    if len(signal) < 2: return np.nan
    
    denominator = signal[:-1].copy()
    
    # Replace zeros and very small values with eps
    mask = np.abs(denominator) < epsilon
    denominator[mask] = epsilon * np.sign(denominator[mask])
    denominator[denominator == 0] = epsilon 
    
    diffs = np.diff(signal)
    ratios = np.abs(diffs / denominator)
    
    # Remove infinite or nan values before taking mean
    valid_ratios = ratios[np.isfinite(ratios)]
    
    if len(valid_ratios) == 0: return np.nan
        
    return np.mean(valid_ratios)

@name("mean_second_derivative_central")
def calculate_mean_second_derivative_central(signal, **kwargs):
    """
    Calculate the mean value of the central approximation of the second derivative of a time series.
    
    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    --------
    float
        The mean value of the central approximations of the second derivative. 
        If the input signal has fewer than three elements, the function returns `NaN`.
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.mean(np.diff(signal, n=2)) / 2 if len(signal) > 2 else np.NaN

@name("median_second_derivative_central")
def calculate_median_second_derivative_central(signal, **kwargs):
    """
    Calculate the median of the central second derivative of a given signal.

    The second derivative is computed using the central difference method, 
    which approximates the second derivative by taking the difference of 
    differences between consecutive data points. The median of these 
    second derivative values is then returned.

    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    --------
    float
        The median of the second derivative values of the input signal.
    """
    second_derivative = np.diff(signal, n=2)
    return np.median(second_derivative)

@name("mode")
def calculate_mode(signal, **kwargs):
    """
    Calculates the mode of the time series

    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    --------
    float
        The mode of the input signal.
    """
    return mode(signal, keepdims=False)[0]

@name("number_of_inflection_points")
def calculate_number_of_inflection_points(signal, **kwargs):
    """
    Calculate the number of inflection points in a signal.

    An inflection point is where the second derivative of the signal changes sign,
    indicating a change in the direction of curvature.

    Parameters:
    -----------
    signal : np.array 
        Input signal as a 1D array or list.

    Returns:
    --------
    int
        The number of inflection points in the signal.
            
    References:
    ---------
        - https://en.wikipedia.org/wiki/Inflection_point
    """
    second_derivative = np.diff(signal, n=2)
    inflection_points = np.sum(np.diff(np.sign(second_derivative)) != 0)

    return inflection_points

@name("peak_to_peak_distance")
def calculate_peak_to_peak_distance(signal, **kwargs):
    """
    Calculates the peak-to-peak distance of a signal.
    The peak-to-peak distance is the difference between the maximum and minimum values of the signal.

    Parameters:
    -----------
    signal : np.array 
        Input signal as a 1D array or list.

    Returns:
    --------
    float
        The peak-to-peak distance of the signal.

    References:
    ----------
        - Robert Lobbia (2024). Peak to Peak of signal 
        (https://www.mathworks.com/matlabcentral/fileexchange/20314-peak-to-peak-of-signal), 
        MATLAB Central File Exchange. Retrieved September 5, 2024.

    """
    return np.ptp(signal)

@name("percentage_of_negative_values")
def calculate_percentage_of_negative_values(signal, **kwargs):
    """
    Calculate the percentage of values in the time series that are negative
    
    Parameters:
    -----------
    signal : np.array 
        Input signal as a 1D array or list.

    Returns:
    --------
    float
        Percentage of negative values
            
    References:
    ---------
    
    """
    return np.mean(signal < 0) * 100

@name("percentage_of_positive_values")
def calculate_percentage_of_positive_values(signal, **kwargs):
    """
    Calculate the percentage of values in the time series that are positive
    
    Parameters:
    -----------
    signal : np.array
        Input signal as a 1D array or list.

    Returns:
    --------
    float
        Percentage of positive values
            
    References:
    ---------
    
    """
    return np.mean(signal > 0) * 100

@name("percentage_of_reoccurring_datapoints_to_all_datapoints")
def calculate_percentage_of_reoccurring_datapoints_to_all_datapoints(signal, **kwargs):
    """
    Calculates the percentage of non-unique values in the time series.
    
    Parameters:
    -----------
    signal : array-like
        The input time series is to be calculated.

    Returns:
    --------
    float
        Percentage of reoccurring values 
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    unique, counts = np.unique(signal, return_counts=True)
    return 100 * np.sum(counts[counts > 1]) / len(signal)

@name("percentage_of_reoccurring_values_to_all_values")
def calculate_percentage_of_reoccurring_values_to_all_values(signal, **kwargs):
    """
    Calculates the percentage of values that occur more than once in the signal
    
    Parameters:
    -----------
    signal : array-like
        The input time series is to be calculated.

    Returns:
    --------
    float
        Percentage of reoccurring values 
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    unique, counts = np.unique(signal, return_counts=True)

    if counts.shape[0] == 0:
        return 0

    return 100 * np.sum(counts > 1) / float(counts.shape[0])

@name("ratio_beyond_r_sigma_{}", "r_sigma")
def calculate_ratio_beyond_r_sigma(signal, r_sigma, **kwargs):
    """
    Calculates the ratio of data points in the signal that are beyond 'r' times the standard deviation from the mean.

    Parameters:
    -----------
    signal : np.array
        The input signal data.
    r_sigma : array-like
        The multiplier for the standard deviation to define the threshold.

    Returns:
    --------
    float
        The ratio of data points in the signal that are beyond 'r' standard deviations from the mean.

    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on
        the basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77.
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    results = []
    for multiplier in r_sigma:
        std_dev = np.std(signal)
        mean_val = np.mean(signal)
        results.append(np.sum(np.abs(signal - mean_val) > multiplier * std_dev) / len(signal))
    return results

@name(["ratio_positive", "ratio_negative", "ratio_positive_to_negative"])
def calculate_ratio_of_fluctuations(signal, **kwargs):
    """
    Calculates the ratio of positive to negative fluctuations in the signal.

    Parameters:
    ----------
    signal : np.array
        The input signal data.

    Returns:
    -------
    tuple: A tuple containing:
        - ratio_positive (float): The ratio of positive fluctuations.
        - ratio_negative (float): The ratio of negative fluctuations.
        - ratio_positive_to_negative (float): The ratio of positive to negative fluctuations. 
        Returns 'inf' if there are no negative fluctuations.
    """
    increases = np.sum(np.diff(signal) > 0)
    decreases = np.sum(np.diff(signal) < 0)
    total = increases + decreases
    ratio_positive = increases / total if total != 0 else 0
    ratio_negative = decreases / total if total != 0 else 0
    return ratio_positive, ratio_negative, ratio_positive / ratio_negative if ratio_negative != 0 else float('inf')

@name("ratio_value_number_to_sequence_length")
def calculate_ratio_value_number_to_sequence_length(signal, **kwargs):
    """
    Calculates the ratio of the number of unique values in the signal to its length.

    Parameters:
    ----------
    signal : np.array
        The input signal data.

    Returns:
    -------
    float
        The ratio of unique values in the signal to the total number of data points in the signal.
    """
    unique_values = len(np.unique(signal))
    return unique_values / len(signal)

@name("second_order_difference")
def calculate_second_order_difference(signal, **kwargs):
    """
    Calculates the second-order difference of the signal.

    Parameters:
    ----------
    signal : array-like
        The input signal data.

    Returns:
    -------
    np.array
        The second-order difference of the signal.
    
    References:
    ----------
        Intuition of this feature:
            https://stats.stackexchange.com/questions/351697/what-is-the-intuition-behind-second-order-differencing
    """
    return np.diff(signal, n=2)

@name("signal_resultant")
def calculate_signal_resultant(signal, **kwargs):
    """
    Calculates the resultant magnitude of the signal vector.

    Parameters:
    ----------
    signal : np.array
        The input signal data.

    Returns:
    --------
    float
        The square root of the sum of squares of the signal values.
    """
    return np.sqrt(np.sum(signal**2))

@name("signal_to_noise_ratio")
def calculate_signal_to_noise_ratio(signal, **kwargs):
    """
    Calculates the Signal-to-Noise Ratio (SNR) of the signal.

    Parameters:
    ----------
    signal : array-like 
        The input signal data.

    Returns:
    -------
    float
        The Signal-to-Noise Ratio (SNR) of the signal. 
        Returns 'inf' if the noise standard deviation is zero.
    
    References:
    -----------
        - https://en.wikipedia.org/wiki/Signal-to-noise_ratio
        - Johnson, D. H. (2006). Signal-to-Noise Ratio. Scholarpedia, 1(12), 2088.
        https://doi.org/10.4249/SCHOLARPEDIA.2088
    """
    mean_signal = np.mean(signal)
    std_noise = np.std(signal)
    return mean_signal / std_noise if std_noise > 0 else float('inf')

@name("smoothing_by_binomial_filter")
def calculate_smoothing_by_binomial_filter(signal, **kwargs):
    """
    Applies binomial smoothing to the signal using a simple [1, 2, 1] kernel.

    Parameters:
    -----------
    signal : array-like
        The input signal data.

    Returns:
    --------
    np.array
        The smoothed signal.
    
    References:
    ---------
        - https://www.wavemetrics.com/products/igorpro/dataanalysis/signalprocessing/smoothing
    """
    kernel = np.array([1, 2, 1]) / 4.0
    return convolve(signal, kernel, mode='reflect')

@name("stochastic_oscillator_value")
def calculate_stochastic_oscillator_value(signal, **kwargs):
    """
    Calculates the stochastic oscillator value, a measure used in technical analysis of financial markets.

    Parameters:
    ----------
    signal : array-like
        The input signal data.

    Returns:
    --------
    float
        The stochastic oscillator value, calculated as:
        100 * (current_value - low_min) / (high_max - low_min).

    References:
    ----------
        - https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """
    low_min = np.min(signal)
    high_max = np.max(signal)
    current_value = signal[-1]
    return 100 * (current_value - low_min) / (high_max - low_min)

@name("total_sum")
def calculate_sum(signal, **kwargs):
    """
    Calculates the sum of all values in the time series.

    Parameters:
    ----------
    signal : array-like
        The input time series

    Returns:
    --------
    float
        The sum of all values in the time series.
        
    References:
    ---------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    return np.sum(signal)

@name("sum_of_negative_values")
def calculate_sum_of_negative_values(signal, **kwargs):
    """
    Calculates the sum of all negative values in the time series.
    
    Parameters:
    ----------
    signal : array-like
        The input time series
        
    Returns:
    -------
    float
        The sum of all negative values in the time series
    """
    return np.sum(signal[signal < 0])

@name("sum_of_positive_values")
def calculate_sum_of_positive_values(signal, **kwargs):
    """
    Calculates the sum of all positive values in the time series.
    
    Parameters:
    ----------
    signal : array-like
        The input time series
        
    Returns:
    -------
    float
        The sum of all positive values in the time series
    """
    return np.sum(signal[signal > 0])

@name("sum_of_reoccurring_values")
def calculate_sum_of_reoccurring_values(signal, **kwargs):
    """
    Calculate the sum of all unique reoccurring values in the given signal.

    This function identifies the unique values in the input signal that appear 
    more than once and returns the sum of these reoccurring values. It does not 
    count how many times the values occur, but rather sums the values themselves 
    if they occur more than once.

    Parameters:
    -----------
    signal : array-like
        The input time series

    Returns:
    --------
    float or int
        The sum of the unique values that appear more than once in the signal.

    Example:
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 2, 3, 3, 3, 4])
    >>> calculate_sum_of_reoccurring_values(signal)
    5

    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    unique, counts = np.unique(signal, return_counts=True)
    return np.sum(unique[counts > 1])

@name("sum_of_reoccurring_data_points")
def calculate_sum_of_reoccurring_data_points(signal, **kwargs):
    """
    Calculate the sum of all data points that correspond to reoccurring values in the signal.

    This function identifies values in the input signal that occur more than once 
    and calculates the sum of all data points that contribute to these reoccurring 
    values. It multiplies each reoccurring value by its frequency and then sums 
    the results.

    Parameters:
    -----------
    signal : array-like
        The input time series is to be calculated. 

    Returns:
    --------
    float or int
        The sum of all data points associated with values that appear more than 
        once in the signal.

    Example:
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 2, 3, 3, 3, 4])
    >>> calculate_sum_of_reoccurring_data_points(signal)
    13

    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    unique, counts = np.unique(signal, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)

@name("variance_of_absolute_differences")
def calculate_variance_of_absolute_differences(signal, **kwargs):
    """
    Calculate the variance of the absolute differences between consecutive elements in a time series.

    This measure provides insight into the variability or volatility of changes in the signal. 
    It calculates the absolute difference between each consecutive pair of elements in the signal 
    and then computes the variance of these absolute differences.

    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the time series.

    Returns:
    --------
    float
        The variance of the absolute differences between consecutive elements. If the signal has fewer than 
        two elements, the function returns `NaN`.
    
    References:
    ----------
        - Liang, H., & Hong, L. (2015). The Absolute Difference Law For Expectations. The American Statistician, 
        69(1), 8–10. https://doi.org/10.1080/00031305.2014.994712
    """
    # Ensure the signal has at least two elements
    if len(signal) < 2:
        return float('NaN')

    abs_diffs = np.abs(np.diff(signal))
    return np.var(abs_diffs)

@name("winsorized_mean")
def calculate_winsorized_mean(signal, wm_limits, **kwargs):
    """
    Calculate the Winsorized Mean of a given signal.

    The Winsorized Mean is a robust measure of central tendency that involves 
    limiting extreme values in the data. Specifically, the highest and lowest 
    values are replaced with the nearest values that fall within a specified 
    percentile range, reducing the effect of outliers.

    Parameters:
    -----------
    signal : array-like
        The input time series

    wm_limits : list of two floats, optional
        The proportions of data to Winsorize from the lower and upper ends of the 
        distribution. The default is [0.05, 0.05], which means 5% of the data is 
        Winsorized from both ends. The values should be between 0 and 0.5.
        
    References:
    ----------
        https://www.investopedia.com/terms/w/winsorized_mean.asp
    """
    return stats.mstats.winsorize(signal, limits=wm_limits).mean()

@name("zero_crossing_rate")
def calculate_zero_crossing_rate(signal, **kwargs):
    """
    Calculate the Zero Crossing Rate (ZCR) of a given signal.

    The Zero Crossing Rate is the rate at which the signal changes sign, 
    i.e., from positive to negative or vice versa.
    
    Parameters:
    -----------
    signal : array-like
        The input time series

    Returns:
    --------
    float
        The Zero Crossing Rate.
        
    References:
    ---------
        - McFee, B., Matt McVicar, Daniel Faronbi, Iran Roman, Matan Gover, Stefan Balke, Scott Seyfarth, Ayoub Malek, 
        Colin Raffel, Vincent Lostanlen, Benjamin van Niekirk, Dana Lee, Frank Cwitkowitz, Frank Zalkow, Oriol Nieto, 
        Dan Ellis, Jack Mason, Kyungyun Lee, Bea Steers, … Waldir Pimenta. (2024). librosa/librosa: 0.10.2.post1 (0.10.2.post1). 
        Zenodo. https://doi.org/10.5281/zenodo.11192913
    """
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    return len(zero_crossings) / len(signal)

#TODO: NEED TO FIX THIS
@name(["detrended_fluctuation_analysis_segments", "detrended_fluctuation_analysis_values"])
def calculate_detrended_fluctuation_analysis(signal, dfa_order=1, dfa_minimal=20, **kwargs):
    """
    Performs detrended fluctuation analysis (DFA) on the given signal.
        
    Parameters:
    ----------
    signal : array
        The input time series.
    dfa_order: integer
        The order of the polynomial fit for local detrending default is 1 for linear detrending).
    dfa_minimal: integer
        The minimum segment size to consider.
        
    Returns:
    -------
    segment_sizes: array
        Array of segment sizes.
    fluctuation_values: array
        Fluctuation function values corresponding to segment sizes.
        
    References:
    ----------
        - Bryce, R. M., & Sprague, K. B. (2012). Revisiting detrended 
        fluctuation analysis. Scientific Reports 2012 2:1, 2(1), 1–6. 
        https://doi.org/10.1038/srep00315
        - Zhang, H.-Y., Feng, Z.-Q., Feng, S.-Y., & Zhou, Y. (2023). 
        A Survey of Methods for Estimating Hurst Exponent of Time 
        Sequence. https://arxiv.org/abs/2310.19051v1
    """
    # Calculate the cumulative sum of the mean-shifted signal
    signal_mean = np.mean(signal)
    mean_shifted_signal = signal - signal_mean
    cumulative_sum_signal = np.cumsum(mean_shifted_signal)

    N = len(signal)

    def divisors(N, minimal=20, **kwargs):
        d = []
        for i in range(dfa_minimal, N // minimal + 1):
            if N % i == 0:
                d.append(i)
        return d

    def find_opt_n(N, minimal, **kwargs):
        """
        Find such a natural number opt_n that possesses the largest number of
        divisors among all natural numbers in the interval [0.99*N, N]
        """
        N0 = int(0.99 * N)
        # The best length is the one which have more divisiors
        d_count = [len(divisors(i, minimal)) for i in range(N0, N + 1)]
        opt_n = N0 + d_count.index(max(d_count))
        return opt_n

    opt_n = find_opt_n(len(signal), minimal=dfa_minimal)
    segment_sizes = divisors(opt_n, minimal=dfa_minimal)
    fluctuation_values = []

    for m in segment_sizes:
        k = opt_n // m
        Y = np.reshape(cumulative_sum_signal[N - opt_n:], [m, k], order='F')
        F = np.copy(Y)
        # t = 1, 2, ..., m
        t = np.linspace(1, m, m)
        for i in range(k):
            p = np.polyfit(t, Y[:, i], 1)
            F[:, i] = Y[:, i] - t * p[0] - p[1]
        fluctuation_values.append(np.mean(np.std(F)))

    return segment_sizes, np.array(fluctuation_values)

@name("hurst_exponent")
def calculate_hurst_exponent(signal, **kwargs):
    """
    Calculate the Hurst Exponent of a given time series using Detrended Fluctuation Analysis (DFA).

    The Hurst Exponent is a measure of the long-term memory of time series data, 
    indicating whether the data is a random walk (H ≈ 0.5), a trending series (H > 0.5), 
    or a mean-reverting series (H < 0.5). This function estimates the Hurst Exponent 
    using the Detrended Fluctuation Analysis (DFA) method.
        
    Parameters:
    ----------
    signal : np.array
        The input time series

    Returns:
    -------
    float
        The Hurst exponent of the given time series.

    References:
    ----------
        - Bryce, R. M., & Sprague, K. B. (2012). Revisiting detrended 
        fluctuation analysis. Scientific Reports 2012 2:1, 2(1), 1–6. 
        https://doi.org/10.1038/srep00315
        - Zhang, H.-Y., Feng, Z.-Q., Feng, S.-Y., & Zhou, Y. (2023). 
        A Survey of Methods for Estimating Hurst Exponent of Time 
        Sequence. https://arxiv.org/abs/2310.19051v1
        - https://github.com/GrAbsRD/HurstExponent
    """
    segment_size, fluctuation_values = calculate_detrended_fluctuation_analysis(signal)

    poly = np.polyfit(np.log(segment_size), np.log(fluctuation_values), 1)
    hurst = poly[0]
    return hurst

@name(["adf_teststats", "adf_pvalue", "adf_usedlag"])
def calculate_augmented_dickey_fuller_test(signal, **kwargs):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity in a given time series signal.

    The ADF test is a statistical test used to determine if a time series is stationary or has a unit root.
    A stationary time series has constant mean and variance over time.

    Parameters:
    ----------
    signal : np.array
        The time series data to be tested for stationarity.

    Returns:
    -------
    np.array
        A numpy array containing the test statistic, p-value, and number of lags used in the test.
        If the test fails due to an exception, returns NaN.

    References:
    ---------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    try:
        test_stat, p_value, used_lag, _,_,_ = adfuller(signal)
        adf_vals = np.array([test_stat, p_value, used_lag])
    except:
        return np.array([np.nan, np.nan, np.nan])
    return adf_vals

@name("has_duplicates")
def calculate_duplicates(signal, **kwargs):
    """
    Checks if the time series has duplicate values.
    
    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    -------
    boolean
        True: duplicates present
        False: No duplicates
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    unique, counts = np.unique(signal, return_counts=True)
    result = True if len(counts[counts > 1]) else False
    return result

@name("max_has_duplicates")   
def calculate_max_duplicates(signal, **kwargs):
    """
    Checks if the maximum value of the time series occurs more than once.
    
    Parameters:
    ----------
    signal : np.array
        The input time series.

    Returns:
    --------
    boolean
        True: maximum value has duplicates 
        False: maximum value has no duplicates
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    max_value = np.max(signal)
    unique, counts = np.unique(signal, return_counts=True)
    result = True if counts[np.where(unique == max_value)[0][0]] > 1 else False
    return result

@name("min_has_duplicates")
def calculate_min_duplicates(signal, **kwargs):
    """
    Checks if the minimum value of the time series occurs more than once.
    
    Parameters:
    ----------
    signal : np.array 
        The input time series.

    Returns:
    --------
    boolean
        True: minimum value has duplicates 
        False: minimum value has no duplicates
    
    References:
    ----------
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    max_value = np.min(signal)
    unique, counts = np.unique(signal, return_counts=True)
    result = True if counts[np.where(unique == max_value)[0][0]] > 1 else False
    return result

@name("large_std")
def calculate_large_std(signal, **kwargs):
    """
    Determines if the standard deviation of a signal is large relative to its range.

    This function compares the standard deviation of the given signal to a threshold, which 
    is calculated as a multiple of the signal's range. The multiplier (`r`) is determined 
    based on the length of the signal (`N`). Specifically, if the length of the signal 
    is between 15 and 70, `r` is set to 4; otherwise, it is set to 6.

    Parameters:
    -----------
    signal : array-like
        A sequence of numerical values representing the signal or time series data.

    Returns:
    --------
    bool
        `True` if the standard deviation of the signal is greater than the threshold 
        (`r * range`), `False` otherwise.

    Notes:
    ------
    - The `calculate_range` function is assumed to return the range of the signal as the 
    first element of a tuple.
    - The function is particularly useful for detecting whether the signal exhibits high 
    variability relative to its range.

    References:
    ---------
        - Wan, X., Wang, W., Liu, J., & Tong, T. (2014). Estimating the sample mean and standard 
        deviation from the sample size, median, range and/or interquartile range. BMC Medical 
        Research Methodology, 14(1), 1–13. https://doi.org/10.1186/1471-2288-14-135/TABLES/3
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series Feature 
        Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 
        307, 72–77. https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    range = calculate_range(signal)
    N = len(signal)
    r = 4 if 15 < N <=70 else 6

    return np.std(signal) > (r * range)

@name("lempel_ziv_complexity")
def calculate_lempel_ziv_complexity(signal, lz_bins, **kwargs):
    """
    Calculate the Lempel-Ziv complexity of a given time series.

    Lempel-Ziv complexity is defined as the number of distinct patterns or sub-sequences 
    required to represent the time series when scanned from left to right. The signal 
    is first discretized into a specified number of bins. Then, the time series is 
    converted into a sequence of sub-sequences, and the complexity is calculated as 
    the ratio of the number of unique sub-sequences to the total length of the time series.

    Parameters:
    -----------
    signal : array-like
        The input time series
    lz_bins : int
        The number of bins to discretize the time series into.

    Returns:
    --------
    float
        The Lempel-Ziv complexity estimate, representing the ratio of the number 
        of unique sub-sequences to the length of the time series.

    References:
    ----------
    Based on the implementation from:
        - https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lempel_ziv_complexity.py
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """
    signal = np.asarray(signal)

    # Discretize the signal into bins
    bin_edges = np.linspace(np.min(signal), np.max(signal), lz_bins + 1)[1:]
    discretized_sequence = np.searchsorted(bin_edges, signal, side="left")

    unique_subsequences = set()
    length = len(discretized_sequence)

    ind = 0
    inc = 1
    while ind + inc <= length:
        # Convert to tuple to make it hashable
        subsequence = tuple(discretized_sequence[ind : ind + inc])
        if subsequence in unique_subsequences:
            inc += 1
        else:
            unique_subsequences.add(subsequence)
            ind += inc
            inc = 1

    return len(unique_subsequences) / length

@name("cid_ce")
def calculate_cid_ce(signal, cid_ce_normalize, **kwargs):
    """
    Calculate the Complexity Estimate (CE) of a time series signal.

    This function computes the complexity estimate of a given time series signal, which is a measure of 
    the complexity or irregularity of the signal's behavior. The complexity estimate is calculated as the 
    square root of the sum of squared differences of consecutive signal values. If the `normalize` parameter 
    is set to True, the signal is first normalized by subtracting the mean and dividing by the standard deviation.

    Parameters:
    ----------
    signal : array-like
        The input time series.
    
    normalize : bool
        If True, the signal is normalized before computing the complexity estimate. Normalization ensures that 
        the signal has zero mean and unit variance. If the standard deviation is zero (i.e., the signal is constant), 
        the function returns 0.0.

    Returns:
    -------
    float
        The complexity estimate of the signal. A higher value indicates a more complex signal.

    References:
    ---------
        - Batista, G. E. A. P. A., Wang, X., & Keogh, E. J. (2011). A complexity-invariant distance measure 
        for time series. Proceedings of the 11th SIAM International Conference on Data Mining, SDM 2011, 
        699–710. https://doi.org/10.1137/1.9781611972818.60
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series Feature 
        Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 
        307, 72–77. https://doi.org/10.1016/J.NEUCOM.2018.03.067

    Notes:
    -----
        - The CID-CE measure is useful in time series analysis for comparing the complexity of different 
        time series, as it is invariant to linear transformations when normalized.
        - If `normalize` is set to True, the signal is scaled to have zero mean and unit variance. This 
        step is crucial when comparing signals of different scales.
    """
    if cid_ce_normalize:
        s = np.std(signal)
        if s != 0:
            signal = (signal - np.mean(signal)) / s
        else:
            return 0.0

    signal = np.diff(signal)
    return np.sqrt(np.dot(signal, signal))

@exclude()
@name("conditional_entropy")
def calculate_conditional_entropy(signal, **kwargs):
    """
    Calculates the entropy of the signal X, given the entropy of X

    Args:
        signal: np.array
            The input time series.
        
    Returns:
        array: An array containing the conditional entropy of the signal.
        
    References:
    ---------
    
    """
    pass

@name("benford_correlation")
def calculate_benford_correlation(signal, **kwargs):
    """
    Compute the correlation between the first digit distribution of the input data
    and the Benford's law distribution.
    
    Parameters:
    -----------
    signal : array-like
        The input time series.
    
    Returns:
    --------
    float: Correlation coefficient between the data distribution and Benford's law distribution
    
    References:
    -----------
        - Benford, F. (1938). The Law of Anomalous Numbers (Vol. 78, Issue 4).
        - Newcomb, S. (1881). Note on the Frequency of Use of the Different Digits in Natural Numbers. 
        American Journal of Mathematics, 4(1/4), 39. https://doi.org/10.2307/2369148
        - Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on 
        basis of Scalable Hypothesis tests (tsfresh – A Python package). Neurocomputing, 307, 72–77. 
        https://doi.org/10.1016/J.NEUCOM.2018.03.067
    """   
    signal = np.asarray(signal) 
    # Handle NaN values
    signal = np.nan_to_num(signal)
    
    # Get the first digit of the absolute values of the data
    first_digit = np.array([int(str(np.format_float_scientific(i))[:1]) for i in np.abs(signal)])
    
    # Compute the Benford's law distribution
    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
    
    # Compute the data distribution
    data_distribution = np.array([(first_digit == n).mean() for n in range(1, 10)])

    return np.corrcoef(benford_distribution, data_distribution)[0, 1]

@name("number_cwt_peaks_{}","cwt_peaks_n")
def calculate_number_cwt_peaks(signal, cwt_peaks_n, **kwargs):
    """
    Calculate the number of peaks in a signal using the Continuous Wavelet Transform (CWT) method 
    for a range of widths.

    Parameters:
    ----------
    signal : array-like
        The input signal for which peaks are to be detected.
        
    cwt_peaks_n : list of int
        A list of integers specifying the maximum width for each CWT analysis.
        For each value `n` in this list, peak detection will be performed with widths ranging from 1 to `n`.
        
    Returns:
    -------
    peak_lengths : list of int
        A list where each element corresponds to the number of peaks detected for the respective `cwt_peaks_n` value.
        
    Notes:
    -----
    - The function uses the Ricker wavelet (Mexican hat wavelet) for the CWT.
    """
    peak_lengths = []
    
    for i in cwt_peaks_n:
        peak_lengths.append(len(
            find_peaks_cwt(vector=signal, widths=np.array(list(range(1, i + 1))), wavelet=ricker)
        ))
        
    return peak_lengths
