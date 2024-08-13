import numpy as np
from scipy.stats import skew, kurtosis, moment, gmean, hmean, trim_mean, entropy, linregress, mode, pearsonr
from statsmodels.tsa.stattools import acf, adfuller
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import detrend, argrelextrema, find_peaks
from itertools import groupby
from scipy.ndimage.filters import convolve
from scipy import stats




def calculate_mean(signal):
        """
        Calculates the mean of the given signal.

        Parameters:
        ---------
        signal: np.array
            An array of values corresponding to the signal.

        Returns:
        -------
        np.array
            An array containing the mean of the signal.
        
        References:
            Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
        """
        return np.array([np.mean(signal)])


def calculate_geometric_mean(signal):
        """
        Calculates the geometric mean of the given signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.

        Returns:
        -------
            np.array
                An array containing the geometric mean of the signal.
        
        References:
        ----------
            Chaddad et al., 2014, DOI: 10.1117/12.2062143
        """
        signal = signal[signal > 0]
        return np.array([gmean(signal)])


def calculate_harmonic_mean(self, signal):
        """
        Calculates the harmonic mean of the given signal.
        Only positive values in the signal are considered.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.

        Returns:
        -------
        np.array
            An array containing the harmonic mean of the signal.
        
        References:
        ----------
            Chaddad et al., 2014, DOI: 10.1117/12.2062143
        """
        # Filter out non-positive values
        signal = signal[signal > 0]
        return np.array([hmean(signal)])


def calculate_trimmed_mean(signal, trimmed_mean_thresholds):
        """
        Calculate the trimmed mean of the given signal for different proportions.
        
        The trimmed mean excludes a fraction of the smallest and largest values 
        from the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the signal.
        self.trimmed_mean_thresholds (list): 
            List of proportions to cut from each end of the signal.

        Returns:
        -------
            np.array: An array containing the trimmed means for each proportion.
        
        References:
        ----------
            Chaddad et al., 2014, DOI: 10.1117/12.2062143
        """
        feats = []
        for proportiontocut in trimmed_mean_thresholds:
            feats.append(trim_mean(signal, proportiontocut=proportiontocut))
        return np.array(feats)

def calculate_mean_abs(signal):
        """
        Calculate the mean of the absolute values of the given signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.

        Returns:
        -------
            np.array
                An array containing the mean of the absolute values of the signal.
        
        References:
        ----------
            Myroniv et al., 2017, https://www.researchgate.net/publication/323935725
            Phinyomark et al., 2012, DOI: 10.1016/j.eswa.2012.01.102
            Purushothaman et al., 2018, DOI: 10.1007/s13246-018-0646-7
        """
        return np.array([np.mean(np.abs(signal))])


def calculate_geometric_mean_abs(signal):
        """
        Calculate the geometric mean of the absolute values of the given signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.

        Returns:
        -------
            np.array
                An array containing the geometric mean of the absolute values of the signal.
        References:
        ----------
            DOI:10.1134/S1064226917050060
        """
        return np.array([gmean(np.abs(signal))])


def calculate_harmonic_mean_abs(signal):
        """
        Calculate the harmonic mean of the absolute values of the given signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.

        Returns:
        -------
            np.array
                An array containing the harmonic mean of the absolute values of the signal.
        Reference:
        ---------
        
        """
        return np.array([hmean(np.abs(signal))])


def calculate_trimmed_mean_abs(signal, trimmed_mean_thresholds):
        """
        Calculate the trimmed mean of the absolute values of the given signal for 
        different proportions.
        
        The trimmed mean excludes a fraction of the smallest and largest values 
        from the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the signal.
        self.trimmed_mean_thresholds: list
            List of proportions to cut from each end of the absolute values of the signal.

        Returns:
        -------
            np.array
                An array containing the trimmed means for each proportion.
        Reference:
        ---------
        ---------
        """
        feats = []
        for proportiontocut in trimmed_mean_thresholds:
            feats.append(trim_mean(np.abs(signal), proportiontocut=proportiontocut))
        return np.array(feats)


def calculate_std(signal):
        """
        Calculates the standard deviation of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the standard deviation of the signal.
            
        Reference:
        ---------
            Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
        """
        return np.array([np.std(signal)])

def calculate_std_abs(signal):
        """
        Calculates the standard deviation of the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the standard deviation of the absolute values of the signal.
            
        Reference:
        ---------
        """
        return np.array([np.std(np.abs(signal))])

def calculate_skewness(signal):
        """
        Calculates the skewness of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the skewness of the signal.
            
        References:
        ----------
            Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
            Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        """        
        return np.array([skew(signal)])

def calculate_skewness_abs(signal):
        """
        Calculates the skewness of the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array 
                An array containing the skewness of the absolute values of the signal.
            
        Reference:
        ---------
        """
        return np.array([skew(np.abs(signal))])

def calculate_kurtosis(signal):
        """
        Calculates the kurtosis of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array:
                An array containing the kurtosis of the signal.
            
        References:
        ----------
            Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
            Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        """
        return np.array([kurtosis(signal)])

def calculate_kurtosis_abs(signal):
        """
        Calculates the kurtosis of the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the kurtosis of the absolute values of the signal.
            
        Reference:
        """
        return np.array([kurtosis(np.abs(signal))])

def calculate_median(signal):
        """
        Calculates the median of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array:
                An array containing the median  of the signal.
            
        Reference:
        ---------
            Banos et al., 2012, DOI: 10.1016/j.eswa.2012.01.164
        """
        
        return np.array([np.median(signal)])

def calculate_median_abs(signal):
        """
        Calculates the median of the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the median of the absolute values of the signal.
            
        Reference:
        ---------
        """
        return np.array([np.median(np.abs(signal))])

def calculate_min(signal):
        """
        Calculates the minimum value of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the minimum value of the signal.
            
        Reference:
        ---------
            18th International Conference on Computer Communications and Networks, DOI: 10.1109/ICCCN15201.2009
        """
        
        min_val = np.min(signal)
        return np.array([min_val])

def calculate_min_abs(signal):
        """
        Calculates the minimum value of the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the minimum value of the absolute values of the signal.
            
        Reference:
        ---------
        """
        min_abs_val = np.min(np.abs(signal))
        return np.array([min_abs_val])

def calculate_max(signal):
        """
        Calculates the maximum value of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the maximum value of the signal.
            
        Reference:
        ---------
            Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        """
        
        max_val = np.max(signal)
        return np.array([max_val])

def calculate_max_abs(signal):
        """
        Calculates the maximum value of the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the maximum value of the absolute values of the signal.
            
        Reference:
        ---------
        """
        max_abs_val = np.max(np.abs(signal))
        return np.array([max_abs_val])

def calculate_range(signal):
        """
        Calculates the range of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the range of the signal.
            
        Reference:
        ---------
        """
        return np.array([np.max(signal) - np.min(signal)])

def calculate_range_abs(signal):
        """
        Calculates the range of the absolute values of the signal.

        Parameters:
        ----------
        signal : np.array
            An array of values corresponding to the input signal.
            
        Returns:
        -------
            np.array
                An array containing the range of the absolute values of the signal.
            
        Reference:
        ---------
        """
        abs_signal = np.abs(signal)
        return np.array([np.max(abs_signal) - np.min(abs_signal)])

def calculate_variance(signal):
        """
        Calculates the variance of the signal.

        Parameters:
        ----------
            signal (array-like): The input time series.
            
        Returns:
        -------
            np.array
                An array containing the variance of the signal.
            
        Reference:
        ---------
            Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        """
        return np.array([np.var(signal)])

def calculate_variance_abs(signal):
        """
        Calculates the variance of the absolute values of the signal.

        Parameters:
        ----------
            signal (array-like): The input time series.
            
        Returns:
        -------
            np.array
                An array containing the variance of the absolute values of the  signal.
            
        Reference:
        ---------
        """
        return np.array([np.var(np.abs(signal))])

def calculate_interquartile_range(signal):
        """
        Calculate the interquartile range (IQR) of the given signal.
        
        The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the signal.

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the interquartile range of the signal.
        
        References:
        ----------
            Bedeeuzzaman et al., 2012, DOI: 10.5120/6304-8614
        """
        return np.array([np.percentile(signal, 75) - np.percentile(signal, 25)])

def calculate_mean_absolute_deviation(signal):
        """
        Calculate the mean absolute deviation (MAD) of the given signal.
        
        The MAD is the average of the absolute deviations from the mean of the signal.

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the mean absolute deviation of the signal.
        
        References:
            - Pham-Gia, T., & Hung, T. L. (2001). The mean and median absolute deviations. Mathematical and Computer Modelling,
            34(7–8), 921–936. https://doi.org/10.1016/S0895-7177(01)00109-1
        """
        return np.array([np.mean(np.abs(signal - np.mean(signal)))])


def calculate_root_mean_square(signal):
        """
        Calculate the root mean square (RMS) of the given signal.
        
        The RMS is the square root of the mean of the squares of the signal values.

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the root mean square of the signal.
        
        References:
        ----------
            - Whitaker, B. M., Suresha, P. B., Liu, C., -,  al, Wang, G., Huang, J., Zhang -, F., 
            Mohd Khan, S., Ali Khan, A., Farooq -, O., Khorshidtalab, A., E Salami, M. J., & 
            Hamedi, M. (2013). Robust classification of motor imagery EEG signals using statistical 
            time–domain features. Physiological Measurement, 34(11), 1563. https://doi.org/10.1088/0967-3334/34/11/1563
            - 18th International Conference on Computer Communications and Networks, DOI: 10.1109/ICCCN15201.2009
        """
        return np.array([np.sqrt(np.mean(signal**2))])


def calculate_signal_energy(signal):
        """
        Calculate the energy of the given signal.
        
        The energy of the signal is the sum of the squares of its values.

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the energy of the signal.
        
        References:
            Rafiuddin et al., 2011, DOI: 10.1109/MSPCT.2011.6150470
        """
        return np.array([np.sum(signal**2)])


def calculate_log_energy(signal):
        """
        Calculate the logarithm of the energy of the given signal.
        
        The log energy is the natural logarithm of the sum of the squares of the signal values.

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array:
                An array containing the logarithm of the energy of the signal.
            
        Reference:
        ---------
        https://mathworks.com/help/audio/ref/mfcc.html
        """
        return np.array([np.log(np.sum(signal**2))])


def calculate_entropy(signal, window_size):
        """
        Calculate the entropy of the given signal.
        
        The entropy is a measure of the uncertainty or randomness in the signal,
        calculated from its histogram.

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array:
                An array containing the entropy of the signal.
        
        References:
        ----------
            Guido, 2018, DOI: 10.1016/j.inffus.2017.09.006
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
        return np.array([entropy])
    
def calculate_sample_entropy(signal):
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
    1. https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html
    2. Kumar et al, 2012, https://doi.org/10.1109/SCEECS.2012.6184830
    """
    return entropy(np.histogram(signal, bins=10)[0])

def calculate_differential_entropy(signal):
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
        Zhu et al., 2021, https://doi.org/10.3389/FPHY.2020.629620/BIBTEX
    """
    probability, _ = np.histogram(signal, bins=10, density=True)
    probability = probability[probability > 0]
    return entropy(probability)

def calculate_approximate_entropy(signal):
    """
    Calculate the approximate entropy of a signal.

    Approximate entropy is a statistic used to quantify the amount of regularity and the 
    unpredictability of fluctuations in a time series. Lower values indicate a more 
    predictable series, while higher values indicate more complexity or randomness.
    
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
        Delgado-Bonal et al., 2019, https://doi.org/10.3390/E21060541
    """
    count, _ = np.histogram(signal, bins=10, density=True)
    count = count[count > 0]  # Avoid log(0) issue
    return -np.sum(count * np.log(count))
    
def calculate_renyi_entropy(signal, window_size, renyi_alpha_parameter):
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
    numpy.ndarray
        A numpy array containing the calculated Rényi entropy of the signal. If an error 
        occurs during the calculation, the function returns NaN.

    References:
    -----------
    1. Beadle et al., 2008, DOI: 10.1109/ACSSC.2008.5074715
    """
    try:
        # Calculate the histogram
        hist, _ = np.histogram(signal, bins= window_size//2, density=True)
        # Replace zero values with a small epsilon
        epsilon = 1e-10
        hist = np.where(hist > 0, hist, epsilon)

        if renyi_alpha_parameter == 1:
            # Return the Shannon entropy
            return np.array([-sum([p * (0 if p == 0 else np.log(p)) for p in hist])])
        else:
            return np.array([(1 / (1 - renyi_alpha_parameter)) * np.log(sum([p ** renyi_alpha_parameter for p in hist]))])
    except:
        return np.array([np.nan])
        
def calculate_tsallis_entropy(signal, window_size, tsallis_q_parameter):
        """
        Calculates the tsallis entropy of the signal.
        Tsallis entropy generalizes the Shannon entropy parameterized by a parameter 𝑞
        q, which introduces non-extensivity (a form of non-linearity) into the entropy measure.

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array:
                An array containing the Tsallis entropy of the signal.
        Reference:
        ---------
            Sneddon et al., 2007, https://doi.org/10.1016/J.PHYSA.2007.05.065
        """
        try:
            # Calculate the histogram
            hist, _ = np.histogram(signal, bins=window_size//2, density=True)
            # Replace zero values with a small epsilon
            epsilon = 1e-10
            hist = np.where(hist > 0, hist, epsilon)
            if tsallis_q_parameter == 1:
                # Return the Boltzmann–Gibbs entropy
                return np.array([-sum([p * (0 if p == 0 else np.log(p)) for p in hist])])
            else:
                return np.array([(1 - sum([p ** tsallis_q_parameter for p in hist])) / (tsallis_q_parameter - 1)])
        except:
            return np.array([np.nan])
        
def calculate_svd_entropy(signal, svd_entropy_order, svd_entropy_delay):
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
    numpy.ndarray
        A numpy array containing the calculated SVD entropy value. If an error occurs during the 
        calculation, the function returns NaN.

    References:
    -----------
    1. Banerjee et al., 2014, DOI: 10.1016/j.ins.2013.12.029
    2. Strydom et al., 2021, DOI: 10.3389/fevo.2021.623141
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
    return np.array([svd_entropy_value])
    
def calculate_permutation_entropy(signal, window_size,permutation_entropy_order, permutation_entropy_delay):
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
    
    References:
    -----------
        Bandt et al., 2002, DOI: 10.1103/PhysRevLett.88.174102
        Zanin et al., 2012, DOI: 10.3390/e14081553
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
    return np.array([entropy_value])

def calculate_zero_crossings( signal):
        """
        Calculates the number of times the signal crosses zero
        
        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the number of times(integer) the signal crosses zero
        
        References:
        ----------
            Myroniv et al., 2017, https://www.researchgate.net/publication/323935725_Analyzing_User_Emotions_via_Physiology_Signals
            Sharma et al., 2020, DOI: 10.1016/j.apacoust.2019.107020
            Purushothaman et al., 2018, DOI: 10.1007/s13246-018-0646-7
        """
        

        # Compute the difference in signbit (True if number is negative)
        zero_cross_diff = np.diff(np.signbit(signal))
        # Sum the differences to get the number of zero-crossings
        num_zero_crossings = zero_cross_diff.sum()
        return np.array([num_zero_crossings])

def calculate_crest_factor(signal):
        """
        Calculate the crest factor of the given signal.
        
        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the crest factor(float) of the signal.
        
        References:
        ----------
            Formula from Cempel, 1980, DOI: 10.1016/0022-460X(80)90667-7
            Wang et al., 2015, DOI: 10.3390/S150716225, https://doi.org/10.3390/S150716225  
        """
        crest_factor = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))
        return np.array([crest_factor])

def calculate_clearance_factor(signal):
        """
        Calculate the clearance factor of the given signal.
        
        The clearance factor is a measure used in signal processing to 
        quantify the peakiness of a signal. It isdefined as the ratio 
        of the maximum absolute value of the signal to the square of 
        the mean square root of the absolute values of the signal.
        
        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the clearance factor (float) of the signal.
        
        References:
        ----------     
            Formula from The MathWorks Inc., 2022, Available: [Signal Features](https://www.mathworks.com)
            Wang et al., 2015, DOI: 10.3390/S150716225, https://doi.org/10.3390/S150716225       
        """
        
        clearance_factor = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal))) ** 2)
        return np.array([clearance_factor])

def calculate_shape_factor(signal):
    """
    Calculates the shape factor of the time series.
    
    The shape factor is a measure of the waveform shape of a signal, which is the ratio of the root mean square (RMS) value to the mean absolute value of the signal.

    Parameter:
    ---------
        signal (array-like): The input time series.

    Returns:
    -------
    np.array
        An array containing the shape factor (float) of the signal.
        
    Reference:
    ----------
        Cempel, 1980, DOI: 10.1016/0022-460X(80)90667-7
    """
    shape_factor = np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal))
    return np.array([shape_factor])

def calculate_mean_crossing(signal):
    """
    Calculate the number of mean crossings in a time series signal.

    Mean crossing refers to the number of times the signal crosses its mean value.

    Parameter:
    ---------
        signal (array-like): The input time series.

    Returns:
    -------
    np.array
        An array containing the shape factor (float) of the signal.
        
    Reference:
    ----------
        Myroniv et al., 2017, https://www.researchgate.net/publication/323935725
    """
    mean_value = np.mean(signal)
    mean_crossings = np.where(np.diff(np.sign(signal - mean_value)))[0]
    return np.array([len(mean_crossings)])

def calculate_impulse_factor(signal):
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
    np.ndarray
        A single-element array containing the impulse factor (float) of the signal. Higher values indicate a signal with sharp 
        peaks relative to its average level, while lower values suggest a more uniform signal.

    References:
    -----------
        Cempel, 1980, DOI: 10.1016/0022-460X(80)90667-7
    """
    impulse_factor = np.max(np.abs(signal)) / np.mean(np.abs(signal))
    return np.array([impulse_factor])

def calculate_mean_auto_correlation(signal, n_lags_auto_correlation):
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
        A single-element array containing the mean auto-correlation value (float) of the signal.
        
    References:
    ----------
        Fulcher, 2017, DOI: 10.48550/arXiv.1709.08055
        Banos et al., 2012, DOI: 10.1016/j.eswa.2012.01.164
    
    """
    auto_correlation_values = acf(signal, nlags= n_lags_auto_correlation)[1:]
    return np.array([np.mean(auto_correlation_values)])

def calculate_higher_order_moments(signal, moment_orders):
    """
    Calculates the higher order moments of the given time series.

    Parameters:
    ---------
        signal (array-like): The input time series.
        moment_orders:

    moment_orders : array-like
        A list or array of integers specifying the orders of the moments to be calculated. For example, [3, 4] will calculate 
        the 3rd and 4th moments (skewness and kurtosis).
        
    Returns:
    -------
        np.array
            An array containing the higher order moments of the signal.
        
    Reference:
    ---------
        Clerk et al., 2022, https://doi.org/10.1016/J.HELIYON.2022.E08833
    """
    feats = []
    for order in moment_orders:
            feats.append(moment(signal, moment=order))
    return np.array(feats)

def calculate_coefficient_of_variation(signal):
    """
    Calculate the coefficient of variation (CV) of a time series signal.

    The coefficient of variation is a standardized measure of the dispersion of the data points in a signal. It is defined as 
    the ratio of the standard deviation to the mean, providing a relative measure of variability regardless of the signal's 
    scale. This metric is particularly useful in comparing the degree of variation between signals with different units or 
    widely different means.
    
    Parameters:
    ---------
        signal (array-like): The input time series.

    Returns:
    -------
        np.array
            A single-element array containing the coefficient of variation (float) of the signal. 
        
    Reference:
    ---------
        Jalilibal et al., 2021, DOI: 10.1016/j.cie.2021.107600
    
    """
    coefficient_of_variation = np.std(signal) / np.mean(signal)
    return np.array([coefficient_of_variation])

def calculate_median_absolute_deviation(signal, adjusted):
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
        np.array
            A single-element array containing the MAD (float) of the signal. This value represents the median of the absolute 
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
    return np.array([mad])

# def calculate_signal_magnitude_area(self, signal):
#         # Formula from Khan et al., 2010, DOI: 10.1109/TITB.2010.2051955
#         # Formula from Rafiuddin et al., 2011, DOI: 10.1109/MSPCT.2011.6150470
#         return np.array([np.sum(np.abs(signal))])

def calculate_avg_amplitude_change(signal):
    """
    Calculates the average wavelength of the time series.
    
    Parameters:
    ----------
    signal : array-like
        The input time series data.

    Returns:
    -------
        np.array:
            A single-element array containing the average amplitude change (float) of the signal.
    
    Reference:
    ----------
        - Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. 
        Expert Systems with Applications, 39(8), 7420–7431. https://doi.org/10.1016/J.ESWA.2012.01.102
    """
    avg_amplitude_change = np.mean(np.abs(np.diff(signal)))
    return np.array([avg_amplitude_change])

def calculate_slope_sign_change(signal, ssc_threshold):
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
    np.ndarray
        A single-element array containing the SSC value.
        
    Reference:
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
    return np.array([slope_sign_change])

def calculate_higuchi_fractal_dimensions(signal, higuchi_k_values):
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

def calculate_hjorth_mobility_and_complexity(signal):
    """
    Calculates mobility and complexity of the time series which are based on 
    the first and second derivatives of the time series.

    Parameter:
    ---------
        signal : array-like
        The input time series data.

    Returns:
    -------
    np.array
        Array containing the mobility and complexity of the time series.
        
    Reference:
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

def calculate_cardinality(self, signal):
        # Parameter
        thresh = 0.05 * np.std(signal)  # threshold
        # Sort data
        sorted_values = np.sort(signal)
        cardinality_array = np.zeros(self.window_size - 1)
        for i in range(self.window_size - 1):
            cardinality_array[i] = np.abs(sorted_values[i] - sorted_values[i + 1]) > thresh
        cardinality = np.sum(cardinality_array)
        return np.array([cardinality])

def calculate_rms_to_mean_abs(signal):
        """
        Calculates the ratio of the root-mean-squared value to the mean
        absolute value.

        Parameters:
        ---------
            signal (array-like): The input time series.

        Returns:
        -------
            np.array
                An array containing the ratio of the root-mean-squared value to the mean
                absolute value.
        
        Reference:
        ---------
        """
        rms_val = np.sqrt(np.mean(signal ** 2))
        mean_abs_val = np.mean(np.abs(signal))
        ratio = rms_val / mean_abs_val
        return np.array([ratio])

def calculate_area_under_curve(signal):
        """
        Calculates the area under the curve of the given signal

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            float: area under curve
        
        Reference:
        ---------
            - Kuremoto, T., Baba, Y., Obayashi, M., Mabu, S., & Kobayashi, K. 
            (2018). Enhancing EEG Signals Recognition Using ROC Curve. Journal 
            of Robotics, Networking and Artificial Life, 4(4), 283. https://doi.org/10.2991/JRNAL.2018.4.4.5
        """
        return simpson(np.abs(signal), dx=1)

def calculate_area_under_squared_curve(signal):
        """
        Calculates the area under the curve of the given signal squared

        Parameters:
        ----------
            signal (array-like): The input time series.

        Returns:
        -------
            float: area under curve of signal squared
        """
        return simpson(signal**2, dx=1)

def calculate_autoregressive_model_coefficients(signal, order=4):
        # https://doi.org/10.1109/IEMBS.2008.4650379
        model = AutoReg(signal, lags=order)
        model_fitted = model.fit()
        return model_fitted.params

def calculate_count(self, signal):
        return len(signal)

def calculate_count_above_mean(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        mean_val = np.mean(signal)
        return np.sum(signal > mean_val)

def calculate_count_below_mean(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        mean_val = np.mean(signal)
        return np.sum(signal < mean_val)

def calculate_count_of_negative_values(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        return np.sum(signal < 0)

def calculate_count_of_positive_values(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        return np.sum(signal > 0)

def calculate_covariance(self, signal, other_signal):
        # https://support.ptc.com/help/mathcad/r9.0/en/index.html#page/PTC_Mathcad_Help/covariance.html
        return np.cov(signal, other_signal)[0, 1]

def calculate_cumulative_energy(self, signal):
        # https://doi.org/10.1016/j.ijepes.2020.106192
        return np.cumsum(np.square(signal))[-1]

def calculate_cumulative_sum(self, signal):
        # https://docs.amd.com/r/2020.2-English/ug1483-model-composer-sys-gen-user-guide/Cumulative-Sum
        return np.cumsum(signal)[-1]

def calculate_energy_ratio_by_chunks(self, signal, chunks=4):
        # https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_extraction/feature_calculators.py#L2212
        chunk_size = len(signal) // chunks
        energies = np.array([np.sum(signal[i*chunk_size:(i+1)*chunk_size]**2) for i in range(chunks)])
        total_energy = np.sum(signal**2)
        return energies / total_energy
    
def calculate_moving_average(self, signal, window_size=10):
        # https://cyclostationary.blog/2021/05/23/sptk-the-moving-average-filter/
        if len(signal) < window_size:
            return np.nan
        return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
    
def calculate_weighted_moving_average(self, signal, weights=None):
        # https://www.mathworks.com/help/signal/ug/signal-smoothing.html
        if weights is None:
            weights = np.linspace(1, 0, num=len(signal))
        weights = weights / np.sum(weights)
        return np.convolve(signal, weights, 'valid')

def calculate_exponential_moving_average(self, signal, alpha=0.3):
        """
        Calculates the exponential moving average of the given signal

        Parameters:
        ---------
            signal (array-like): The input time series.
            alpha (float, optional): Defaults to 0.3.

        Returns:
        -------
            float
                last value in the array
        """
        ema = np.zeros_like(signal)
        ema[0] = signal[0]
        for i in range(1, len(signal)):
            ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
        return ema[-1]

def calculate_first_location_of_maximum(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        return np.argmax(signal)

def calculate_first_location_of_minimum(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        return np.argmin(signal)

def calculate_first_order_difference(self, signal):
        # https://cran.r-project.org/web/packages/doremi/vignettes/first-order.html
        return np.diff(signal, n=1)

def calculate_first_quartile(self, signal):
        return np.percentile(signal, 25)

def calculate_fisher_information(self, signal):
        # https://www.researchgate.net/publication/311396939_Analysis_of_Signals_by_the_Fisher_Information_Measure
        variance = np.var(signal)
        return 1 / variance if variance != 0 else float('inf')

def calculate_histogram_bin_frequencies(self, signal, bins=10):
        # https://doi.org/10.1016/B978-044452075-3/50032-7
        hist, _ = np.histogram(signal, bins=bins)
        return hist

def calculate_intercept_of_linear_fit(self, signal):
        # https://www.mathworks.com/help/matlab/data_analysis/linear-regression.html
        slope, intercept, _, _, _ = linregress(np.arange(len(signal)), signal)
        return intercept

def calculate_katz_fractal_dimension(self, signal):
        # https://doi.org/10.3390/fractalfract8010009
        distance = np.max(np.abs(np.diff(signal)))
        length = np.sum(np.abs(np.diff(signal)))
        return np.log10(length / distance)

def calculate_last_location_of_maximum(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        return np.max(np.where(signal == np.max(signal))[0])

def calculate_last_location_of_minimum(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        return np.max(np.where(signal == np.min(signal))[0])

def calculate_linear_trend_with_full_linear_regression_results(self, signal):
        # https://www.mathworks.com/help/matlab/data_analysis/linear-regression.html
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(signal)), signal)
        return slope, intercept, r_value**2, p_value, std_err

def calculate_local_maxima_and_minima(self, signal):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        local_max = find_peaks(signal)[0]
        local_min = find_peaks(-signal)[0]
        return len(local_max), len(local_min)

def calculate_log_return(self, signal):
        # https://pypi.org/project/stockstats/
        
        # Filter out non-positive values
        signal = signal[signal > 0]
        return np.log(signal[-1] / signal[0]) if signal[0] != 0 else float('inf')

def calculate_longest_strike_above_mean(self, signal):
        # https://tsfresh.readthedocs.io/en/v0.8.1/api/tsfresh.feature_extraction.html
        mean_val = np.mean(signal)
        return max([sum(1 for i in g) for k, g in groupby(signal > mean_val) if k])

def calculate_longest_strike_below_mean(self, signal):
        # https://tsfresh.readthedocs.io/en/v0.8.1/api/tsfresh.feature_extraction.html
        mean_val = np.mean(signal)
        return max([sum(1 for i in g) for k, g in groupby(signal < mean_val) if k])

def calculate_lower_complete_moment(self, signal, order=2):
        mean_val = np.mean(signal)
        return np.mean([(x - mean_val)**order for x in signal if x < mean_val])

def calculate_mean_absolute_change(self, signal):
        # https://en.wikipedia.org/wiki/Mean_absolute_difference
        return np.mean(np.abs(np.diff(signal)))

def calculate_mean_crossings(self, signal):
        # https://sensiml.com/documentation/pipeline-functions/feature-generators.html
        mean_val = np.mean(signal)
        return np.sum(np.diff(signal > mean_val))

def calculate_mean_relative_change(self, signal):
        # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.1365-2745.2007.01281.x
        return np.mean(np.abs(np.diff(signal) / signal[:-1]))

def calculate_mean_second_derivative_central(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
        return np.mean(np.diff(signal, n=2))

def calculate_median_second_derivative_central(self, signal):
        second_derivative = np.diff(signal, n=2)
        return np.median(second_derivative)

def calculate_mode(self, signal):
        return mode(signal)[0]

def calculate_number_of_inflection_points(self, signal):
        # https://en.wikipedia.org/wiki/Inflection_point
        second_derivative = np.diff(signal, n=2)
        return np.sum(np.diff(np.sign(second_derivative)) != 0)

def calculate_peak_to_peak_distance(self, signal):
        # https://www.mathworks.com/matlabcentral/fileexchange/20314-peak-to-peak-of-signal
        return np.ptp(signal)

def calculate_pearson_correlation_coefficient(self, signal):
        # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        if len(signal) < 2:
            return np.nan
        return pearsonr(signal[:-1], signal[1:])[0]

def calculate_percentage_of_negative_values(self, signal):
        return np.mean(signal < 0) * 100

def calculate_percentage_of_positive_values(self, signal):
        return np.mean(signal > 0) * 100

def calculate_percentage_of_reoccurring_datapoints_to_all_datapoints(self, signal):
        # https://tsfresh.readthedocs.io/en/v0.8.1/api/tsfresh.feature_extraction.html
        unique, counts = np.unique(signal, return_counts=True)
        return 100 * np.sum(counts > 1) / len(signal)

def calculate_percentage_of_reoccurring_values_to_all_values(self, signal):
        # https://tsfresh.readthedocs.io/en/v0.8.1/api/tsfresh.feature_extraction.html
        unique, counts = np.unique(signal, return_counts=True)
        return 100 * np.sum(counts[counts > 1]) / np.sum(counts)
    
def calculate_percentile(self, signal, percentiles=[25, 50, 75]):
        return np.percentile(signal, percentiles)

def calculate_petrosian_fractal_dimension(self, signal):
        # https://doi.org/10.7555%2FJBR.33.20190009
        N = len(signal)
        nzc = np.sum(np.diff(signal) != 0)
        return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nzc)))

def calculate_ratio_beyond_r_sigma(self, signal, r=2):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
        std_dev = np.std(signal)
        mean_val = np.mean(signal)
        return np.sum(np.abs(signal - mean_val) > r * std_dev) / len(signal)

def calculate_ratio_of_fluctuations(self, signal):
        increases = np.sum(np.diff(signal) > 0)
        decreases = np.sum(np.diff(signal) < 0)
        total = increases + decreases
        ratio_positive = increases / total if total != 0 else 0
        ratio_negative = decreases / total if total != 0 else 0
        return ratio_positive, ratio_negative, ratio_positive / ratio_negative if ratio_negative != 0 else float('inf')

def calculate_ratio_value_number_to_sequence_length(self, signal):
        unique_values = len(np.unique(signal))
        return unique_values / len(signal)

def calculate_second_order_difference(self, signal):
        # https://numpy.org/doc/stable/reference/generated/numpy.diff.html
        return np.diff(signal, n=2)
    
def calculate_signal_resultant(self, signal):
        return np.sqrt(np.sum(signal**2))

def calculate_signal_to_noise_ratio(self, signal):
        # https://en.wikipedia.org/wiki/Signal-to-noise_ratio
        # DOI:10.4249/SCHOLARPEDIA.2088
        mean_signal = np.mean(signal)
        std_noise = np.std(signal)
        return mean_signal / std_noise if std_noise > 0 else float('inf')

def calculate_slope_of_linear_fit(self, signal):
        # https://www.mathworks.com/help/matlab/data_analysis/linear-regression.html
        slope, _, _, _, _ = linregress(np.arange(len(signal)), signal)
        return slope

def calculate_smoothing_by_binomial_filter(self, signal):
        # https://www.wavemetrics.com/products/igorpro/dataanalysis/signalprocessing/smoothing
        kernel = np.array([1, 2, 1]) / 4.0
        return convolve(signal, kernel, mode='reflect')

def calculate_stochastic_oscillator_value(self, signal):
        # https://www.investopedia.com/terms/s/stochasticoscillator.asp
        low_min = np.min(signal)
        high_max = np.max(signal)
        current_value = signal[-1]
        return 100 * (current_value - low_min) / (high_max - low_min)

def calculate_sum(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        return np.sum(signal)

def calculate_sum_of_negative_values(self, signal):
        return np.sum(signal[signal < 0])

def calculate_sum_of_positive_values(self, signal):
        return np.sum(signal[signal > 0])

def calculate_sum_of_reoccurring_data_points(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        unique, counts = np.unique(signal, return_counts=True)
        return np.sum(unique[counts > 1])

def calculate_sum_of_reoccurring_values(self, signal):
        # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#
        unique, counts = np.unique(signal, return_counts=True)
        return np.sum(counts[counts > 1])

def calculate_third_quartile(self, signal):
        return np.percentile(signal, 75)

def calculate_variance_of_absolute_differences(self, signal):
        # https://doi.org/10.1080/00031305.2014.994712
        abs_diffs = np.abs(np.diff(signal))
        return np.var(abs_diffs)

def calculate_winsorized_mean(self, signal, limits=[0.05, 0.05]):
        # https://www.investopedia.com/terms/w/winsorized_mean.asp
        return stats.mstats.winsorize(signal, limits=limits).mean()

def calculate_zero_crossing_rate(self, signal):
        # https://librosa.org/doc/0.10.2/generated/librosa.feature.zero_crossing_rate.html#librosa-feature-zero-crossing-rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        return len(zero_crossings) / len(signal)

def calculate_detrended_fluctuation_analysis(self, signal, order=1, minimal = 20):
    """
    Performs detrended fluctuation analysis (DFA) on the given signal.
        
    Parameters:
    ----------
    signal: array
        The input time series.
    order: integer
        The order of the polynomial fit for local detrending default is 1 for linear detrending).
    minimal: integer
        The minimum segment size to consider
        
    Returns:
    -------
    segment_sizes: array
        Array of segment sizes.
    fluctuation_values: array
        Fluctuation function values corresponding to segment sizes.
        
    References:
    ----------
    [1] Bryce, R. M., & Sprague, K. B. (2012). Revisiting detrended 
        fluctuation analysis. Scientific Reports 2012 2:1, 2(1), 1–6. 
        https://doi.org/10.1038/srep00315
    [2] Zhang, H.-Y., Feng, Z.-Q., Feng, S.-Y., & Zhou, Y. (2023). 
        A Survey of Methods for Estimating Hurst Exponent of Time 
        Sequence. https://arxiv.org/abs/2310.19051v1
    """   
    # Calculate the cumulative sum of the mean-shifted signal
    signal_mean = np.mean(signal)
    mean_shifted_signal = signal - signal_mean
    cumulative_sum_signal = np.cumsum(mean_shifted_signal)
        
    N = len(signal)
        
    def Divisors(N, minimal=20):
        D = []
        for i in range(minimal, N // minimal + 1):
            if N % i == 0:
                D.append(i)
        return D
        
    def findOptN(N, minimal=20):
        """
        Find such a natural number OptN that possesses the largest number of
            divisors among all natural numbers in the interval [0.99*N, N]
            """
        N0 = int(0.99 * N)
        # The best length is the one which have more divisiors
        Dcount = [len(Divisors(i, minimal)) for i in range(N0, N + 1)]
        OptN = N0 + Dcount.index(max(Dcount))
        return OptN
        
        
    OptN = findOptN(len(signal), minimal=minimal)
    segment_sizes = Divisors(OptN, minimal=minimal)
    fluctuation_values = []

    for m in segment_sizes:
        k = OptN // m
        Y = np.reshape(cumulative_sum_signal[N - OptN:], [m, k], order='F')
        F = np.copy(Y)
        # t = 1, 2, ..., m
        t = np.linspace(1, m, m)
        for i in range(k):
            p = np.polyfit(t, Y[:, i], 1)
            F[:, i] = Y[:, i] - t * p[0] - p[1]
        fluctuation_values.append(np.mean(np.std(F)))
        
    return segment_sizes, np.array(fluctuation_values)
    

def calculate_hurst_exponent(self, signal):
        """
        References:
        ----------
        [1] Bryce, R. M., & Sprague, K. B. (2012). Revisiting detrended 
            fluctuation analysis. Scientific Reports 2012 2:1, 2(1), 1–6. 
            https://doi.org/10.1038/srep00315
        [2] Zhang, H.-Y., Feng, Z.-Q., Feng, S.-Y., & Zhou, Y. (2023). 
            A Survey of Methods for Estimating Hurst Exponent of Time 
            Sequence. https://arxiv.org/abs/2310.19051v1
        """
        segment_size, fluctuation_values = self.calculate_detrended_fluctuation_analysis(signal)
    
        poly = np.polyfit(np.log(segment_size), np.log(fluctuation_values), 1)
        hurst = poly[0]
        return hurst
    
def calculate_augmented_dickey_fuller_test(self, signal):
        """
        Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity in a given time series signal.

        The ADF test is a statistical test used to determine if a time series is stationary or has a unit root.
        A stationary time series has constant mean and variance over time.

        Parameters:
        ----------
        signal (array-like): 
            The time series data to be tested for stationarity.

        Returns:
        -------
        np.array or float:
                    A numpy array containing the test statistic, p-value, and number of lags used in the test.
                    If the test fails due to an exception, returns NaN.
        Reference:
        ---------
            Christ et al., 2018, https://doi.org/10.1016/J.NEUCOM.2018.03.067
        """
        adf_vals_names = np.array(["teststats", "pvalue", "usedlag"])
        try:
            test_stat, p_value, used_lag, _,_,_ = adfuller(signal)
            adf_vals = np.array([test_stat, p_value, used_lag])
        except:
            return np.nan
        return adf_vals, adf_vals_names

    
    
    
    
    
    
    
def calculate_conditional_entropy(self, signal):
        """
        Calculates the entropy of the signal X, given the entropy of X

        Args:
            signal: np.array
                The input time series.
            
        Returns:
            array: An array containing the conditional entropy of the signal.
            
        Reference:
        ---------
        
        """
        pass