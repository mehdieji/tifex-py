import json
import yaml
import pywt
import numpy as np


class BaseFeatureParams:
    def __init__(self, calculators=None):
        self.calculators = calculators

    @classmethod
    def from_json(cls, file_path):
        """
        Load the settings from a json file. Overwrites the current settings.

        Parameters:
        -----------
        file_path : str
            Path to the file.
        """
        with open(file_path, 'r') as file:
            settings = json.load(file)
        return cls(**settings)

    @classmethod
    def from_yaml(cls, file_path):
        """
        Load the settings from a yaml file. Overwrites the current settings.

        Parameters:
        -----------
        file_path : str
            Path to the file.
        """
        with open(file_path, 'r') as file:
            settings = yaml.safe_load(file)
        return cls(**settings)

    def to_json(self, file_path):
        """
        Save the settings to a json file.

        Parameters:
        -----------
        file_path : str
            Path to the file.
        """
        settings = self.get_settings_as_dict()
        with open(file_path, 'w') as file:
            json.dump(settings, file, indent=4)

    def to_yaml(self, file_path):
        """
        Save the settings to a yaml file.

        Parameters:
        -----------
        file_path : str
            Path to the file.
        """
        settings = self.get_settings_as_dict()
        with open(file_path, 'w') as file:
            yaml.dump(settings, file, default_flow_style=False)

    def get_settings_as_dict(self):
        """
        Get the settings as a dictionary.

        Returns:
        --------
        dict
            Dictionary with the settings.
        """
        d = {key:value for key, value in self.__dict__.items() 
             if not key.startswith('__') 
             and not callable(value) 
             and not callable(getattr(value, "__get__", None))} # <- important
        return d

class StatisticalFeatureParams(BaseFeatureParams):
    """
    Parameters for statistical feature extraction.

    Attributes:
    ----------
    window_size : int
        Size of the window to compute the features.
    n_lags_auto_correlation : int, optional
        Used in calculation of mean_of_auto_corr_lags.
        Number of lags to compute the auto-correlation. The default is None.
    moment_orders : array-like, optional
        Used in calculation of moment_order_*.
        A list or array of integers specifying the orders of the moments to be calculated.
        The default is None.
    trimmed_mean_thresholds : list, optional
        Used in calculation of trimmed_mean_*.
        List of proportions to cut from each end of the signal. The default is None.
    higuchi_k_values : list, optional
        Used in calculation of higuchi_fractal_dimensions_k=*.
        Values of k for the Higuchi Fractal Dimension. The default is None.
    tsallis_q_parameter : int, optional
        Used in calculation of tsallis_entropy.
        The q-parameter of the Tsallis entropy.. The default is 1.
    renyi_alpha_parameter : float, optional
        Used in calculation of renyi_entropy.
        The order alpha of the Rényi entropy. If alpha = 1, the function 
        returns the Shannon entropy. The default is 2.
    permutation_entropy_order : int, optional
        Used in calculation of permutation_entropy.
        The embedding dimension (order), which determines the length of the ordinal
        patterns (permutations) considered. The default is 3.
    permutation_entropy_delay : int or list/array of int, optional
        Used in calculation of permutation_entropy.
        The time delay between consecutive values used to form the ordinal patterns.
        If multiple delays are passed as a list or array, the function will return
        the average permutation entropy across all delays. The default is 1.
    svd_entropy_order : int, optional
        Used in calculation of svd_entropy.
        The embedding dimension or order used to construct the embedding matrix.
        This determines the number of rows in the embedding matrix. The default is 3.
    svd_entropy_delay : int, optional
        Used in calculation of svd_entropy.
        The time delay used in the embedding process. This determines the step size
        between consecutive points in the embedding matrix. The default is 1.
    adjusted : bool, optional
        Used in calculation of mean_abs_deviation.
        If True, returns the adjusted MAD, making it consistent with the standard deviation
        for normally distributed data by multiplying by 1.4826. The default is False.
    ssc_threshold : int, optional
        Used in calculation of no._of_slope_sign_changes.
        The threshold value to determine significant slope changes. The default is 0.
    ar_model_coefficients_order : int, optional
        Used in calculation of autoregressive_model_coefficients_*.
        The number of lags to include in the model. The default is 4.
    energy_ratio_chunks : int, optional
        Used in calculation of energy_ratio_by_chunks_*.
        The number of chunks to divide the signal into. The default is 4.
    mode : str, optional
        Used in calculation of moving_average and weighted_moving_average.
        Options include:
            - 'valid': Returns output of length max(M, N) - min(M, N) + 1. This means no padding is applied and 
            the result is only calculated where the window fully overlaps with the signal.
            - 'same': Returns output of the same length as the input signal. Padding may be applied to ensure the output 
            has the same length as the input.
            - 'full': Returns the convolution at each point of overlap, with padding. The output length is M + N - 1.
        Default is 'valid'.
    weights : list, optional
        The weights to apply to the signal. If not provided, a linearly decreasing set of weights is used.
        The length of weights must be equal to or less than the length of the signal.
        Weights for the energy ratio. The default is None.
    ema_alpha : float, optional
        Used in calculation of exponential_moving_average.
        Alpha for the EMA. The default is 0.3.
    dfa_order : int, optional
        Used in the calculation of the detrended_fluctuation_analysis_segments and detrended_fluctuation_analysis_values.
        The order of the polynomial fit for local detrending default is 1 for linear detrending).
        The default is 1.
    dfa_minimal : int, optional
        Used in the calculation of the detrended_fluctuation_analysis_segments and detrended_fluctuation_analysis_values.
        The minimum segment size to consider. The default is 20.
    wm_limits : list, optional
        Used in the calculation of winsorized_mean.
        The proportions of data to Winsorize from the lower and upper ends of the 
        distribution. The default is [0.05, 0.05], which means 5% of the data is 
        Winsorized from both ends. The values should be between 0 and 0.5.
    entropy_bins : list, optional
        Used in the calculation of binned_entropy.
        If `entropy_bins` is an integer, it defines the number of equal-width bins in the histogram.
        If `entropy_bins` is a sequence, it defines the bin edges, including the rightmost edge.
        The default is 10.
    count_below_or_above_x : int, optional
        Used in the calculation of count_below and count_above.
        Value of interest. The default is 0.
    cid_ce_normalize : list, optional
        Used in the calculation of cid_ce.
        If True, the signal is normalized before computing the complexity estimate. Normalization ensures that 
        the signal has zero mean and unit variance. If the standard deviation is zero (i.e., the signal is constant), 
        the function returns 0.0. The default is True.
    hist_bins : int, optional
        Used in the calculation of histogram_bin_frequencies_*.
        Bins for the histogram. The default is 10.
    q : list, optional
        Used in the calculation of quantile_*.
        Quantiles. The default is [0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9].
    r_sigma: list, optional
        Used in the calculation of ratio_beyond_r_sigma_*.
        The multiplier for the standard deviation to define the threshold. The default is [0.1, 0.2, 0.3, 0.4, 0.5].
    lz_bins : int, optional
        Used in the calculation of lempel_ziv_complexity.
        Bins for the Lempel-Ziv. The default is 10. 
    values: list
        Used in the calculation of value_count_*.
        Values for which their number of occurances will be calculated. The default is [0,1,-1]                                                                                                                                                                                                                                                  
    m: int
        Used in the calculation of approximate_entropy_*.
        Length of compared run of data for approximate entropy.
    r: list
        Used in the calculation of approximate_entropy_*.
        Filtering levels for approximate entropy.
        Length of compared run of data for approximate entropy
    cwt_peaks_n: list
        Upper limits of widths for determining the peaks
    """
    def __init__(self,
                 window_size,
                 n_lags_auto_correlation=None,
                 moment_orders=None,
                 trimmed_mean_thresholds=None,
                 higuchi_k_values=None,
                 tsallis_q_parameter=1,
                 renyi_alpha_parameter=2,
                 permutation_entropy_order=3,
                 permutation_entropy_delay=1,
                 svd_entropy_order=3,
                 svd_entropy_delay=1,
                 adjusted=False,
                 ssc_threshold = 0,
                 ar_model_coefficients_order=4,
                 energy_ratio_chunks=4,
                 mode='valid',
                 weights=None,
                 ema_alpha=0.3, 
                 dfa_order=1,
                 dfa_minimal=20,
                 wm_limits=[0.05, 0.05],
                 entropy_bins=10,
                 count_below_or_above_x=0,
                 cid_ce_normalize=True,
                 hist_bins=10,
                 q=[0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9],
                 r_sigma=[0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10],
                 lz_bins=10,
                 values=[0,1,-1],
                 m=2,
                 r=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                 cwt_peaks_n=[1,5],
                 calculators=None
                ):
        super().__init__(calculators)
        self.window_size = window_size
        self.tsallis_q_parameter = tsallis_q_parameter
        self.renyi_alpha_parameter = renyi_alpha_parameter
        self.permutation_entropy_order = permutation_entropy_order
        self.permutation_entropy_delay = permutation_entropy_delay
        self.svd_entropy_order = svd_entropy_order
        self.svd_entropy_delay = svd_entropy_delay
        self.adjusted = adjusted
        self.ssc_threshold = ssc_threshold
        self.ar_model_coefficients_order = ar_model_coefficients_order
        self.energy_ratio_chunks = energy_ratio_chunks
        self.mode = mode
        self.weights = weights
        self.ema_alpha = ema_alpha
        self.dfa_order = dfa_order
        self.dfa_minimal = dfa_minimal
        self.wm_limits = wm_limits
        self.entropy_bins = entropy_bins
        self.count_below_or_above_x = count_below_or_above_x
        self.cid_ce_normalize = cid_ce_normalize
        self.hist_bins = hist_bins
        self.q = q
        self.r_sigma = r_sigma
        self.lz_bins = lz_bins
        self.values= values
        self.m= m
        self.r=r
        self.cwt_peaks_n=cwt_peaks_n

        if n_lags_auto_correlation is None:
            self.n_lags_auto_correlation = int(min(10 * np.log10(window_size), window_size - 1))
        else:
            self.n_lags_auto_correlation = n_lags_auto_correlation

        if moment_orders is None:
            self.moment_orders = [3, 4]
        else:
            self.moment_orders = moment_orders

        if trimmed_mean_thresholds is None:
            self.trimmed_mean_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
        else:
            self.trimmed_mean_thresholds = trimmed_mean_thresholds

        if higuchi_k_values is None:
            self.higuchi_k_values = list({5, 10, 20, window_size // 5})
        else:
            self.higuchi_k_values = list(higuchi_k_values)


class SpectralFeatureParams(BaseFeatureParams):
    """
    Parameters for spectral feature extraction.
    
    Attributes:
    -----------
    fs : int
        Sampling frequency of the signal.
    f_bands : list
        Used in calculation of spectral_contrast_band_* , total_band_power, absolute_band_power_*,
        and relative_band_power_*.
        A list of tuples specifying the frequency bands for which to calculate features.
        The default is [[0.5,4], [4,8], [8,12], [12,30], [30,100]].
    n_dom_freqs : int
        Used in calculation of peak_freq_*.
        The number of dominant frequencies to calculate. The default is 5.
    cumulative_power_thresholds : list, optional
        Used in calculation of edge_freq_thresh_*.
        The thresholds for the cumulative power. The default is [.5, .75, .85, .9, 0.95].
    centroid_orders : list, optional
        Used in calculation of spectral_centroid_order_*.
        The order of the centroid calculation. Default is 1, which calculates the standard spectral centroid
        (mean frequency). Higher orders can be used for other types of spectral centroid calculations.
        The default is [1, 2, 3, 4, 5].
    bandwidth_orders : list, optional
        Used in calculation of spectral_bandwidth_order_*.
        The orders of the bandwidth. The default is [1, 2, 3, 4].
    abs_dev_orders : list, optional
        Used in calculation of spectral_absolute_deviation_order_*.
        The orders of the deviation calculation.  The default is [1, 3].
    flux_orders : list, optional
        Used in calculation of spectral_flux_order_*.
        The orders of the flux calculation. The default is [2].
    thresholds_freq_below : list, optional
        Used in calculation of spectral_cumulative_frequency_below_threshold_*.
        List of values between 0 and 1 indicating the cumulative power threshold (e.g., 0.8 for 80%).
        The default is [0.5, 0.75].
    thresholds_freq_above : list, optional
        Used in calculation of spectral_cumulative_frequency_above_threshold_*.
        List of values between 0 and 1 indicating the cumulative power threshold (e.g., 0.8 for 80%).
        The default is [0.5, 0.75].
    
    """
    def __init__(self,
                 fs,
                 f_bands=[[0.5,4], [4,8], [8,12], [12,30], [30,100]],
                 n_dom_freqs=5,
                 cumulative_power_thresholds=None,
                 centroid_orders=[1, 2, 3, 4, 5],
                 bandwidth_orders=[1, 2, 3, 4],
                 abs_dev_orders=[1, 3],
                 flux_orders=[2],
                 thresholds_freq_below=[0.5, 0.75],
                 thresholds_freq_above=[0.5, 0.75],
                 calculators=None):
        super().__init__(calculators)
        self.fs = fs
        self.f_bands = f_bands
        self.n_dom_freqs = n_dom_freqs
        if cumulative_power_thresholds is None:
            self.cumulative_power_thresholds = [.5, .75, .85, .9, 0.95]
        else:
            self.cumulative_power_thresholds = cumulative_power_thresholds
        self.centroid_orders = centroid_orders
        self.bandwidth_orders = bandwidth_orders
        self.abs_dev_orders = abs_dev_orders
        self.flux_orders = flux_orders
        self.thresholds_freq_below = thresholds_freq_below
        self.thresholds_freq_above = thresholds_freq_above


class TimeFrequencyFeatureParams(BaseFeatureParams):
    """
    Parameters for time-frequency feature extraction.

    Attributes:
    ----------
    window_size : int
        Size of the window to compute the features.
    wavelet : str
        Used in calculation of wavelet features.
    decomposition_level : int, optional
        Used in calculation of wavelet features.
    stft_window : str
        Used in calculation of spectoram and STFT features.
    nperseg : int, optional
        Used in calculation of spectoram and STFT features.
    tkeo_sf_params : StatisticalFeatureParams, optional
        Parameters for the TKEO feature calculation. The default is None.
    wavelet_sf_params : StatisticalFeatureParams, optional
        Parameters for the wavelet feature calculation. The default is None.
    spectogram_sf_params : StatisticalFeatureParams, optional
        Parameters for the spectogram feature calculation. The default is None.
    stft_sf_params : StatisticalFeatureParams, optional
        Parameters for the STFT feature calculation. The default is None.
    """
    def __init__(self,
                 window_size,
                 wavelet="db4",
                 decomposition_level=None,
                 stft_window="hann",
                 nperseg=None,
                 tkeo_sf_params=None,
                 wavelet_sf_params=None,
                 spectogram_sf_params=None,
                 stft_sf_params=None,
                 calculators=None
                ):
        super().__init__(calculators)
        self.window_size = window_size
        self.wavelet = wavelet
        self.stft_window = stft_window
        self.nperseg = nperseg if nperseg else window_size // 4
        if decomposition_level is None:
            wavelet_length = len(pywt.Wavelet(wavelet).dec_lo)
            self.decomposition_level = int(np.round(np.log2(self.window_size/wavelet_length) - 1))
        else:
            self.decomposition_level = decomposition_level
        if tkeo_sf_params:
            self.tkeo_sf_params = tkeo_sf_params
        else:
            self.tkeo_sf_params = StatisticalFeatureParams(window_size)
        if wavelet_sf_params:
            self.wavelet_sf_params = wavelet_sf_params
        else:
            self.wavelet_sf_params = StatisticalFeatureParams(window_size)
        if spectogram_sf_params:
            self.spectogram_sf_params = spectogram_sf_params
        else:
            self.spectogram_sf_params = StatisticalFeatureParams(window_size)
        if stft_sf_params:
            self.stft_sf_params = stft_sf_params
        else:
            self.stft_sf_params = StatisticalFeatureParams(window_size)

    def get_settings_as_dict(self):
        """
        Get the settings as a dictionary.

        Returns:
        --------
        dict
            Dictionary with the settings.
        """
        d = super().get_settings_as_dict()
        d["tkeo_sf_params"] = self.tkeo_sf_params.get_settings_as_dict()
        d["wavelet_sf_params"] = self.wavelet_sf_params.get_settings_as_dict()
        d["spectogram_sf_params"] = self.spectogram_sf_params.get_settings_as_dict()
        d["stft_sf_params"] = self.stft_sf_params.get_settings_as_dict()
        return d

    @classmethod
    def from_json(cls, file_path):
        """
        Load the settings from a json file. Overwrites the current settings. Loads
        the statistical feature parameters for each time frequency feature.

        Parameters:
        -----------
        file_path : str
            Path to the file.

        Returns:
        --------
        TimeFrequencyFeatureParams
            TimeFrequencyFeatureParams instance.
        """
        with open(file_path, 'r') as file:
            settings = json.load(file)
        settings["tkeo_sf_params"] = StatisticalFeatureParams(**settings["tkeo_sf_params"])
        settings["wavelet_sf_params"] = StatisticalFeatureParams(**settings["wavelet_sf_params"])
        settings["spectogram_sf_params"] = StatisticalFeatureParams(**settings["spectogram_sf_params"])
        settings["stft_sf_params"] = StatisticalFeatureParams(**settings["stft_sf_params"])
        return TimeFrequencyFeatureParams(**settings)

    @classmethod
    def from_yaml(cls, file_path):
        """
        Load the settings from a yaml file. Overwrites the current settings. Loads
        the statistical feature parameters for each time frequency feature.

        Parameters:
        -----------
        file_path : str
            Path to the file.
        
        Returns:
        --------
        TimeFrequencyFeatureParams
            TimeFrequencyFeatureParams instance.
        """
        with open(file_path, 'r') as file:
            settings = yaml.safe_load(file)
        settings["tkeo_sf_params"] = StatisticalFeatureParams(**settings["tkeo_sf_params"])
        settings["wavelet_sf_params"] = StatisticalFeatureParams(**settings["wavelet_sf_params"])
        settings["spectogram_sf_params"] = StatisticalFeatureParams(**settings["spectogram_sf_params"])
        settings["stft_sf_params"] = StatisticalFeatureParams(**settings["stft_sf_params"])
        return TimeFrequencyFeatureParams(**settings)
