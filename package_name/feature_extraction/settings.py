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
        Number of lags to compute the auto-correlation. The default is None.
    moment_orders : array-like, optional
        Orders of the moments to compute. The default is None.
    trimmed_mean_thresholds : list, optional
        Thresholds for the trimmed mean. The default is None.
    higuchi_k_values : list, optional
        Values of k for the Higuchi Fractal Dimension. The default is None.
    tsallis_q_parameter : int, optional
        Parameter for the Tsallis entropy. The default is 1.
    renyi_alpha_parameter : float, optional
        Parameter for the Renyi entropy. The default is 2.
    permutation_entropy_order : int, optional
        Order for the permutation entropy. The default is 3.
    permutation_entropy_delay : int, optional
        Delay for the permutation entropy. The default is 1.
    svd_entropy_order : int, optional
        Order for the SVD entropy. The default is 3.
    svd_entropy_delay : int, optional
        Delay for the SVD entropy. The default is 1.
    adjusted : bool, optional
        Adjusted entropy. The default is False.
    ssc_threshold : int, optional
        Threshold for the SSC. The default is 0.
    ar_model_coefficients_order : int, optional
        Order for the AR model coefficients. The default is 4.
    energy_ratio_chunks : int, optional
        Number of chunks for the energy ratio. The default is 4.
    mode : str, optional
        Mode for the energy ratio. The default is 'valid'.
    weights : list, optional
        Weights for the energy ratio. The default is None.
    ema_alpha : float, optional
        Alpha for the EMA. The default is 0.3.
    dfa_order : int, optional
        Order for the DFA. The default is 1.
    dfa_minimum : int, optional
        Minimum for the DFA. The default is 20.
    wm_limits : list, optional
        Limits for the WM. The default is [0.05, 0.05].
    bins : list, optional
        Bins for the histogram. The default is [2,3,4,10,100].
    count_below_or_above_x : int, optional
        Count below or above x. The default is 0.
    cid_ce_normalize : list, optional
        Normalize for the CID CE. The default is [True, False].
    hist_bins : int, optional
        Bins for the histogram. The default is 10.
    q : list, optional
        Quantiles. The default is [0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9].
    r_sigma: list, optional
        STD multiplier. The default is [0.1, 0.2, 0.3, 0.4, 0.5].
    lz_bins : int, optional
        Bins for the Lempel-Ziv. The default is 10. 
    values: list
        Values for which their number of occurances will be calculated. The default is [0,1,-1]                                                                                                                                                                                                                                                  
    m: int
        Length of compared run of data for approximate entropy
    r: list 
        Filtering levels for approximate entropy
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
                 dfa_minimum=20,
                 wm_limits=[0.05, 0.05],
                 bins=[2,3,4,10,100],
                 count_below_or_above_x=0,
                 cid_ce_normalize=[True, False],
                 hist_bins=10,
                 q=[0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9],
                 r_sigma=[1, 2],
                 lz_bins=10,
                 values=[0,1,-1],
                 m=2,
                 r=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
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
        self.dfa_minimum = dfa_minimum
        self.wm_limits = wm_limits
        self.bins = bins
        self.count_below_or_above_x = count_below_or_above_x
        self.cid_ce_normalize = cid_ce_normalize
        self.hist_bins = hist_bins
        self.q = q
        self.r_sigma = r_sigma
        self.lz_bins = lz_bins
        self.values= values
        self.m= m
        self.r=r

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
