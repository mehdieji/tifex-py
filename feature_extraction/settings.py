import numpy as np
from feature_extraction.extraction import calculate_statistical_features


class StatisticalFeatures:
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
                adjusted = False,
                ssc_threshold = 0,
                ar_model_coefficients_order = 4,
                energy_ratio_chunks = 4,
                mode = 'valid',
                weights = None,
                ema_alpha = 0.3, 
                dfa_order = 1,
                dfa_minimum = 20,
                wm_limits = [0.05, 0.05],
                bins = [2,3,4,10,100],
                count_below_or_above_x = 0,
                cid_ce_normalize = [True, False],
                hist_bins = 10
                ):

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
            
    calculate_statistical_features = calculate_statistical_features