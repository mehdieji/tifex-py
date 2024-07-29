import numpy as np
from scipy.stats import skew, kurtosis, moment, gmean, hmean, trim_mean, entropy, linregress, mode, pearsonr
from statsmodels.tsa.stattools import acf
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
from scipy import integrate
from scipy.signal import detrend, argrelextrema, find_peaks
from itertools import groupby
from scipy.ndimage.filters import convolve
from scipy import stats



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
                 ):

        self.window_size = window_size
        self.tsallis_q_parameter = tsallis_q_parameter
        self.renyi_alpha_parameter = renyi_alpha_parameter
        self.permutation_entropy_order = permutation_entropy_order
        self.permutation_entropy_delay = permutation_entropy_delay
        self.svd_entropy_order = svd_entropy_order
        self.svd_entropy_delay = svd_entropy_delay

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

    def calculate_statistical_features(self, signal, signal_name):
        # A list for storing all the features
        feats = []
        # A list for storing feature names
        feats_names = []

        # Mean of the signal
        feats.extend(self.calculate_mean(signal))
        feats_names.append(f"{signal_name}_mean")
        # Geometric mean
        feats.extend(self.calculate_geometric_mean(signal))
        feats_names.append(f"{signal_name}_geometric_mean")
        # Trimmed mean
        feats.extend(self.calculate_trimmed_mean(signal))
        for threshold in self.trimmed_mean_thresholds:
            feats_names.append(f"{signal_name}_trimmed_mean_thresh_{str(threshold)}")
        # Mean of the absolute signal
        feats.extend(self.calculate_mean_abs(signal))
        feats_names.append(f"{signal_name}_mean_of_abs")
        # Geometric mean of the absolute signal
        feats.extend(self.calculate_geometric_mean_abs(signal))
        feats_names.append(f"{signal_name}_geometric_mean_of_abs")
        # Harmonic mean of the absolute signal
        feats.extend(self.calculate_harmonic_mean_abs(signal))
        feats_names.append(f"{signal_name}_harmonic_mean_of_abs")
        # Trimmed mean of the absolute signal
        feats.extend(self.calculate_trimmed_mean_abs(signal))
        for threshold in self.trimmed_mean_thresholds:
            feats_names.append(f"{signal_name}_trimmed_mean_of_abs_thresh_{str(threshold)}")
        # Standard deviation
        feats.extend(self.calculate_std(signal))
        feats_names.append(f"{signal_name}_std")
        # Standard deviation of the absolute signal
        feats.extend(self.calculate_std_abs(signal))
        feats_names.append(f"{signal_name}_std_of_abs")
        # Skewness
        feats.extend(self.calculate_skewness(signal))
        feats_names.append(f"{signal_name}_skewness")
        # Skewness of the absolute signal
        feats.extend(self.calculate_skewness_abs(signal))
        feats_names.append(f"{signal_name}_skewness_of_abs")
        # Kurtosis
        feats.extend(self.calculate_kurtosis(signal))
        feats_names.append(f"{signal_name}_kurtosis")
        # Kurtosis of the absolute signal
        feats.extend(self.calculate_kurtosis_abs(signal))
        feats_names.append(f"{signal_name}_kurtosis_of_abs")
        # Median
        feats.extend(self.calculate_median(signal))
        feats_names.append(f"{signal_name}_median")
        # Median of the absolute signal
        feats.extend(self.calculate_median_abs(signal))
        feats_names.append(f"{signal_name}_median_of_abs")
        # Maximum value
        feats.extend(self.calculate_max(signal))
        feats_names.append(f"{signal_name}_max")
        # Maximum value of the absolute signal
        feats.extend(self.calculate_max_abs(signal))
        feats_names.append(f"{signal_name}_max_of_abs")
        # Minimum value
        feats.extend(self.calculate_min(signal))
        feats_names.append(f"{signal_name}_min")
        # Minimum value of the absolute signal
        feats.extend(self.calculate_min_abs(signal))
        feats_names.append(f"{signal_name}_min_of_abs")
        # Range of the amplitude
        feats.extend(self.calculate_range(signal))
        feats_names.append(f"{signal_name}_range")
        # Range of the amplitude for the absolute signal
        feats.extend(self.calculate_range_abs(signal))
        feats_names.append(f"{signal_name}_range_of_abs")
        # Variance
        feats.extend(self.calculate_variance(signal))
        feats_names.append(f"{signal_name}_var")
        # Variance of the absolute signal
        feats.extend(self.calculate_variance_abs(signal))
        feats_names.append(f"{signal_name}_var_of_abs")
        # Coefficient of variation
        feats.extend(self.calculate_coefficient_of_variation(signal))
        feats_names.append(f"{signal_name}_coefficient_of_variation")
        # Inter-quartile range
        feats.extend(self.calculate_interquartile_range(signal))
        feats_names.append(f"{signal_name}_iqr")
        # Mean absolute deviation
        feats.extend(self.calculate_mean_absolute_deviation(signal))
        feats_names.append(f"{signal_name}_mean_abs_deviation")
        # Root-mean-square
        feats.extend(self.calculate_root_mean_square(signal))
        feats_names.append(f"{signal_name}_rms")
        # Energy
        feats.extend(self.calculate_signal_energy(signal))
        feats_names.append(f"{signal_name}_energy")
        # Log of energy
        feats.extend(self.calculate_log_energy(signal))
        feats_names.append(f"{signal_name}_log_of_energy")
        # Entropy
        feats.extend(self.calculate_entropy(signal))
        feats_names.append(f"{signal_name}_entropy")
        # Number of zero-crossings
        feats.extend(self.calculate_zero_crossings(signal))
        feats_names.append(f"{signal_name}_no._of_zero_crossings")
        # Crest factor
        feats.extend(self.calculate_crest_factor(signal))
        feats_names.append(f"{signal_name}_crest_factor")
        # Clearance factor
        feats.extend(self.calculate_clearance_factor(signal))
        feats_names.append(f"{signal_name}_clearance_factor")
        # Shape factor
        feats.extend(self.calculate_shape_factor(signal))
        feats_names.append(f"{signal_name}_shape_factor")
        # Number of mean-crossings
        feats.extend(self.calculate_mean_crossing(signal))
        feats_names.append(f"{signal_name}_no._of_mean_crossings")
        # Impulse factor
        feats.extend(self.calculate_impulse_factor(signal))
        feats_names.append(f"{signal_name}_impulse_factor")
        # Auto-correlation
        feats.extend(self.calculate_mean_auto_correlation(signal))
        feats_names.append(f"{signal_name}_mean_of_auto_corr_lag_1_to_{self.n_lags_auto_correlation}")
        # High-order moments
        feats.extend(self.calculate_higher_order_moments(signal))
        for order in self.moment_orders:
            feats_names.append(f"{signal_name}_moment_order_{order}")
        # Median absolute deviation
        feats.extend(self.calculate_median_absolute_deviation(signal))
        feats_names.append(f"{signal_name}_median_abs_deviation")
        # Magnitude area
        feats.extend(self.calculate_signal_magnitude_area(signal))
        feats_names.append(f"{signal_name}_magnitude-area")
        # Average amplitude change
        feats.extend(self.calculate_avg_amplitude_change(signal))
        feats_names.append(f"{signal_name}_avg_amplitude_change")
        # Number of slope sign changes
        feats.extend(self.calculate_slope_sign_change(signal))
        feats_names.append(f"{signal_name}_no._of_slope_sign_changes")
        # Higuchi Fractal Dimensions
        feats.extend(self.calculate_higuchi_fractal_dimensions(signal))
        for k in self.higuchi_k_values:
            feats_names.append(f"{signal_name}_higuchi_fractal_dimensions_k={k}")
        # Permutation entropy
        feats.extend(self.calculate_permutation_entropy(signal))
        feats_names.append(f"{signal_name}_permutation_entropy")
        # Singular Value Decomposition entropy
        feats.extend(self.calculate_svd_entropy(signal))
        feats_names.append(f"{signal_name}_svd_entropy")
        # Hjorth parameters
        feats.extend(self.calculate_hjorth_mobility_and_complexity(signal))
        feats_names.append(f"{signal_name}_hjorth_mobility")
        feats_names.append(f"{signal_name}_hjorth_complexity")
        # Cardinality
        feats.extend(self.calculate_cardinality(signal))
        feats_names.append(f"{signal_name}_cardinality")
        # RMS to mean absolute ratio
        feats.extend(self.calculate_rms_to_mean_abs(signal))
        feats_names.append(f"{signal_name}_rms_to_mean_of_abs")
        # Tsallis entropy
        feats.extend(self.calculate_tsallis_entropy(signal))
        feats_names.append(f"{signal_name}_tsallis_entropy")
        # Renyi entropy
        feats.extend(self.calculate_renyi_entropy(signal))
        feats_names.append(f"{signal_name}_renyi_entropy")

        # Absolute Energy
        energy = self.calculate_absolute_energy(signal)
        feats.append(energy)
        feats_names.append(f"{signal_name}_absolute_energy")

        # Approximate Entropy
        app_entropy = self.calculate_approximate_entropy(signal)
        feats.append(app_entropy)
        feats_names.append(f"{signal_name}_approximate_entropy")

        # Area Under the Curve
        area = self.calculate_area_under_curve(signal)
        feats.append(area)
        feats_names.append(f"{signal_name}_area_under_curve")

        # Area Under the Squared Curve
        area_squared = self.calculate_area_under_squared_curve(signal)
        feats.append(area_squared)
        feats_names.append(f"{signal_name}_area_under_squared_curve")

        # Autoregressive Model Coefficients
        ar_coeffs = self.calculate_autoregressive_model_coefficients(signal)
        for i, coeff in enumerate(ar_coeffs):
            feats.append(coeff)
            feats_names.append(f"{signal_name}_ar_coefficient_{i+1}")

        # Count
        count = self.calculate_count(signal)
        feats.append(count)
        feats_names.append(f"{signal_name}_count")

        # Count Above Mean
        count_above_mean = self.calculate_count_above_mean(signal)
        feats.append(count_above_mean)
        feats_names.append(f"{signal_name}_count_above_mean")

        # Count Below Mean
        count_below_mean = self.calculate_count_below_mean(signal)
        feats.append(count_below_mean)
        feats_names.append(f"{signal_name}_count_below_mean")

        # Count of Negative Values
        count_of_negative_values = self.calculate_count_of_negative_values(signal)
        feats.append(count_of_negative_values)
        feats_names.append(f"{signal_name}_count_of_negative_values")

        # Count of Positive Values
        count_pos = self.calculate_count_of_positive_values(signal)
        feats.append(count_pos)
        feats_names.append(f"{signal_name}_count_of_positive_values")

        # Covariance with a shifted version of the signal (for demonstration)
        cov = self.calculate_covariance(signal, np.roll(signal, 1))
        feats.append(cov)
        feats_names.append(f"{signal_name}_covariance")

        # Cumulative Energy
        cum_energy = self.calculate_cumulative_energy(signal)
        feats.append(cum_energy)
        feats_names.append(f"{signal_name}_cumulative_energy")

        # Cumulative Sum
        cum_sum = self.calculate_cumulative_sum(signal)
        feats.append(cum_sum)
        feats_names.append(f"{signal_name}_cumulative_sum")

        # Differential Entropy
        diff_entropy = self.calculate_differential_entropy(signal)
        feats.append(diff_entropy)
        feats_names.append(f"{signal_name}_differential_entropy")

        # Energy Ratio by Chunks
        energy_ratio = self.calculate_energy_ratio_by_chunks(signal)
        for i, ratio in enumerate(energy_ratio):
            feats.append(ratio)
            feats_names.append(f"{signal_name}_energy_ratio_chunk_{i+1}")

        # Exponential Moving Average
        ema = self.calculate_exponential_moving_average(signal)
        feats.append(ema)
        feats_names.append(f"{signal_name}_exponential_moving_average")

        # First Location of Maximum
        first_max = self.calculate_first_location_of_maximum(signal)
        feats.append(first_max)
        feats_names.append(f"{signal_name}_first_location_of_maximum")

        # First Location of Minimum
        first_min = self.calculate_first_location_of_minimum(signal)
        feats.append(first_min)
        feats_names.append(f"{signal_name}_first_location_of_minimum")

        # First Order Difference
        first_diff = self.calculate_first_order_difference(signal)
        feats.append(first_diff[-1])  # Assuming we want the last first order difference
        feats_names.append(f"{signal_name}_first_order_difference")

        # First Quartile
        first_quartile = self.calculate_first_quartile(signal)
        feats.append(first_quartile)
        feats_names.append(f"{signal_name}_first_quartile")

        # Fisher Information
        fisher_info = self.calculate_fisher_information(signal)
        feats.append(fisher_info)
        feats_names.append(f"{signal_name}_fisher_information")

        # Histogram Bin Frequencies
        hist_bins = self.calculate_histogram_bin_frequencies(signal)
        for i, bin_count in enumerate(hist_bins):
            feats.append(bin_count)
            feats_names.append(f"{signal_name}_histogram_bin_{i}")

        # Intercept of Linear Fit
        intercept = self.calculate_intercept_of_linear_fit(signal)
        feats.append(intercept)
        feats_names.append(f"{signal_name}_intercept_of_linear_fit")

        # Katz Fractal Dimension
        katz_fd = self.calculate_katz_fractal_dimension(signal)
        feats.append(katz_fd)
        feats_names.append(f"{signal_name}_katz_fractal_dimension")

        # Last Location of Maximum
        last_max = self.calculate_last_location_of_maximum(signal)
        feats.append(last_max)
        feats_names.append(f"{signal_name}_last_location_of_maximum")

        # Last Location of Minimum
        last_min = self.calculate_last_location_of_minimum(signal)
        feats.append(last_min)
        feats_names.append(f"{signal_name}_last_location_of_minimum")

        # Linear Trend with Full Linear Regression Results
        lin_trend_results = self.calculate_linear_trend_with_full_linear_regression_results(signal)
        feats.extend(lin_trend_results)
        feats_names.extend([f"{signal_name}_slope", f"{signal_name}_intercept", f"{signal_name}_r_squared", f"{signal_name}_p_value", f"{signal_name}_std_err"])

        # Local Maxima and Minima
        num_maxima, num_minima = self.calculate_local_maxima_and_minima(signal)
        feats.append(num_maxima)
        feats_names.append(f"{signal_name}_local_maxima")
        feats.append(num_minima)
        feats_names.append(f"{signal_name}_local_minima")

        # Log Return
        log_return = self.calculate_log_return(signal)
        feats.append(log_return)
        feats_names.append(f"{signal_name}_log_return")

        # Longest Strike Above Mean
        longest_above = self.calculate_longest_strike_above_mean(signal)
        feats.append(longest_above)
        feats_names.append(f"{signal_name}_longest_strike_above_mean")

        # Longest Strike Below Mean
        longest_below = self.calculate_longest_strike_below_mean(signal)
        feats.append(longest_below)
        feats_names.append(f"{signal_name}_longest_strike_below_mean")

        # Lower Complete Moment
        lower_moment = self.calculate_lower_complete_moment(signal)
        feats.append(lower_moment)
        feats_names.append(f"{signal_name}_lower_complete_moment")

        # Mean Absolute Change
        mean_abs_change = self.calculate_mean_absolute_change(signal)
        feats.append(mean_abs_change)
        feats_names.append(f"{signal_name}_mean_absolute_change")

        # Mean Crossings
        mean_crossings = self.calculate_mean_crossings(signal)
        feats.append(mean_crossings)
        feats_names.append(f"{signal_name}_mean_crossings")

        # Mean Relative Change
        mean_rel_change = self.calculate_mean_relative_change(signal)
        feats.append(mean_rel_change)
        feats_names.append(f"{signal_name}_mean_relative_change")

        # Mean Second Derivative Central
        mean_sec_deriv = self.calculate_mean_second_derivative_central(signal)
        feats.append(mean_sec_deriv)
        feats_names.append(f"{signal_name}_mean_second_derivative_central")

        # Median Second Derivative Central
        median_second_derivative = self.calculate_median_second_derivative_central(signal)
        feats.append(median_second_derivative)
        feats_names.append(f"{signal_name}_median_second_derivative_central")

        # Mode
        signal_mode = self.calculate_mode(signal)
        feats.append(signal_mode)
        feats_names.append(f"{signal_name}_mode")

        # Moving Average
        moving_average = self.calculate_moving_average(signal, window_size=10)  # Example window size
        feats.append(moving_average)
        feats_names.append(f"{signal_name}_moving_average")

        # Number of Inflection Points
        inflection_points = self.calculate_number_of_inflection_points(signal)
        feats.append(inflection_points)
        feats_names.append(f"{signal_name}_number_of_inflection_points")

        # Peak to Peak Distance
        peak_to_peak_distance = self.calculate_peak_to_peak_distance(signal)
        feats.append(peak_to_peak_distance)
        feats_names.append(f"{signal_name}_peak_to_peak_distance")

        # Pearson Correlation Coefficient
        pearson_correlation = self.calculate_pearson_correlation_coefficient(signal)
        feats.append(pearson_correlation)
        feats_names.append(f"{signal_name}_pearson_correlation_coefficient")

        # Percentage of Negative Values
        percentage_negative = self.calculate_percentage_of_negative_values(signal)
        feats.append(percentage_negative)
        feats_names.append(f"{signal_name}_percentage_of_negative_values")

        # Percentage of Positive Values
        percentage_positive = self.calculate_percentage_of_positive_values(signal)
        feats.append(percentage_positive)
        feats_names.append(f"{signal_name}_percentage_of_positive_values")

        # Percentage of Reoccurring Datapoints to All Datapoints
        percentage_reoccurring_datapoints = self.calculate_percentage_of_reoccurring_datapoints_to_all_datapoints(signal)
        feats.append(percentage_reoccurring_datapoints)
        feats_names.append(f"{signal_name}_percentage_of_reoccurring_datapoints_to_all_datapoints")

        # Percentage of Reoccurring Values to All Values
        percentage_reoccurring_values = self.calculate_percentage_of_reoccurring_values_to_all_values(signal)
        feats.append(percentage_reoccurring_values)
        feats_names.append(f"{signal_name}_percentage_of_reoccurring_values_to_all_values")

        # Percentile
        percentiles = self.calculate_percentile(signal)
        for i, perc in enumerate([25, 50, 75]):
            feats.append(percentiles[i])
            feats_names.append(f"{signal_name}_percentile_{perc}")

        # Petrosian Fractal Dimension
        pfd = self.calculate_petrosian_fractal_dimension(signal)
        feats.append(pfd)
        feats_names.append(f"{signal_name}_petrosian_fractal_dimension")

        # Ratio Beyond r Sigma
        rbs = self.calculate_ratio_beyond_r_sigma(signal)
        feats.append(rbs)
        feats_names.append(f"{signal_name}_ratio_beyond_r_sigma")

        # Ratio of Fluctuations
        ratio_positive, ratio_negative, ratio_pn = self.calculate_ratio_of_fluctuations(signal)
        feats.append(ratio_positive)
        feats_names.append(f"{signal_name}_ratio_of_positive_fluctuations")
        feats.append(ratio_negative)
        feats_names.append(f"{signal_name}_ratio_of_negative_fluctuations")
        feats.append(ratio_pn)
        feats_names.append(f"{signal_name}_ratio_of_positive_to_negative_fluctuations")

        # Ratio Value Number to Sequence Length
        rvnsl = self.calculate_ratio_value_number_to_sequence_length(signal)
        feats.append(rvnsl)
        feats_names.append(f"{signal_name}_ratio_value_number_to_sequence_length")


        # Sample Entropy
        samp_ent = self.calculate_sample_entropy(signal)
        feats.append(samp_ent)
        feats_names.append(f"{signal_name}_sample_entropy")

        # Second Order Difference
        second_diff = self.calculate_second_order_difference(signal)
        feats.append(second_diff[-1])
        feats_names.append(f"{signal_name}_second_order_difference")

        # Signal Resultant
        signal_resultant = self.calculate_signal_resultant(signal)
        feats.append(signal_resultant)
        feats_names.append(f"{signal_name}_signal_resultant")

        # Signal to Noise Ratio
        snr = self.calculate_signal_to_noise_ratio(signal)
        feats.append(snr)
        feats_names.append(f"{signal_name}_signal_to_noise_ratio")

        # Slope of Linear Fit
        slope = self.calculate_slope_of_linear_fit(signal)
        feats.append(slope)
        feats_names.append(f"{signal_name}_slope_of_linear_fit")

        # Smoothing by Binomial Filter
        smoothed_signal = self.calculate_smoothing_by_binomial_filter(signal)
        feats.append(smoothed_signal[-1])  # Assuming last value after smoothing is of interest
        feats_names.append(f"{signal_name}_smoothed_signal_last_value")

        # Stochastic Oscillator Value
        stochastic_value = self.calculate_stochastic_oscillator_value(signal)
        feats.append(stochastic_value)
        feats_names.append(f"{signal_name}_stochastic_oscillator_value")

        # Sum
        total_sum = self.calculate_sum(signal)
        feats.append(total_sum)
        feats_names.append(f"{signal_name}_total_sum")

        # Sum of Negative Values
        sum_negatives = self.calculate_sum_of_negative_values(signal)
        feats.append(sum_negatives)
        feats_names.append(f"{signal_name}_sum_of_negative_values")

        # Sum of Positive Values
        sum_positives = self.calculate_sum_of_positive_values(signal)
        feats.append(sum_positives)
        feats_names.append(f"{signal_name}_sum_of_positive_values")

        # Sum of Reoccurring Data Points
        sum_reoccurring_data_points = self.calculate_sum_of_reoccurring_data_points(signal)
        feats.append(sum_reoccurring_data_points)
        feats_names.append(f"{signal_name}_sum_of_reoccurring_data_points")

        # Sum of Reoccurring Values
        sum_reoccurring_values = self.calculate_sum_of_reoccurring_values(signal)
        feats.append(sum_reoccurring_values)
        feats_names.append(f"{signal_name}_sum_of_reoccurring_values")

        # Third Quartile
        third_quartile = self.calculate_third_quartile(signal)
        feats.append(third_quartile)
        feats_names.append(f"{signal_name}_third_quartile")

        # Variance of Absolute Differences
        variance_abs_diffs = self.calculate_variance_of_absolute_differences(signal)
        feats.append(variance_abs_diffs)
        feats_names.append(f"{signal_name}_variance_of_absolute_differences")

        # Weighted Moving Average
        weighted_ma = self.calculate_weighted_moving_average(signal)
        feats.append(weighted_ma[-1])  # Assuming last value of WMA is of interest
        feats_names.append(f"{signal_name}_weighted_moving_average")

        # Winsorized Mean
        winsorized_mean = self.calculate_winsorized_mean(signal)
        feats.append(winsorized_mean)
        feats_names.append(f"{signal_name}_winsorized_mean")

        # Zero Crossing Rate
        zero_crossing_rate = self.calculate_zero_crossing_rate(signal)
        feats.append(zero_crossing_rate)
        feats_names.append(f"{signal_name}_zero_crossing_rate")

        # return np.array(feats), feats_names
        return feats, feats_names

    def calculate_mean(self, signal):
        # Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
        return np.array([np.mean(signal)])

    def calculate_geometric_mean(self, signal):
        # Chaddad et al., 2014, DOI: 10.1117/12.2062143
        return np.array([gmean(signal)])

    def calculate_harmonic_mean(self, signal):
        # Chaddad et al., 2014, DOI: 10.1117/12.2062143
        
        # Filter out non-positive values
        positive_signal = signal[signal > 0]
        
        return np.array([hmean(positive_signal)])

    def calculate_trimmed_mean(self, signal):
        # Chaddad et al., 2014, DOI: 10.1117/12.2062143
        feats = []
        for proportiontocut in self.trimmed_mean_thresholds:
            feats.append(trim_mean(signal, proportiontocut=proportiontocut))
        return np.array(feats)

    def calculate_mean_abs(self, signal):
        # Myroniv et al., 2017, https://www.researchgate.net/publication/323935725
        # Phinyomark et al., 2012, DOI: 10.1016/j.eswa.2012.01.102
        # Purushothaman et al., 2018, DOI: 10.1007/s13246-018-0646-7
        return np.array([np.mean(np.abs(signal))])

    def calculate_geometric_mean_abs(self, signal):
        return np.array([gmean(np.abs(signal))])

    def calculate_harmonic_mean_abs(self, signal):
        return np.array([hmean(np.abs(signal))])

    def calculate_trimmed_mean_abs(self, signal):
        feats = []
        for proportiontocut in self.trimmed_mean_thresholds:
            feats.append(trim_mean(np.abs(signal), proportiontocut=proportiontocut))
        return np.array(feats)

    def calculate_std(self, signal):
        # Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
        return np.array([np.std(signal)])

    def calculate_std_abs(self, signal):
        return np.array([np.std(np.abs(signal))])

    def calculate_skewness(self, signal):
        # Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
        # Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        return np.array([skew(signal)])

    def calculate_skewness_abs(self, signal):
        return np.array([skew(np.abs(signal))])

    def calculate_kurtosis(self, signal):
        # Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
        # Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        return np.array([kurtosis(signal)])

    def calculate_kurtosis_abs(self, signal):
        return np.array([kurtosis(np.abs(signal))])

    def calculate_median(self, signal):
        # Banos et al., 2012, DOI: 10.1016/j.eswa.2012.01.164
        return np.array([np.median(signal)])

    def calculate_median_abs(self, signal):
        return np.array([np.median(np.abs(signal))])

    def calculate_min(self, signal):
        # 18th International Conference on Computer Communications and Networks, DOI: 10.1109/ICCCN15201.2009
        min_val = np.min(signal)
        return np.array([min_val])

    def calculate_min_abs(self, signal):
        min_abs_val = np.min(np.abs(signal))
        return np.array([min_abs_val])

    def calculate_max(self, signal):
        # Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        max_val = np.max(signal)
        return np.array([max_val])

    def calculate_max_abs(self, signal):
        max_abs_val = np.max(np.abs(signal))
        return np.array([max_abs_val])

    def calculate_range(self, signal):
        return np.array([np.max(signal) - np.min(signal)])

    def calculate_range_abs(self, signal):
        abs_signal = np.abs(signal)
        return np.array([np.max(abs_signal) - np.min(abs_signal)])

    def calculate_variance(self, signal):
        # Khorshidtalab et al., 2013 , DOI: 10.1088/0967-3334/34/11/1563
        return np.array([np.var(signal)])

    def calculate_variance_abs(self, signal):
        return np.array([np.var(np.abs(signal))])

    def calculate_interquartile_range(self, signal):
        # Formula from Bedeeuzzaman et al., 2012, DOI: 10.5120/6304-8614
        # Bedeeuzzaman et al., 2012, DOI: 10.5120/6304-8614
        return np.array([np.percentile(signal, 75) - np.percentile(signal, 25)])

    def calculate_mean_absolute_deviation(self, signal):
        # Formula from Khair et al., 2017, DOI: 10.1088/1742-6596/930/1/012002
        return np.array([np.mean(np.abs(signal - np.mean(signal)))])

    def calculate_root_mean_square(self, signal):
        # Formula from Khorshidtalab et al., 2013, DOI: 10.1088/0967-3334/34/11/1563
        # 18th International Conference on Computer Communications and Networks, DOI: 10.1109/ICCCN15201.2009
        return np.array([np.sqrt(np.mean(signal**2))])

    def calculate_signal_energy(self, signal):
        # Formula from Rafiuddin et al., 2011, DOI: 10.1109/MSPCT.2011.6150470
        return np.array([np.sum(signal**2)])

    def calculate_log_energy(self, signal):
        return np.array([np.log(np.sum(signal**2))])

    def calculate_entropy(self, signal):
        try:
            # Calculate the histogram
            hist, _ = np.histogram(signal, bins=self.window_size//2, density=True)
            # Replace zero values with a small epsilon
            epsilon = 1e-10
            hist = np.where(hist > 0, hist, epsilon)
            # Calculate the entropy
            # Guido, 2018, DOI: 10.1016/j.inffus.2017.09.006
            entropy = -np.sum(hist * np.log2(hist))
        except:
            entropy = np.nan
        return np.array([entropy])

    def calculate_zero_crossings(self, signal):
        # Myroniv et al., 2017, https://www.researchgate.net/publication/323935725_Analyzing_User_Emotions_via_Physiology_Signals
        # Sharma et al., 2020, DOI: 10.1016/j.apacoust.2019.107020
        # Purushothaman et al., 2018, DOI: 10.1007/s13246-018-0646-7

        # Compute the difference in signbit (True if number is negative)
        zero_cross_diff = np.diff(np.signbit(signal))
        # Sum the differences to get the number of zero-crossings
        num_zero_crossings = zero_cross_diff.sum()
        return np.array([num_zero_crossings])

    def calculate_crest_factor(self, signal):
        # Formula from Cempel, 1980, DOI: 10.1016/0022-460X(80)90667-7
        # DOI: 10.3390/S150716225
        crest_factor = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))
        return np.array([crest_factor])

    def calculate_clearance_factor(self, signal):
        # Formula from The MathWorks Inc., 2022, Available: [Signal Features](https://www.mathworks.com)
        # DOI: 10.3390/S150716225
        clearance_factor = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal))) ** 2)
        return np.array([clearance_factor])

    def calculate_shape_factor(self, signal):
        # Formula from Cempel, 1980, DOI: 10.1016/0022-460X(80)90667-7
        shape_factor = np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal))
        return np.array([shape_factor])

    def calculate_mean_crossing(self, signal):
        # Formula from Myroniv et al., 2017, https://www.researchgate.net/publication/323935725
        mean_value = np.mean(signal)
        mean_crossings = np.where(np.diff(np.sign(signal - mean_value)))[0]
        return np.array([len(mean_crossings)])

    def calculate_impulse_factor(self, signal):
        # Formula from Cempel, 1980, DOI: 10.1016/0022-460X(80)90667-7
        impulse_factor = np.max(np.abs(signal)) / np.mean(np.abs(signal))
        return np.array([impulse_factor])

    def calculate_mean_auto_correlation(self, signal):
        # Fulcher, 2017, DOI: 10.48550/arXiv.1709.08055
        # Banos et al., 2012, DOI: 10.1016/j.eswa.2012.01.164
        auto_correlation_values = acf(signal, nlags=self.n_lags_auto_correlation)[1:]
        return np.array([np.mean(auto_correlation_values)])

    def calculate_higher_order_moments(self, signal):
        feats = []
        for order in self.moment_orders:
            feats.append(moment(signal, moment=order))
        return np.array(feats)

    def calculate_coefficient_of_variation(self, signal):
        # Formula from Jalilibal et al., 2021, DOI: 10.1016/j.cie.2021.107600
        coefficient_of_variation = np.std(signal) / np.mean(signal)
        return np.array([coefficient_of_variation])

    def calculate_median_absolute_deviation(self, signal):
        # Formula from Pham-Gia, 2001, DOI: 10.1016/S0895-7177(01)00109-1
        # Rafiuddin et al., 2011, DOI: 10.1109/MSPCT.2011.6150470
        median_value = np.median(signal)
        mad = np.median(np.abs(signal - median_value))
        return np.array([mad])

    def calculate_signal_magnitude_area(self, signal):
        # Formula from Khan et al., 2010, DOI: 10.1109/TITB.2010.2051955
        # Formula from Rafiuddin et al., 2011, DOI: 10.1109/MSPCT.2011.6150470
        return np.array([np.sum(np.abs(signal))])

    def calculate_avg_amplitude_change(self, signal):
        # Formula from Phinyomark et al., 2012, DOI: 10.1016/j.eswa.2012.01.102
        avg_amplitude_change = np.mean(np.abs(np.diff(signal)))
        return np.array([avg_amplitude_change])

    def calculate_slope_sign_change(self, signal):
        # Purushothaman et al., 2018, DOI: 10.1007/s13246-018-0646-7
        slope_sign_change = np.count_nonzero(np.abs(np.diff(np.sign(np.diff(signal)))))
        return np.array([slope_sign_change])

    def calculate_higuchi_fractal_dimensions(self, signal):
        # Wanliss et al., 2022, DOI: 10.1007/s11071-022-07353-2
        # Wijayanto et al., 2019, DOI: 10.1109/ICITEED.2019.8929940
        def compute_length_for_interval(data, interval, start_time):
            data_size = data.size
            num_intervals = np.floor((data_size - start_time) / interval).astype(np.int64)
            normalization_factor = (data_size - 1) / (num_intervals * interval)
            sum_difference = np.sum(np.abs(np.diff(data[start_time::interval], n=1)))
            length_for_time = (sum_difference * normalization_factor) / interval
            return length_for_time

        def compute_average_length(data, interval):
            compute_length_series = np.frompyfunc(
                lambda start_time: compute_length_for_interval(data, interval, start_time), 1, 1)
            average_length = np.average(compute_length_series(np.arange(1, interval + 1)))
            return average_length

        feats = []
        for max_interval in self.higuchi_k_values:
            try:
                compute_average_length_series = np.frompyfunc(lambda interval: compute_average_length(signal, interval), 1, 1)
                interval_range = np.arange(1, max_interval + 1)
                average_lengths = compute_average_length_series(interval_range).astype(np.float64)
                fractal_dimension, _ = - np.polyfit(np.log2(interval_range), np.log2(average_lengths), 1)
            except:
                fractal_dimension = np.nan
            feats.append(fractal_dimension)
        return np.array(feats)

    def calculate_permutation_entropy(self, signal):
        # Bandt et al., 2002, DOI: 10.1103/PhysRevLett.88.174102
        # Zanin et al., 2012, DOI: 10.3390/e14081553
        try:
            order = self.permutation_entropy_order
            delay = self.permutation_entropy_delay

            if order * delay > self.window_size:
                raise ValueError("Error: order * delay should be lower than signal length.")
            if delay < 1:
                raise ValueError("Delay has to be at least 1.")
            if order < 2:
                raise ValueError("Order has to be at least 2.")
            embedding_matrix = np.zeros((order, self.window_size - (order - 1) * delay))
            for i in range(order):
                embedding_matrix[i] = signal[(i * delay): (i * delay + embedding_matrix.shape[1])]
            embedded_signal = embedding_matrix.T

            # Continue with the permutation entropy calculation. If multiple delay are passed, return the average across all
            if isinstance(delay, (list, np.ndarray, range)):
                return np.mean([self.calculate_permutation_entropy(signal) for d in delay])

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

    def calculate_svd_entropy(self, signal):
        # Banerjee et al., 2014, DOI: 10.1016/j.ins.2013.12.029
        # Strydom et al., 2021, DOI: 10.3389/fevo.2021.623141
        try:
            order = self.svd_entropy_order
            delay = self.svd_entropy_delay
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

    def calculate_hjorth_mobility_and_complexity(self, signal):
        # Hjorth, 1970, DOI:10.1016/0013-4694(70)90143-4
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

    def calculate_rms_to_mean_abs(self, signal):
        # Compute rms value
        # Formula from Khorshidtalab et al., 2013, DOI: 10.1088/0967-3334/34/11/1563
        rms_val = np.sqrt(np.mean(signal ** 2))
        # Compute mean absolute value
        # Formula from Khorshidtalab et al., 2013, DOI: 10.1088/0967-3334/34/11/1563
        mean_abs_val = np.mean(np.abs(signal))
        # Compute ratio of RMS value to mean absolute value
        ratio = rms_val / mean_abs_val
        return np.array([ratio])

    def calculate_tsallis_entropy(self, signal):
        try:
            # Calculate the histogram
            hist, _ = np.histogram(signal, bins=self.window_size//2, density=True)
            # Replace zero values with a small epsilon
            epsilon = 1e-10
            hist = np.where(hist > 0, hist, epsilon)
            if self.tsallis_q_parameter == 1:
                # Return the Boltzmann–Gibbs entropy
                return np.array([-sum([p * (0 if p == 0 else np.log(p)) for p in hist])])
            else:
                return np.array([(1 - sum([p ** self.tsallis_q_parameter for p in hist])) / (self.tsallis_q_parameter - 1)])
        except:
            return np.array([np.nan])

    def calculate_renyi_entropy(self, signal):
        # Beadle et al., 2008, DOI: 10.1109/ACSSC.2008.5074715
        try:
            # Calculate the histogram
            hist, _ = np.histogram(signal, bins=self.window_size//2, density=True)
            # Replace zero values with a small epsilon
            epsilon = 1e-10
            hist = np.where(hist > 0, hist, epsilon)

            if self.renyi_alpha_parameter == 1:
                # Return the Shannon entropy
                return np.array([-sum([p * (0 if p == 0 else np.log(p)) for p in hist])])
            else:
                return np.array([(1 / (1 - self.renyi_alpha_parameter)) * np.log(sum([p ** self.renyi_alpha_parameter for p in hist]))])
        except:
            return np.array([np.nan])

    def calculate_absolute_energy(self, signal):
        # https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html
        return np.sum(signal**2)

    def calculate_approximate_entropy(self, signal):
        # https://doi.org/10.3390%2Fe21060541
        count, _ = np.histogram(signal, bins=10, density=True)
        count = count[count > 0]  # Avoid log(0) issue
        return -np.sum(count * np.log(count))

    def calculate_area_under_curve(self, signal):
        # https://www.researchgate.net/publication/324936696_Enhancing_EEG_Signals_Recognition_Using_ROC_Curve
        return simpson(np.abs(signal), dx=1)

    def calculate_area_under_squared_curve(self, signal):
        return simpson(signal**2, dx=1)

    def calculate_autoregressive_model_coefficients(self, signal, order=4):
        # https://doi.org/10.1109/IEMBS.2008.4650379
        model = AutoReg(signal, lags=order, old_names=False)
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

    def calculate_differential_entropy(self, signal):
        # https://www.frontiersin.org/articles/10.3389/fphy.2020.629620/full
        probability, _ = np.histogram(signal, bins=10, density=True)
        probability = probability[probability > 0]
        return entropy(probability)

    def calculate_energy_ratio_by_chunks(self, signal, chunks=4):
        # https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_extraction/feature_calculators.py#L2212
        chunk_size = len(signal) // chunks
        energies = np.array([np.sum(signal[i*chunk_size:(i+1)*chunk_size]**2) for i in range(chunks)])
        total_energy = np.sum(signal**2)
        return energies / total_energy

    def calculate_exponential_moving_average(self, signal, alpha=0.3):
        s = np.zeros_like(signal)
        s[0] = signal[0]
        for i in range(1, len(signal)):
            s[i] = alpha * signal[i] + (1 - alpha) * s[i - 1]
        return s[-1]

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

    def calculate_moving_average(self, signal, window_size=10):
        # https://cyclostationary.blog/2021/05/23/sptk-the-moving-average-filter/
        if len(signal) < window_size:
            return np.nan
        return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

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

    def calculate_sample_entropy(self, signal):
        # https://raphaelvallat.com/antropy/build/html/generated/antropy.sample_entropy.html
        # https://doi.org/10.1109/SCEECS.2012.6184830
        return entropy(np.histogram(signal, bins=10)[0])  # Simplified example

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

    def calculate_weighted_moving_average(self, signal, weights=None):
        # https://www.mathworks.com/help/signal/ug/signal-smoothing.html
        if weights is None:
            weights = np.linspace(1, 0, num=len(signal))
        weights = weights / np.sum(weights)
        return np.convolve(signal, weights, 'valid')

    def calculate_winsorized_mean(self, signal, limits=[0.05, 0.05]):
        # https://www.investopedia.com/terms/w/winsorized_mean.asp
        return stats.mstats.winsorize(signal, limits=limits).mean()

    def calculate_zero_crossing_rate(self, signal):
        # https://librosa.org/doc/0.10.2/generated/librosa.feature.zero_crossing_rate.html#librosa-feature-zero-crossing-rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        return len(zero_crossings) / len(signal)


# features not impletement
# Conditional Entropy
# Detrended Fluctuation Analysis
# Higuchi Fractal Dimension
# Hurst Exponent
