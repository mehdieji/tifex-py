import numpy as np
from scipy.stats import skew, kurtosis, moment, gmean, hmean, trim_mean
from statsmodels.tsa.stattools import acf


class StatisticalFeatures:
    def __init__(self,
                 window_size,
                 n_lags_auto_correlation=None,
                 moment_orders=None,
                 trimmed_mean_thresholds=None,
                 higuchi_k_values=None,
                 tsallis_q_parameer=1,
                 renyi_alpha_parameter=2,
                 permutation_entropy_order=3,
                 permutation_entropy_delay=1,
                 svd_entropy_order=3,
                 svd_entropy_delay=1,
                 ):

        self.window_size = window_size
        self.tsallis_q_parameer = tsallis_q_parameer
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
        # Harmonic mean
        feats.extend(self.calculate_harmonic_mean(signal))
        feats_names.append(f"{signal_name}_harmonic_mean")
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

        return np.array(feats), feats_names

    def calculate_mean(self, signal):
        # Manolopoulos et al., 2001, https://www.researchgate.net/publication/234800113
        return np.array([np.mean(signal)])

    def calculate_geometric_mean(self, signal):
        # Chaddad et al., 2014, DOI: 10.1117/12.2062143
        return np.array([gmean(signal)])

    def calculate_harmonic_mean(self, signal):
        # Chaddad et al., 2014, DOI: 10.1117/12.2062143
        return np.array([hmean(signal)])

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
        # Formula from ETSI ES 202 211 V1.1.1 (2003-11)
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
        crest_factor = np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))
        return np.array([crest_factor])

    def calculate_clearance_factor(self, signal):
        # Formula from The MathWorks Inc., 2022, Available: [Signal Features](https://www.mathworks.com)
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
            #
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
            # Generate permutations using sorting and hashing approach
            order_range = range(order)
            hash_multiplier = np.power(order, order_range)

            # Sort indices to create permutations
            sorted_indices = embedded_signal.argsort(kind="quicksort")
            hash_values = (np.multiply(sorted_indices, hash_multiplier)).sum(1)
            
            # Compute probabilities of each permutation
            _, counts = np.unique(hash_values, return_counts=True)
            probabilities = np.true_divide(counts, counts.sum())

            # Calculate permutation entropy
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

            log_values = np.zeros(normalized_singular_values.shape)
            log_values[normalized_singular_values < 0] = np.nan
            valid = normalized_singular_values > 0
            log_values[valid] = normalized_singular_values[valid] * np.log2(normalized_singular_values[valid])
            #Calculation of the normalized svd entropy
            svd_entropy_value = - log_values.sum()/np.log2(len(normalized_singular_values))
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
            if self.tsallis_q_parameer == 1:
                # Return the Boltzmann–Gibbs entropy
                return np.array([-sum([p * (0 if p == 0 else np.log(p)) for p in hist])])
            else:
                return np.array([(1 - sum([p ** self.tsallis_q_parameer for p in hist])) / (self.tsallis_q_parameer - 1)])
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
