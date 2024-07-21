# proj-adl-classification

## statistical_feature.py
Using GPU to compute statistical features based on PyTorch.  

Also compare the results with the features computed by CPU (Numpy).  

The return is a pd dataframe with columns: 'feature name', 'feature value gpu', 'feature value cpu', and 'time consumption'. 

#### X - Time series
"*": No reference
<br>
"**": Reference but questionable
<br>
"??": Reference not directly linked to ADLs
<br>
"* **": More than one reference and one is questionable
<br>
"$": Implementation is questionable
<br>
"~": Questionable feature

## Statistical Features


| Feature    | Description | Reference |
| -------- | ------- | ------- |
| calculate_mean(X) | Calculates the mean of X |
| calculate_geometric_mean(X) | Calculates the geometric mean of X |
| calculate_trimmed_mean(X) | Calculates the trimmed mean of X |
| calculate_mean_abs(X) | Calculates the mean absolute value of X |
| calculate_geometric_mean_abs(X) | Calculates the geometric mean of the absolute X | * |
| calculate_harmonic_mean_abs(X)| Calculates the harmonic mean of the absolute X| * |
|calculate_trimmed_mean_abs(X)| Calculates the trimmed mean of absolute X| * |
|calculate_std(X) | Calculates the standard deviation of X |
|calculate_std_abs(X) | Calculates the standard deviation of absolute X | * |
|calculate_skewness(X)|Calculates the skewness of X|
|calculate_skewness_abs(X)| Calculate skewness of absolute X|*|
|calculate_kurtosis(X)|Calculates the kurtosis of X|
|calculate_kurtosis_abs(X)|Calculates the kurtosis of absolute X| * |
|calculate_median(X)|Calculates median of X|
|calculate_median_abs|Calculates the median of absolute X|*|
|calculate_min(X)|Calculates the minimum value of X| **|
|calculate_min_abs(X)|Calculates the minimum value of the absolute values of X| *|
|calculate_max(X)|Calculates the maximum value of X|
|calculate_max_abs(X)|Calculates the maximum value of the absolute values of X| *|
|calculate_range(X)|Calculates the range of X| * |
|calculate_range_abs(X)|Calculates the range of absolute X|*|
|calculate_variance(X)|Calculates the variance of X |
|calculate_variance_abs(X)|Calculates the variance of absolute X|*|
|calculate_interquartile_range(X)|Calculates the interquartile range of X| ??|
|calculate_mean_absolute_deviation(X)|Calculates the mean of the absolute deviation of X ** |
|calculate_root_mean_square(X)|Calculates the root mean square of X| * **|
|calculate_signal_energy(X)|Calculates the energy of X|
|calculate_log_energy(X)|Calculates the log of the energy of X| *|
|calculate_entropy(X)|Calculates the entropy of X|
|calculate_zero_crossings(X)|Calculates the number of times X crosses zero|
|calculate_crest_factor(X)|Calculates the crest factor of X|
|calculate_clearance_factor(X)|Calculates the clearance factor of X|
|calculate_shape_factor(X)|Calculates the shape factor of X|
|calculate_mean_crossing(X)|Calculates the number of times X crosses the mean|
|calculate_impulse_factor(X)|Calculates the impulse factor of X|
|calculate_mean_auto_correlation(X)|Calculates the mean of the autocorrelation of X|
|calculate_higher_order_moments(X)|Calculates the higher order moments of X| * |
|calculate_coefficient_of_variation(X)|Calculates the coefficient of X|
|calculate_median_absolute_deviation(X)|Calculates the median deviation of absolute X|
|calculate_signal_magnitude_area(X)|Calculates the magnitude area of X. The sum of the absolute values of X.| $|
|calculate_avg_amplitude_change(X)|Calculates the average wavelength of X|
|calculate_slope_sign_change(X)|Calculates the number of times the slope of X changes sign|
|calculate_higuchi_fractal_dimensions(X)|Uses the Higuchi method to calculate the fractal dimensions of X|
|calculate_permutation_entropy(X)|Calculates the permutation entropy of X|
|calculate_svd_entropy(X)|Singular Value Decomposition|
|calculate_hjorth_mobility_and_complexity(X)|Calculates features for X based on the first and second derivatives of X|
|calculate_cardinality(X)||*|
|calculate_rms_to_mean_abs(X)|Computes the ratio of the RMS value to mean absolute value of X|
|calculate_tsallis_entropy(X)|Tsallis entropy estimates the information X|*|
|calculate_renyi_entropy(X)|Computes the Renyi entropy of X|
|calculate_absolute_energy(X)|Calculates the absolute energy of X|
|calculate_approximate_entropy(X)|Computes the approximate entropy of X|
|calculate_area_under_curve(X)|Calculates the area under the curve of X|
|calculate_area_under_squared_curve(X)|Computed the area under the curve of X squared|*|
|calculate_autoregressive_model_coefficients(X)|Calculates the autoregressive model coefficients of X|
|calculate_count(X)|Returns the number of values in X|*|
|calculate_count_above_mean(X)|Computes the number of values of X above the mean of X|
|calculate_count_below_mean(X)|Computes the number of values of X below the mean of X|
|calculate_count_of_negative_values(X)|Returns the number of values of X that are negative|
|calculate_count_of_positive_values(X)|Returns the number of values of X that are positive|
|calculate_covariance(X,Y)|Calculates the covariance of X and Y|
|calculate_cumulative_energy(X)|Calculates the cumulative energy of X|
|calculate_cumulative_sum(X)|Calculates the cumulative sum along the specified dimension of X|
|calculate_differential_entropy(X)|Computes the differential entropy of X|
|calculate_energy_ratio_by_chunks(X, param)|Calculates the energy of chunk i out of N chunks expressed as a ratio with the sum of squares over the X|
|calculate_exponential_moving_average(X, param)|Calculates the exponential moving average of X|*|
|calculate_first_location_of_maximum(X)|Returns the first location of the maximum value of X|
|calculate_first_location_of_minimum(X)|Returns the first location of the minimum value of X|
|calculate_first_order_difference(X)|Returns the first order differential of X|
|calculate_first_quartile(X)|Computes the first quartile of X|*|
|calculate_fisher_information(X)|Computes the Fisher information of X|
|calculate_histogram_bin_frequencies(X, param)|Returns the histogram bin frequencies of X|
|calculate_intercept_of_linear_fit(X)||~|
|calculate_katz_fractal_dimension(X)|Computes the Katz fractal dimension of X|
|calculate_last_location_of_maximum(X)|Returns the last location of the maximum value of X|
|calculate_last_location_of_minimum(X)|Returns the last location of the minimum value of X|
|calculate_linear_trend_with_full_linear_regression_results(X)||~|
|calculate_local_maxima_and_minima(X)|Calculates the local maxima and minima of X|
|calculate_log_return(X)|Returns the logarithm of the ratio between the last and first values of  which is a measure of the percentage change in X|~|
|calculate_longest_strike_above_mean(X)|Computes the length of the longest consecutive subsequence of X  that is greater than the mean of X|
|calculate_longest_strike_below_mean(X)|Computes the length of the longest consecutive subsequence of X  that is lesser than the mean of X|
|calculate_lower_complete_moment(X)||*|
|calculate_mean_absolute_change(X)||
|calculate_mean_crossings(X)|Calculates the number of times X crossed the mean value|
|calculate_mean_relative_change(X)|Returns the mean relative change of X|
|calculate_mean_second_derivative_central(X)|Returns the mean of the second derivative of X|
|calculate_median_second_derivative_central(X)|Calculates the median of the second derivative of X|*|
|calculate_mode(X)|Returns the mode of X|*|
|calculate_moving_average(X)|Returns the moving average of X|
|calculate_number_of_inflection_points(X)|Computes the number of inflection points in X|
|calculate_peak_to_peak_distance(X)|Calculates the peak-to-peak distance of X|
|calculate_pearson_correlation_coefficient(X)|Calculates the pearson correlation coefficient of X|
|calculate_percentage_of_negative_values(X)|Returns the percentage of negative values of X |*|
|calculate_percentage_of_positive_values(X)|Returns the percentage of positive values of X |
|calculate_percentage_of_reoccurring_datapoints_to_all_datapoints(X)|Returns the percentage of values that occur another time in the time series X|
|calculate_percentage_of_reoccurring_values_to_all_values(X)|Calculates the percentage of values in X that occur more than once|
|calculate_percentile(X)|Calculates the 20th, 50th and 75th percentile of X|*|
|calculate_petrosian_fractal_dimension(X)|Computes the petrosian fractal dimension of X|
|calculate_ratio_beyond_r_sigma(X)|Returns the ratio of values that are more than r * std away from the mean of X|
|||
|||
|||
|||
|||
|||
|||




<br>
<br>
<br>
<br>

## Time Frequency Features
| Feature    | Description | Reference |
| -------- | ------- | ------- |
|extract_wavelet_features(params)||
|extract_spectrogram_features(params)||
|extract_stft_features(params)||
|teager_kaiser_energy_operator(X)||