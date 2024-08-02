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
"* **": More than one reference and one is questionable
<br>
"~": Further research required on feature
<br>

## Statistical Features


|Number| Feature    | Description | Reference |
| -------- | ------- | ------- | ------- |
1| calculate_mean(X) | Calculates the mean of X |
2| calculate_geometric_mean(X) | Calculates the geometric mean of X |
3| calculate_trimmed_mean(X) | Calculates the trimmed mean of X |
4| calculate_mean_abs(X) | Calculates the mean of the absolute values of X |
5| calculate_geometric_mean_abs(X) | Calculates the geometric mean of the absolute values of X ||
6| calculate_harmonic_mean_abs(X)| Calculates the harmonic mean of the absolute values of X| * |
7|calculate_trimmed_mean_abs(X)| Calculates the trimmed mean of absolute values of X| * |
8|calculate_std(X) | Calculates the standard deviation of X |
9|calculate_std_abs(X) | Calculates the standard deviation of the absolute values of X | * |
10|calculate_skewness(X)|Calculates the skewness of X|
11|calculate_skewness_abs(X)| Calculate skewness of absolute values of X|*|
12|calculate_kurtosis(X)|Calculates the kurtosis of X|
13|calculate_kurtosis_abs(X)|Calculates the kurtosis of the absolute values of X| * |
14|calculate_median(X)|Calculates median of X|
15|calculate_median_abs|Calculates the median of the absolute values of X|*|
16|calculate_min(X)|Calculates the minimum value of X| **|
17|calculate_min_abs(X)|Calculates the minimum value of the absolute values of X| *|
18|calculate_max(X)|Calculates the maximum value of X|
19|calculate_max_abs(X)|Calculates the maximum value of the absolute values of X| *|
20|calculate_range(X)|Calculates the range of X| * |
21|calculate_range_abs(X)|Calculates the range of the absolute values of X|*|
22|calculate_variance(X)|Calculates the variance of X |
23|calculate_variance_abs(X)|Calculates the variance of the absolute values of X|*|
24|calculate_interquartile_range(X)|Calculates the interquartile range of X|
25|calculate_mean_absolute_deviation(X)|Calculates the mean of the absolute deviation of X ** |
26|calculate_root_mean_square(X)|Calculates the root mean square of X| * **|
27|calculate_signal_energy(X)|Calculates the energy of X|
28|calculate_log_energy(X)|Calculates the log of the energy of X||
29|calculate_entropy(X)|Calculates the entropy of X|
30|calculate_zero_crossings(X)|Calculates the number of times X crosses zero|
31|calculate_crest_factor(X)|Calculates the crest factor of X|
32|calculate_clearance_factor(X)|Calculates the clearance factor of X|
33|calculate_shape_factor(X)|Calculates the shape factor of X|
34|calculate_mean_crossing(X)|Calculates the number of times X crosses the mean|
35|calculate_impulse_factor(X)|Calculates the impulse factor of X|
36|calculate_mean_auto_correlation(X)|Calculates the mean of the autocorrelation of X|
37|calculate_higher_order_moments(X)|Calculates the higher order moments of X| * |
38|calculate_coefficient_of_variation(X)|Calculates the coefficient of X|
39|calculate_median_absolute_deviation(X)|Calculates the median deviation of absolute X|
40|calculate_signal_magnitude_area(X)|Calculates the magnitude area of X. The sum of the absolute values of X|
41|calculate_avg_amplitude_change(X)|Calculates the average wavelength of X|
42|calculate_slope_sign_change(X)|Calculates the number of times the slope of X changes sign|
43|calculate_higuchi_fractal_dimensions(X)|Uses the Higuchi method to calculate the fractal dimensions of X|
44|calculate_permutation_entropy(X)|Calculates the permutation entropy of X|
45|calculate_svd_entropy(X)|Singular Value Decomposition|
46|calculate_hjorth_mobility_and_complexity(X)|Calculates mobility and complexity of X which are bases on the first and second derivatives of X|
47|calculate_cardinality(X)||*|
48|calculate_rms_to_mean_abs(X)|Computes the ratio of the RMS value to mean absolute value of X|
49|calculate_tsallis_entropy(X)|Tsallis entropy estimates the information X|*|
50|calculate_renyi_entropy(X)|Computes the Renyi entropy of X|
51|calculate_absolute_energy(X)|Calculates the absolute energy of X|
52|calculate_approximate_entropy(X)|Computes the approximate entropy of X|
53|calculate_area_under_curve(X)|Calculates the area under the curve of X|
53|calculate_area_under_squared_curve(X)|Computed the area under the curve of X squared|*|
54|calculate_autoregressive_model_coefficients(X)|Calculates the autoregressive model coefficients of X|
55|calculate_count(X)|Returns the number of values in X|*|
56|calculate_count_above_mean(X)|Computes the number of values of X above the mean of X|
57|calculate_count_below_mean(X)|Computes the number of values of X below the mean of X|
58|calculate_count_of_negative_values(X)|Returns the number of values of X that are negative|
59|calculate_count_of_positive_values(X)|Returns the number of values of X that are positive|
60|calculate_covariance(X,Y)|Calculates the covariance of X and Y|
61|calculate_cumulative_energy(X)|Calculates the cumulative energy of X|
62|calculate_cumulative_sum(X)|Calculates the cumulative sum along the specified dimension of X|
63|calculate_differential_entropy(X)|Computes the differential entropy of X|
64|calculate_energy_ratio_by_chunks(X, param)|Calculates the energy of chunk i out of N chunks expressed as a ratio with the sum of squares over the X|
65|calculate_exponential_moving_average(X, param)|Calculates the exponential moving average of X|*|
66|calculate_first_location_of_maximum(X)|Returns the first location of the maximum value of X|
67|calculate_first_location_of_minimum(X)|Returns the first location of the minimum value of X|
68|calculate_first_order_difference(X)|Returns the first order differential of X|
69|calculate_first_quartile(X)|Computes the first quartile of X|*|
70|calculate_fisher_information(X)|Computes the Fisher information of X|
71|calculate_histogram_bin_frequencies(X, param)|Returns the histogram bin frequencies of X|
72|calculate_intercept_of_linear_fit(X)||~|
73|calculate_katz_fractal_dimension(X)|Computes the Katz fractal dimension of X|
74|calculate_last_location_of_maximum(X)|Returns the last location of the maximum value of X|
75|calculate_last_location_of_minimum(X)|Returns the last location of the minimum value of X|
76|calculate_linear_trend_with_full_linear_regression_results(X)||~|
77|calculate_local_maxima_and_minima(X)|Calculates the local maxima and minima of X|
78|calculate_log_return(X)|Returns the logarithm of the ratio between the last and first values of  which is a measure of the percentage change in X|~|
79|calculate_longest_strike_above_mean(X)|Computes the length of the longest consecutive subsequence of X  that is greater than the mean of X|
80|calculate_longest_strike_below_mean(X)|Computes the length of the longest consecutive subsequence of X  that is lesser than the mean of X|
81|calculate_lower_complete_moment(X)||*|
82|calculate_mean_absolute_change(X)||
83|calculate_mean_crossings(X)|Calculates the number of times X crossed the mean value|
84|calculate_mean_relative_change(X)|Returns the mean relative change of X|
85|calculate_mean_second_derivative_central(X)|Returns the mean of the second derivative of X|
86|calculate_median_second_derivative_central(X)|Calculates the median of the second derivative of X|*|
87|calculate_mode(X)|Returns the mode of X|*|
88|calculate_moving_average(X)|Returns the moving average of X|
89|calculate_number_of_inflection_points(X)|Computes the number of inflection points in X|
90|calculate_peak_to_peak_distance(X)|Calculates the peak-to-peak distance of X|
91|calculate_pearson_correlation_coefficient(X)|Calculates the pearson correlation coefficient of X|
92|calculate_percentage_of_negative_values(X)|Returns the percentage of negative values of X |*|
93|calculate_percentage_of_positive_values(X)|Returns the percentage of positive values of X |
94|calculate_percentage_of_reoccurring_datapoints_to_all_datapoints(X)|Returns the percentage of values that occur another time in the time series X|
95|calculate_percentage_of_reoccurring_values_to_all_values(X)|Calculates the percentage of values in X that occur more than once|
96|calculate_percentile(X)|Calculates the 20th, 50th and 75th percentile of X|*|
97|calculate_petrosian_fractal_dimension(X)|Computes the petrosian fractal dimension of X|
98|calculate_ratio_beyond_r_sigma(X)|Returns the ratio of values that are more than r * std away from the mean of X|
99|calculate_ratio_of_fluctuations(X)|Computes the ratio of positive and negative fluctuations in X|*|
100|calculate_ratio_value_number_to_sequence_length(X)|Returns the ratio of length of a set of X to the length X|*|
101|calculate_sample_entropy(X)|Returns the sample entropy of X|
102|calculate_second_order_difference(X)|Returns the second differential of X|**|
103|calculate_signal_resultant(X)||*|
104|calculate_signal_to_noise_ratio(X)|Calculates the signal to noise ratio of X|
105|calculate_slope_of_linear_fit(X)|Returns the slope of X|
106|calculate_smoothing_by_binomial_filter(X)||~|
107|calculate_stochastic_oscillator_value(X)|Calculates the stochastic oscillator of X|~|
108|calculate_sum(X)|Returns the overall sum of values in X|
109|calculate_sum_of_negative_values(X)|Calculates the sum of negative values in X|*|
110|calculate_sum_of_positive_values(X)|Returns the sum of positive values in X|*|
111|calculate_sum_of_reoccurring_data_points(X)|Calculates the sum of values of X that occur more than once|
112|calculate_sum_of_reoccurring_values(X)||
113|calculate_third_quartile(X)|Returns the third quartile of X|*|
114|calculate_variance_of_absolute_differences(X)|Returns variance of the absolute of the first order difference of X|
115|calculate_weighted_moving_average(X)|Returns the weighted moving average of X|
116|calculate_winsorized_mean(X)|Calculates the winsorized mean of X which replaces the lowest and highest outliers with closer non-extreme values before calculating the average|
117|calculate_zero_crossing_rate(X)|Returns the zero-crossing rate of X|

<br>
<br>
<br>
<br>

## Time Frequency Features
|Number| Feature    | Description | Reference |
| -------- | ------- | ------- | ------- |
1|extract_wavelet_features(params)|||
2|extract_spectrogram_features(params)||
3|extract_stft_features(params)||
4|teager_kaiser_energy_operator(X)|Generates new time series based on the Teager Kaiser energy operator|



### Deleted features
Number| Feature    | Reason |
| -------- | ------- | ------- |
1|calculate_roll_mean | Same implementation as *calculate_moving_average*


## Features that should be deleted
Number| Feature    | Reason |
| -------- | ------- | ------- |
1|calculate_first_quartile | calculate_percentile(signal, percentiles=[25, 50, 75]) returns the first, second, and third quartiles|
2|calculate_third_quartile | calculate_percentile(signal, percentiles=[25, 50, 75]) returns the first, second, and third quartiles |


