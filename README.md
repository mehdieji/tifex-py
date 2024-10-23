# proj-adl-classification

## statistical_feature.py
Using GPU to compute statistical features based on PyTorch.  

Also compare the results with the features computed by CPU (Numpy).  

The return is a pd dataframe with columns: 'feature name', 'feature value gpu', 'feature value cpu', and 'time consumption'. 

#### X - Time series
"*": No reference
<br>
"* **": More than one reference and one is questionable
<br>
"~": Further research required on feature
<br>

## Statistical Features


|Number| Feature    | Description | Info |
| -------- | ------- | ------- | ------- |
1| calculate_harmonic_mean_abs(X)| Calculates the harmonic mean of the absolute values of X| * |
2|calculate_trimmed_mean_abs(X)| Calculates the trimmed mean of absolute values of X| * |
3|calculate_std_abs(X) | Calculates the standard deviation of the absolute values of X | * |
4|calculate_skewness_abs(X)| Calculate skewness of absolute values of X|*|
5|calculate_kurtosis_abs(X)|Calculates the kurtosis of the absolute values of X| * |
6|calculate_median_abs|Calculates the median of the absolute values of X|*|
7|calculate_min_abs(X)|Calculates the minimum value of the absolute values of X| *|
8|calculate_range_abs(X)|Calculates the range of the absolute values of X|*|
9|calculate_variance_abs(X)|Calculates the variance of the absolute values of X|*|
10|calculate_mean_absolute_deviation(X)|Calculates the mean of the absolute deviation of X | ~|
11|calculate_signal_magnitude_area(X)|Calculates the magnitude area of X. The sum of the absolute values of X| ~|
12|calculate_cardinality(X)|~|
13|calculate_rms_to_mean_abs(X)|Computes the ratio of the RMS value to mean absolute value of X|*|
14|calculate_area_under_squared_curve(X)|Computed the area under the curve of X squared|*|
15|calculate_exponential_moving_average(X, param)|Calculates the exponential moving average of X|*|
16|calculate_fisher_information(X)|Computes the Fisher information of X|~|
17|calculate_local_maxima_and_minima(X)|Calculates the local maxima and minima of X|*|
18|calculate_log_return(X)|Returns the logarithm of the ratio between the last and first values of  which is a measure of the percentage change in X|~|
19|calculate_lower_complete_moment(X)||*|
20|calculate_mean_second_derivative_central(X)|Returns the mean of the second derivative of X|
21|calculate_median_second_derivative_central(X)|Calculates the median of the second derivative of X|*|
23|calculate_ratio_of_fluctuations(X)|Computes the ratio of positive and negative fluctuations in X|*|
24|calculate_ratio_value_number_to_sequence_length(X)|Returns the ratio of length of a set of X to the length X|*|
25|calculate_second_order_difference(X)|Returns the second differential of X|**|
26|calculate_signal_resultant(X)||*|
27|calculate_sum_of_negative_values(X)|Calculates the sum of negative values in X|*|
28|calculate_sum_of_positive_values(X)|Returns the sum of positive values in X|*|
29|calculate_variance_of_absolute_differences(X)|Returns variance of the absolute of the first order difference of X|
30|calculate_weighted_moving_average(X)|Returns the weighted moving average of X|*
31|calculate_covariance||~|


<br>
<br>

## Statistical Features -  NEW!!
|Number| Feature    | Reference |
| -------- | ------- | ------- |
1.|calculate_mean_to_variance| 
<br>
<br>

## Time-Frequency Features
|Number| Feature    | Reference |
| -------- | ------- | ------- |
1|extract_wavelet_features(params)|||
2|extract_spectrogram_features(params)||
3|extract_stft_features(params)||
4|teager_kaiser_energy_operator(X)|

<br>
<br>

## Spectral Features
|Number| Feature    | Reference |
| -------- | ------- | ------- |
1|calculate_spectral_subdominant_valley|*
2||



<br>


# NOT in tsfresh
<br>

## Spectral Features
1. Median frequency
2. Spectral bandwidth
3. Spectral absolute deviation
4. Spectral slope linear
5. Spectral slope logarithmic
6. Spectral flatness
7. Peak frequencies
8. Spectral edge frequency
9. Band power
10. Spectral entropy
11. Spectral contrast
12. Spectral coefficient variation
13. Spectral flux
14. Spectral rolloff
15. Harmonic ratio
16. Fundamental frequency
17. Spectral crest factor
18. Spectral decrease
19. Spectral irregularity
20. Mean frequency
21. Frequency winsorized mean
22. Total harmonic distortion
23. Inharmonicity


23. Tristimulus
24. Spectral rollon
25. Spectral hole count
26. Spectral autocorrelation
27. Spectral variability
28. Spectral spread ratio
29. Spectral skewness ratio
30. Spectral kurtosis ratio
31. Spectral tonal power ratio
32. Spectral noise to harmonics ratio
33. Spectral even to odd harmonic energy ratio
34. Spectral strongest frequency phase
35. Spectral frequency below peak
36. Spectral frequency above peak
37. Spectral cumulative frequency
38. Spectral cumulative frequency
39. Spectral cumulative frequency above
40. Spectral spread shift
41. Spectral entropy shift
42. Spectral change vector magnitude
43. Spectral low frequency content
44. Spectral mid frequency content
45. Spectral peak-to-valley ratio
46. Spectral valley depth mean
47. Spectral valley depth std
48. Spectral valley depth variance
49. Spectral valley width mode
50. Spectral valley width standard deviation
51. Spectral subdominant valley
52. Spectral valley count
53. Spectral peak broadness
54. Spectral valley broadness
55. Frequency variance
56. Frequency standard deviation
57. Frequency Range
58. Frequency Trimmed mean
59. Harmonic product spectrum
60. Smoothness
61. Roughness

<br>

# Time-Frequency Features
Statistical features from wavelets, spectrogram and short-time fourier transform


# Statistical Features
62. Hurst exponent from detrended fluctuation analysis
62. Winsorized mean
63. Weighted moving average
64. Sum of positive values
65. Sum of negative values
66. Stochastic oscillator value
67. Smoothing by binomial filter
68. Signal-to-noise ratio
69. Signal resultant
70. Second order difference
71. Ratio value number to sequence length 
72. Ratio beyond r signal
73. Petrosian fractal dimension
74. Percentage of positive values
75. Percentage of negative values
76. Pearson correlation coefficient
77. Peak-to-peak distance
78. Number of inflection points
79. Moving average
80. Mode
81. Median second derivative central
82. Mean relative change
83. Mean crossings
84. Lower complete moment
85. Log return 
86. Katz fractal dimension
88. Histogram bin frequencies
89. Fisher information
90. First quartile
91. First order difference
92. Exponential moving average
93. Energy ratio by chunks
94. Differential entropy
95. Cumulative sum
96. Covariance
97. Count
98. Area under curve
99. Area under squared curve
100. Renyi entropy
101. Tsallis entropy
102. Root mean squared to mean absolute
103. Cardinality
104. Hjorth mobility and complexity
105. Singular value decomposition (SVD) entropy
106. Higuchi fractal dimensions
107. Slope sign change
108. Average amplitude change
109. Signal magnitude area*
110. Median absolute deviation
111. Coefficient of variation
112. Higher order moments
113. Mean auto correlation
114. Impulse factor
115. Shape factor
116. Clearance factor
117. Crest factor
118. Zero crossings
119. Entropy
120. Log energy
121. Mean absolute deviation
122. Interquartile range
123. Variance absolute
124. Maximum absolute
125. Minimum absolute
125. Range absolute
126. Range
127. Median absolute
128. Kurtosis absolute
129. Skewness absolute
130. Standard deviation absolute
131. Trimmed mean absolute
132. Trimmed mean
133. Harmonic Mean
134. Harmonic mean absolute
135. Geometric mean 
136. Geometric mean absolute
137. Mean absolute


<br>

# Added features
|Number| Feature    | Description |
| -------- | ------- | ------- |
1| Augmented dickey fuller test| Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity in a given time series signal.|
2| Hurst exponent| Calculate the Hurst Exponent of a given time series using Detrended Fluctuation Analysis (DFA).|

<br>

### Deleted features
Number| Feature    | Reason |
| -------- | ------- | ------- |
1|calculate_roll_mean | Same implementation as *calculate_moving_average*
2|calculate_absolute_energy | Same implementation as signal energy
3|calculate_cumulative_energy | Produces same result as the absolute energy and signal energy. These three will always be the same for a given signal.
4|calculate_intercept_of_linear_fit| This feature is returned again in the calculate_linear_trend_with_full_linear_regression_results function  
5|calculate_pearson_correlation_coefficient| Since this function calculates the Pearson correlation coefficient between the signal and its one-step lagged version, it is fundamentally calculating the autocorrelation of the signal. The autocorrelation is already present(calculate_mean_auto_correlation). Having both is redundant. 
6|calculate_slope_of_linear_fit| This is already calculated in calculate_linear_trend_with_full_linear_regression_results
7|calculate_frequency_std| Same implementation as calculate_spectral_bandwidth with order set to 2
8|calculate_frequency_variance| Same implementation as calculate_spectral_variance
9|calculate_mean_frequency(freqs, magnitudes) | Same as calculate_spectral_centroid with order set to 1
10|calculate_first_quartile | calculate_percentile(signal, percentiles=[25, 50, 75]) returns the first, second, and third quartiles|
11|calculate_third_quartile | calculate_percentile(signal, percentiles=[25, 50, 75]) returns the first, second, and third quartiles |
14| calculate_spectral_entropy_shift|Same implementation as calculate_spectral_entropy but with spectrum_magnitudes as argument and not psd|
13| calculate_spectral_spread_shift| Same spectral standard deviation
14| calculate_spectral_autocorrelatiion| Autocorrelation of magnitudes is backed by literature


## Features that should be deleted
Number| Feature    | Type | Reason|
| -------- | ------- | ------- | ------- |
1| calculate_histogram_bins| statistical
2| calculate_signal_magnitude_area|statistical
3| calculate_spectral_hole_count| spectral | Spectral holes are typically of use in radio signals. Although the aim is to make this a very comprehensive toolbox, this feature is a little bit out of scope.


## Features in Tsfresh but not in SCAI toolbox
Number| Feature | Description|Added yet?|
| -------- | ------- | ------- | -------|
1|absolute sum of changes||✔️|
2|ar_coefficient(x, param)| This feature calculator fits the unconditional maximum likelihood of an autoregressive AR(k) process|
3|benford correlation|||
4|c3| uses c3 statistics to measure non-linearity in the time series
5|count_above(x, t)|Returns the percentage of values in x that are higher than t |✔️|
6|count_below(x, t)| Returns the percentage of values in x that are lower than t|✔️|
7|cid_ce(x, normalize) |This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks, valleys etc.).|✔️|
8|friedrich_coefficients(x, param)|Coefficients of polynomial h(x), which has been fitted to the deterministic dynamics of Langevin model
9|has_duplicate(x)|Checks if any value in x occurs more than once| ✔️|
10|has_duplicate_max(x)|Checks if the maximum value of x is observed more than once| ✔️|
11|has_duplicate_min(x)|Checks if the minimal value of x is observed more than once| ✔️|
12|index_mass_quantile(x, param)|Calculates the relative index i of time series x where q% of the mass of x lies left of i.
13|mean_n_absolute_max(x, number_of_maxima)| Calculates the arithmetic mean of the n absolute maximum values of the time series.
14|large_standard_deviation(x, r)|Does time series have large standard deviation| ✔️ |
15|lempel_ziv_complexity(x, bins)|Calculate a complexity estimate based on the Lempel-Ziv compression algorithm.|✔️|
16|matrix_profile(x, param)|Calculates the 1-D Matrix Profile[1] and returns Tukey's Five Number Set plus the mean of that Matrix Profile.
17|max_langevin_fixed_point(x, r, m)|Largest fixed point of dynamics :math:argmax_x {h(x)=0}` estimated from polynomial h(x), which has been fitted to the deterministic dynamics of Langevin model
18|binned entropy |
19| symmetry looking|Boolean variable denoting if the distribution of x looks symmetric.
20|change_quantiles|First fixes a corridor given by the quantiles ql and qh of the distribution of x.|
21|fft_coefficient| Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast fourier transformation algorithm
22|matrix_profile|Calculates the 1-D Matrix Profile[1] and returns Tukey's Five Number Set plus the mean of that Matrix Profile.
23|mean_n_absolute_max|Calculates the arithmetic mean of the n absolute maximum values of the time series.
24|number_crossing_m|Calculates the number of crossings of x on m.
25|number_cwt_peaks|Number of different peaks in x.
26|number_peaks|Calculates the number of peaks of at least support n in the time series x.
27|partial_autocorrelation|Calculates the value of the partial autocorrelation function at the given lag.
28|query_similarity_count|This feature calculator accepts an input query subsequence parameter, compares the query (under z-normalized Euclidean distance) to all subsequences within the time series, and returns a count of the number of times the query was found in the time series (within some predefined maximum distance threshold).
29|ratio_value_number_to_time_series_length|Returns a factor which is 1 if all values in the time series occur only once, and below one if this is not the case.
30|value_count|Count occurrences of value in time series x.
31|variance_larger_than_standard_deviation|Is variance higher than the standard deviation?



## Observations
1. **calculate_higher_order_moments** does not always produce the same result as mean, variance, skew and kurtosis when moment order is set to [1,2,3,4]
2. **calculate_rms_to_mean_abs** has no direct reference yet
3. **calculate_exponential_moving_average** returns the last value in the array. Is there a reason?
4. 

<br>

Corrections

1. calculate_katz_fractal_dimensions
2. calculate_sum_of_reoccurring_values
3. calculate_sum_of_reoccurring_data_points
4. calculate_petrosian_fractal_dimension
5. calculate_sample_entropy
6. calculate_approximate_entropy












