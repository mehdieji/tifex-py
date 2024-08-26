# # %%
# import numpy as np
# from StatisticalFeatures_cpu import StatisticalFeatures
# from scipy.stats import skew, kurtosis, moment, gmean, hmean, trim_mean, entropy, linregress, mode, pearsonr
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import acf, adfuller


# # %%
# df = pd.read_csv('/home/edumaba/SCAI_lab/proj-adl-classification/Public Dataset/UniMiBSHAR_data.csv')

# # %%
# signal = df[['accx', 'accy', 'accz']].values.tolist()
# signal = np.array(signal)
# # flatten list
# signal = [item for sublist in signal for item in sublist]
# signal = np.array(signal)
# signal

# # %%


# # def calculate_higher_order_moments(signal, moments=[1,2,3, 4, 5, 6]):
# #     feats = []
# #     for order in moments:
# #         feats.append(moment(signal, moment=order))
# #     return np.array(feats)

# # def calculate_mean(signal):
# #     return np.array([np.mean(signal)])

# # def calculate_variance(signal):
# #     return np.array([np.var(signal)])

# # def calculate_skewness(signal):    
# #     return np.array([skew(signal)])

# # def calculate_kurtosis(signal):
# #     return np.array([kurtosis(signal)])


# # print("Higher-order moments:", calculate_higher_order_moments(signal))
# # print("Mean:", calculate_mean(signal))
# # print("Variance:",calculate_variance(signal))
# # print("Skewness:", calculate_skewness(signal))
# # print("Kurtosis:", calculate_kurtosis(signal))


# # %% [markdown]
# # Other signal

# # %%
# # row = np.array([0.5346, 0.5906, 0.6535, 0.7217])
# # column = np.array([0.9273, 1.0247, 1.1328])
# # digit_sounds = np.empty(10, dtype = object)

# # end = 1000

# # #Avoid value 0 so that there is no zero value is the digit_sounds
# # n = np.arange(1, end + 1, end / 1000)

# # digit_sounds[0] = np.sin(0.7217*n) + np.sin(1.0247*n)

# # len_row_without_last = len(row) - 1
# # for i in range(len_row_without_last) :
# #     a = row[i]
# #     for j in range(len(column)) :
# #         b = column[j]
# #         digit_sounds[len_row_without_last * i + j + 1] = np.sin(a*n) + np.sin(b*n)
# # signal = np.empty(0)
# # for i in range(1, len(digit_sounds)):
# #     signal = np.concatenate((signal, digit_sounds[i]))
# # signal


# # %%
# signal_name = 'xavier'

# # %%
# len(signal)

# # %%
# window_size = 1000
# n_lags_auto_correlation = int(min(10 * np.log10(window_size), window_size - 1))
# moment_orders = [3, 4]
# trimmed_mean_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
# higuchi_k_values=list({5, 10, 20, window_size // 5})
# tsallis_q_parameter=1
# renyi_alpha_parameter=2
# permutation_entropy_order=3
# permutation_entropy_delay=1
# svd_entropy_order=3
# svd_entropy_delay=1

# # %% [markdown]
# # 

# # %%
# stats_features = StatisticalFeatures(window_size,n_lags_auto_correlation,moment_orders,trimmed_mean_thresholds,higuchi_k_values,tsallis_q_parameter,renyi_alpha_parameter,permutation_entropy_order,permutation_entropy_delay,svd_entropy_order,svd_entropy_delay)
# features, features_names = stats_features.calculate_statistical_features(signal, signal_name)
# pd.set_option('display.max_rows', None)
# df = pd.DataFrame(
#     {
#         "Name": features_names,
#         "Features": features
#     }
# )
# df

# # %% [markdown]
# # Does the time series have a unit root? Augmented Dickey Fuller Test

# # %%
# def calculate_augmented_dickey_fuller_test(signal):
#         """
#         Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity in a given time series signal.

#         The ADF test is a statistical test used to determine if a time series is stationary or has a unit root.
#         A stationary time series has constant mean and variance over time.

#         Parameters:
#         ----------
#         signal (array-like): 
#             The time series data to be tested for stationarity.

#         Returns:
#         -------
#         np.array or float:
#                     A numpy array containing the test statistic, p-value, and number of lags used in the test.
#                     If the test fails due to an exception, returns NaN.
#         Reference:
#         ---------
#             Christ et al., 2018, https://doi.org/10.1016/J.NEUCOM.2018.03.067
#         """
#         adf_vals_names = np.array(["teststats", "pvalue", "usedlag"])
#         try:
#             if len(signal) <= 9000:
#                 test_stat, p_value, used_lag, _,_,_ = adfuller(signal)
#                 adf_vals = np.array([test_stat, p_value, used_lag])
#             else: 
#                 adf_vals = np.empty(3)
#         except:
#             return np.nan
#         return adf_vals, adf_vals_names

# # %%
# calculate_augmented_dickey_fuller_test(signal)

# # %%
# # def calculate_moving_average(signal, window_size):
# #         # https://cyclostationary.blog/2021/05/23/sptk-the-moving-average-filter/
# #         if len(signal) < window_size:
# #             return np.nan
# #         return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
    

# # %%
# # features_df = pd.DataFrame(features, feature_names)
# # pd.set_option('display.max_rows', None)
# # df = pd.DataFrame(
# #     {
# #         "Name": features_names,
# #         "Features": features
# #     }
# # )
# # print(f"feats:{len(features)} feats_names:{len(features_names)}")
# # df

# # %%
# # pd.set_option('display.max_rows', None)
# # # features_df = features_df.T
# # features_df

# # %%
# # features_df.shape

# # %%
# def calculate_cardinality(signal):
#     # Parameter
#     thresh = 0.05 * np.std(signal)  # threshold
#     # Sort data
#     sorted_values = np.sort(signal)
#     cardinality_array = np.zeros(window_size - 1)
#     for i in range(window_size - 1):
#         cardinality_array[i] = np.abs(sorted_values[i] - sorted_values[i + 1]) 
#     cardinality = np.sum(cardinality_array)
#     return np.array([cardinality])


# # %%
# calculate_cardinality(signal)

# # %%
# hist = np.array([1,2,0,5,4,6,0])
# epsilon = 1e-10
# hist = np.where(hist > 0, hist, epsilon)
# hist


# # %%
# entropy = -np.sum(hist * np.log2(hist))
# entropy

# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%
# def calculate_detrended_fluctuation_analysis(signal, order=1, minimal = 20):
#     """
#     Performs detrended fluctuation analysis (DFA) on the given signal.
    
#     Parameters:
#     ----------
#     signal: array
#         The input signal.
#     order: integer
#         The order of the polynomial fit for local detrending (default is 1 for linear detrending).
#     minimal: integer
#         The minimum segment size to consider
    
#     Returns:
#     -------
#     segment_sizes: array
#         Array of segment sizes.
#     fluctuation_values: array
#         Fluctuation function values corresponding to segment sizes.
#     """   
#     # Calculate the cumulative sum of the mean-shifted signal
#     signal_mean = np.mean(signal)
#     mean_shifted_signal = signal - signal_mean
#     cumulative_sum_signal = np.cumsum(mean_shifted_signal)
    
#     N = len(signal)
    
#     def Divisors(N, minimal=20):
#         D = []
#         for i in range(minimal, N // minimal + 1):
#             if N % i == 0:
#                 D.append(i)
#         return D
    
#     def findOptN(N, minimal=20):
#         """
#         Find such a natural number OptN that possesses the largest number of
#         divisors among all natural numbers in the interval [0.99*N, N]
#         """
#         N0 = int(0.99 * N)
#         # The best length is the one which have more divisiors
#         Dcount = [len(Divisors(i, minimal)) for i in range(N0, N + 1)]
#         OptN = N0 + Dcount.index(max(Dcount))
#         return OptN
    
    
#     OptN = findOptN(len(signal), minimal=minimal)
#     segment_sizes = Divisors(OptN, minimal=minimal)
#     fluctuation_values = []

#     for m in segment_sizes:
#         k = OptN // m
#         Y = np.reshape(cumulative_sum_signal[N - OptN:], [m, k], order='F')
#         F = np.copy(Y)
#         # t = 1, 2, ..., m
#         t = np.linspace(1, m, m)
#         for i in range(k):
#             p = np.polyfit(t, Y[:, i], 1)
#             F[:, i] = Y[:, i] - t * p[0] - p[1]
#         fluctuation_values.append(np.mean(np.std(F)))
    
#     return segment_sizes, np.array(fluctuation_values)
    


# # %%
# def calculate_moving_average(signal, window_size=10):
#         # https://cyclostationary.blog/2021/05/23/sptk-the-moving-average-filter/
#         if len(signal) < window_size:
#             return np.nan
#         return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')


# # %%
# calculate_moving_average(signal)

# # %%
# ss, dfa = calculate_detrended_fluctuation_analysis(signal)
# # dfa

# # %%
# def calculate_hurst_exponent(signal):
#     """
    
#     """
#     segment_size, fluctuation_values = calculate_detrended_fluctuation_analysis(signal)
    
#     poly = np.polyfit(np.log(segment_size), np.log(fluctuation_values), 1)
#     hurst = poly[0]
#     return hurst

# # %%
# calculate_hurst_exponent(signal)

# # %%
# def Divisors(N: int, minimal=20) -> list:
#         D = []
#         for i in range(minimal, N // minimal + 1):
#             if N % i == 0:
#                 D.append(i)
#         return D

# # %%
# def findOptN(N: int, minimal=20) -> int:
#         """
#         Find such a natural number OptN that possesses the largest number of
#         divisors among all natural numbers in the interval [0.99*N, N]
#         """
#         N0 = int(0.99 * N)
#         # The best length is the one which have more divisiors
#         Dcount = [len(Divisors(i, minimal)) for i in range(N0, N + 1)]
#         OptN = N0 + Dcount.index(max(Dcount))
#         return OptN

# # %%
# def EstHurstDFAnalysis(ts, minimal=20, method='L2') -> float:
#         """
#         DFA Calculate the Hurst exponent using DFA analysis.

#         Parameters
#         ----------
#         ts     : Time series.
#         minimal: The box sizes that the sample is divided into, default as 10.
#         method : The method to fit curve, default as minimal l2-norm.

#         Returns
#         -------
#         The Hurst exponent of time series X using
#         Detrended Fluctuation Analysis (DFA).

#         References
#         ----------
#         [1] C.-K.Peng et al. (1994) Mosaic organization of DNA nucleotides,
#         Physical Review E 49(2), 1685-1689.
#         [2] R.Weron (2002) Estimating long range dependence: finite sample
#         properties and confidence intervals, Physica A 312, 285-299.

#         Written by z.q.feng (2022.09.23).
#         Based on dfa.m orginally written by afal Weron (2011.09.30).
#         """
#         DF = []
#         N = len(ts)
#         y = np.cumsum(ts - np.mean(ts))


#         OptN = findOptN(len(ts), minimal=minimal)
#         M = Divisors(OptN, minimal=minimal)


#         for m in M:
#             k = OptN // m
#             Y = np.reshape(y[N - OptN:], [m, k], order='F')
#             F = np.copy(Y)
#             # t = 1, 2, ..., m
#             t = np.linspace(1, m, m)
#             for i in range(k):
#                 p = np.polyfit(t, Y[:, i], 1)
#                 F[:, i] = Y[:, i] - t * p[0] - p[1]
#             DF.append(np.mean(np.std(F)))

#         # slope = __FitCurve(M, DF, method=method)
#         # poly = np.polyfit(np.log(segment_size), np.log(fluctuation_values), 1)
#         print(M)
#         print(DF)
#         slope = np.polyfit(np.log(M), np.log(DF), 1)
#         slope = slope[0]
        
#         return slope

# # %%
# EstHurstDFAnalysis(signal)

# # %%
# [4      6     12     20     36     62    109    189    328    570 990   1718   2981   5173   8976  15576  27029  46902  81387 141226] ,
# [5.34293164e+00 6.91020702e+00 1.18330006e+01 2.00455171e+01 3.86139236e+01 7.68862568e+01 1.86136323e+02 4.49861202e+02  9.42417728e+02 1.87418631e+03 4.76845714e+03 1.16356970e+04 2.65568830e+04 6.06326452e+04 1.47438848e+05 3.62480438e+05 8.75148638e+05 2.38564283e+06 4.78137887e+06 9.58846521e+06]

# # %%
# n_vals, F_vals = calculate_detrended_fluctuation_analysis(signal)


# plt.loglog(n_vals, F_vals, 'o-')
# plt.xlabel('Segment size (n)')
# plt.ylabel('Fluctuation function (F(n))')
# plt.title('Detrended Fluctuation Analysis')
# plt.show()

# # %%
# signal_mean = np.mean(signal)
# mean_shifted_signal = signal - signal_mean
# cumulative_sum_signal = np.cumsum(mean_shifted_signal)



# plt.subplot(2,1,1)
# plt.plot(signal)
# plt.subplot(2,1,2)
# plt.plot(cumulative_sum_signal)
# plt.show()

# # %%
# print(np.log10(4))
# print(np.log10(N//4))

# # %%
# len(signal)/4


# # %%
# N = len(signal)
# n_vals = np.floor(np.logspace(np.log10(4), np.log10(N//4), num=20)).astype(int)
# n_vals

# # %%


# # %%


# # %%


# # %%

# def calculate_moving_average(signal, window_size=10):
#         # https://cyclostationary.blog/2021/05/23/sptk-the-moving-average-filter/
#         if len(signal) < window_size:
#             return np.nan
#         return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
    
# def calculate_weighted_moving_average(signal, weights=None):
#         # https://www.mathworks.com/help/signal/ug/signal-smoothing.html
#         if weights is None:
#             weights = np.linspace(1, 0, num=len(signal))
#         weights = weights / np.sum(weights)
#         return np.convolve(signal, weights, 'valid')

# def calculate_exponential_moving_average(signal, alpha=0.3):
#         """
#         Calculates the exponential moving average of the given signal

#         Parameters:
#         ---------
#             signal (array-like): The input signal.
#             alpha (float, optional): Defaults to 0.3.

#         Returns:
#         -------
#             float
#                 last value in the array
#         """
#         s = np.zeros_like(signal)
#         s[0] = signal[0]
#         for i in range(1, len(signal)):
#             s[i] = alpha * signal[i] + (1 - alpha) * s[i - 1]
#         return s

# # %%
# ma = calculate_moving_average(signal)
# wma = calculate_weighted_moving_average(signal)
# ema = calculate_exponential_moving_average(signal)

# print(wma)

# plt.subplot (2,2,1)
# plt.plot(signal)
# plt.title("Original signal")
# plt.subplot (2,2,2)
# plt.plot(ma)
# plt.title("ma signal")
# plt.subplot (2,2,3)
# plt.plot(wma)
# plt.title("wma signal")
# plt.subplot (2,2,4)
# plt.plot(ema)
# plt.title("ema signal")
# plt.show()

# # %%


# # %%


# # %%


# # %%


# # %% [markdown]
# # Time Frequency Features

# # %% [markdown]
# # Testing wavelet coefficients from TimeFrequency class. <br>
# # *Error on line 128 in file TimeFrequencyFeatures_cpu*

# # %%
# import pywt
# import numpy as np 

# wavelet='db4'
# window_size = 100

# wavelet_length = len(pywt.Wavelet(wavelet).dec_lo)
# decomposition_level = int(np.round(np.log2(window_size/wavelet_length) - 1))
# wavelet_coefficients = pywt.wavedec(signal, wavelet, level=decomposition_level)


# # %%
# statistical_feature_extractor = StatisticalFeatures(window_size)

# # %%
# def extract_wavelet_features(signal_name, wavelet_coefficients):
#         # https://doi.org/10.1016/B978-012047141-6/50006-9
#         feats = []
#         feats_names = []

#         for i_level in range(len(wavelet_coefficients)):
#             coeffs = wavelet_coefficients[i_level]
#             statistical_features, statistical_feature_names = statistical_feature_extractor.calculate_statistical_features(coeffs, signal_name)
#             feats.extend(statistical_features)
#             feats_names.extend([f"{signal_name}_wavelet_lvl_{i_level}_{name}" for name in statistical_feature_names])

#         return feats, feats_names

# # %%
# fet, fet_nam = extract_wavelet_features('mansa', wavelet_coefficients)

# # %% [markdown]
# # Selecting all the data of subject 1
# # <br>
# # Data from 

# # %%


# # %%
# # df_label_9_subject_1 = df[(df['label'] == 9) & (df['subject']) == 1]
# # print(df_label_9_subject_1)

# # %%


# # %%
# class StaHurst:
#     # def __init__(self):
#     # # First implementation in file
#     def calculate_detrended_fluctuation_analysis(self, signal, order=1, minimal = 20):
#         """
#         Performs detrended fluctuation analysis (DFA) on the given signal.
        
#         Parameters:
#         ----------
#         signal: array
#             The input signal.
#         order: integer
#             The order of the polynomial fit for local detrending (default is 1 for linear detrending).
#         minimal: integer
#             The minimum segment size to consider
        
#         Returns:
#         -------
#         segment_sizes: array
#             Array of segment sizes.
#         fluctuation_values: array
#             Fluctuation function values corresponding to segment sizes.
#         """   
#         # Calculate the cumulative sum of the mean-shifted signal
#         signal_mean = np.mean(signal)
#         mean_shifted_signal = signal - signal_mean
#         cumulative_sum_signal = np.cumsum(mean_shifted_signal)
        
#         N = len(signal)
        
#         def Divisors(N, minimal=20):
#             D = []
#             for i in range(minimal, N // minimal + 1):
#                 if N % i == 0:
#                     D.append(i)
#             return D
        
#         def findOptN(N, minimal=20):
#             """
#             Find such a natural number OptN that possesses the largest number of
#             divisors among all natural numbers in the interval [0.99*N, N]
#             """
#             N0 = int(0.99 * N)
#             # The best length is the one which have more divisiors
#             Dcount = [len(Divisors(i, minimal)) for i in range(N0, N + 1)]
#             OptN = N0 + Dcount.index(max(Dcount))
#             return OptN
        
        
#         OptN = findOptN(len(signal), minimal=minimal)
#         segment_sizes = Divisors(OptN, minimal=minimal)
#         fluctuation_values = []

#         for m in segment_sizes:
#             k = OptN // m
#             Y = np.reshape(cumulative_sum_signal[N - OptN:], [m, k], order='F')
#             F = np.copy(Y)
#             # t = 1, 2, ..., m
#             t = np.linspace(1, m, m)
#             for i in range(k):
#                 p = np.polyfit(t, Y[:, i], 1)
#                 F[:, i] = Y[:, i] - t * p[0] - p[1]
#             fluctuation_values.append(np.mean(np.std(F)))
        
#         return segment_sizes, np.array(fluctuation_values)
    

        
#     def calculate_hurst_exponent(self, signal):
#         """
#         References:
#         ----------
#         DOI 10.1038/srep00315
#         arXiv:2310.19051 
#         """
#         segment_size, fluctuation_values = self.calculate_detrended_fluctuation_analysis(signal)
    
#         poly = np.polyfit(np.log(segment_size), np.log(fluctuation_values), 1)
#         hurst = poly[0]
#         return hurst

# # %%
# hs = StaHurst()
# hs.calculate_hurst_exponent(signal)

# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%
# import pandas as pd
# import matplotlib.pyplot as plt

# # Create the DataFrame using the provided data
# data = {
#     'timestamp': pd.date_range(start='2023-01-01', periods=17, freq='S'),
#     'accx': [1.9549347, 1.855483705, 1.843416107, 1.9166144, 2.0315752, 1.954387472,
#             2.02053782, 1.871781105, 2.1465364, 2.118728443, 1.671933377, 1.577529305,
#             1.8016534, 1.855664774, 1.949906612, 2.015232506, 1.842987431],
#     'accy': [-9.73276, -10.02151426, -10.02899607, -10.0232044, -9.862182474, -9.748202525,
#             -9.825025234, -9.839527634, -10.17570026, -9.985893718, -9.9774417, -10.07501063,
#             -10.14258728, -10.14734547, -10.384206, -10.33816296, -10.79094657],
#     'accz': [-0.361180671, -0.214025621, -0.119001998, -0.321721816, -0.567112836, -0.50792218,
#             -0.049956482, 0.163780867, -0.274856293, -0.303148124, -0.012379348, -0.099898024,
#             -0.123357798, 0.014173437, 0.354340242, 0.645211714, 1.08413891],
#     'label': [0] * 17,
#     'subject': [1] * 17
# }

# df = pd.DataFrame(data)

# # Combine accx, accy, and accz into a list for feature extraction
# features = df[['accx', 'accy', 'accz']].values.tolist()

# # Plotting the features
# plt.figure(figsize=(10, 6))

# # Extract individual components
# accx = [feature[0] for feature in features]
# accy = [feature[1] for feature in features]
# accz = [feature[2] for feature in features]
# timestamps = df['timestamp']

# plt.plot(timestamps, accx, label='accx', marker='o')
# plt.plot(timestamps, accy, label='accy', marker='o')
# plt.plot(timestamps, accz, label='accz', marker='o')

# plt.xlabel('Time')
# plt.ylabel('Acceleration')
# plt.title('Accelerometer Readings Over Time')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.show()


# # %%



