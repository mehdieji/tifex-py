import numpy as np
from scipy.signal import welch, find_peaks
from scipy.integrate import simpson
from scipy.stats import entropy, skew, kurtosis, trim_mean, mode
import librosa

class SpectralFeatures:
    def __init__(self,
                fs,
                f_bands,
                n_dom_freqs=5,
                cumulative_power_thresholds=None):
        self.fs = fs
        self.f_bands = f_bands
        self.n_dom_freqs = n_dom_freqs
        if cumulative_power_thresholds is None:
            self.cumulative_power_thresholds = np.array([.5, .75, .85, .9, 0.95])
        else:
            self.cumulative_power_thresholds = cumulative_power_thresholds

    def calculate_frequency_features(self, signal, signal_name):
        # An array for storing the spectral features.
        feats = []
        # A list for storing feature names
        feats_names = []

        # FFT (only positive frequencies)
        spectrum = np.fft.rfft(signal)  # Spectrum of positive frequencies
        spectrum_magnitudes = np.abs(spectrum)  # Magnitude of positive frequencies
        spectrum_magnitudes_normalized = spectrum_magnitudes / np.sum(spectrum_magnitudes)
        length = len(signal)
        freqs_spectrum = np.abs(np.fft.fftfreq(length, 1.0 / self.fs)[:length // 2 + 1])

        # Calculating the power spectral density using Welch's method.
        freqs_psd, psd = welch(signal, fs=self.fs)
        psd_normalized = psd / np.sum(psd)

        # Calculating the spectral features.
        # Spectral centroid (order 1-4)        
        for order in range(1, 5):
            feats.extend(self.calculate_spectral_centroid(freqs_spectrum, spectrum_magnitudes, order=order))
            feats_names.append(f"{signal_name}_spectral_centroid_order_{order}")
        
        # Spectral variance / spectral spread
        feats.extend(self.calculate_spectral_variance(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_var")

        # Spectral skewness
        feats.extend(self.calculate_spectral_skewness(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_skewness")

        # Spectral kurtosis
        feats.extend(self.calculate_spectral_kurtosis(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_kurtosis")

        # Median frequency of the power spectrum of a signal
        feats.extend(self.calculate_median_frequency(freqs_psd, psd))
        feats_names.append(f"{signal_name}_median_frequency")

        # Spectral bandwidth order 1 / Spectral mean deviation
        # https://zenodo.org/badge/latestdoi/6309729
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 1))
        feats_names.append(f"{signal_name}_spectral_mean_deviation")
        # Spectral bandwidth order 2 / Spectral standard deviation
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 2))
        feats_names.append(f"{signal_name}_spectral_std")
        # Spectral bandwidth order 3
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 3))
        feats_names.append(f"{signal_name}_spectral_bandwidth_order_3")
        # Spectral bandwidth order 4
        feats.extend(self.calculate_spectral_bandwidth(freqs_spectrum, spectrum_magnitudes, 4))
        feats_names.append(f"{signal_name}_spectral_bandwidth_order_4")

        # Spectral mean absolute deviation
        # https://zenodo.org/badge/latestdoi/6309729
        feats.extend(self.calculate_spectral_absolute_deviation(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_abs_deviation_order_1")
        # Spectral mean absolute deviation order 3
        feats.extend(self.calculate_spectral_absolute_deviation(freqs_spectrum, spectrum_magnitudes, order=3))
        feats_names.append(f"{signal_name}_spectral_abs_deviation_order_3")

        # Spectral linear slope for spectrum
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_slope_linear(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_linear_slope")

        # Spectral logarithmic slope for spectrum
        feats.extend(self.calculate_spectral_slope_logarithmic(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectrum_logarithmic_slope")

        # Spectral linear slope for psd
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_slope_linear(freqs_psd, psd))
        feats_names.append(f"{signal_name}_power_spectrum_linear_slope")

        # Spectral logarithmic slope for psd
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_slope_logarithmic(freqs_psd, psd))
        feats_names.append(f"{signal_name}_power_spectrum_logarithmic_slope")

        # Spectral flatness
        feats.extend(self.calculate_spectral_flatness(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_flatness")

        # Spectral peaks of power spectral density
        # https://doi.org/10.1016/0025-5564(71)90072-1
        feats.extend(self.calculate_peak_frequencies(freqs_psd, psd))
        for rank in range(1, self.n_dom_freqs+1):
            feats_names.append(f"{signal_name}_peak_freq_{rank}")

        # Spectral edge frequency for different thresholds
        # https://doi.org/10.1111/j.1399-6576.1991.tb03374.x
        feats.extend(self.calculate_spectral_edge_frequency(freqs_psd, psd))
        for threshold in self.cumulative_power_thresholds:
            feats_names.append(f"{signal_name}_edge_freq_thresh_{threshold}")

        # Spectral band power for different bands
        # https://www.mathworks.com/help/signal/ref/bandpower.html
        feats.extend(self.calculate_band_power(freqs_psd, psd))
        feats_names.append(f"{signal_name}_spectral_total_power")
        for band in self.f_bands:
            feats_names.append(f"{signal_name}_spectral_abs_power_band_{str(band)}")
            feats_names.append(f"{signal_name}_spectral_rel_power_band_{str(band)}")

        # Spectral entropy
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_entropy(psd))
        feats_names.append(f"{signal_name}_spectral_entropy")

        # Spectral contrast for different bands
        # https://zenodo.org/badge/latestdoi/6309729
        feats.extend(self.calculate_spectral_contrast(freqs_psd, psd))
        for band in self.f_bands:
            feats_names.append(f"{signal_name}_spectral_contrast_band_{str(band)}")
        
        # Spectral coefficient of variation
        # https://doi.org/10.21105/joss.01272
        feats.extend(self.calculate_spectral_cov(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_coefficient_of_variation")

        # Spectral flux
        # https://doi.org/10.1016/B978-0-08-099388-1.00004-2
        feats.extend(self.calculate_spectral_flux(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_flux")

        # Spectral roll-off
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_rolloff(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_rolloff")

        # Harmonic Ratio
        # https://www.mathworks.com/help/audio/ref/harmonicratio.html
        feats.extend(self.calculate_harmonic_ratio(signal))
        feats_names.append(f"{signal_name}_harmonic_ratio")

        # Fundamental Frequency
        # https://doi.org/10.1121%2F1.4740482
        feats.extend(self.calculate_fundamental_frequency(signal))
        feats_names.append(f"{signal_name}_fundamental_frequency")

        # Spectral Crest Factor
        # https://www.mathworks.com/help/signal/ref/spectralcrest.html#d126e220002
        feats.extend(self.calculate_spectral_crest_factor(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_crest_factor")

        # Spectral Decrease
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_decrease(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_decrease")

        # Spectral Irregularity
        # https://docs.twoears.eu/en/latest/afe/available-processors/spectral-features/
        # https://doi.org/10.1109/ICASSP.2004.1325955
        feats.extend(self.calculate_spectral_irregularity(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_irregularity")

        # Mean Frequency: The average frequency, weighted by amplitude.
        # https://www.mathworks.com/help/signal/ref/meanfreq.html
        feats.extend(self.calculate_mean_frequency(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_mean_frequency")

        # Frequency Winsorized Mean: A mean that reduces the effect of outliers by limiting extreme values.
        # https://doi.org/10.1007/978-94-010-0231-8_13
        feats.extend(self.calculate_frequency_winsorized_mean(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_frequency_winsorized_mean")

        # Total Harmonic Distortion: Measures the distortion due to harmonics in a signal.
        # 10.1109/TCOMM.2011.061511.100749
        # https://zenodo.org/badge/latestdoi/6309729
        feats.extend(self.calculate_total_harmonic_distortion(signal))
        feats_names.append(f"{signal_name}_total_harmonic_distortion")

        # Inharmonicity: Measures the deviation of the frequencies of the overtones from whole number multiples of the fundamental frequency.
        # https://zenodo.org/badge/latestdoi/6309729
        # feats.extend(self.calculate_inharmonicity(signal))
        # feats_names.append(f"{signal_name}_inharmonicity")

        # Tristimulus: Measures the relative amplitude of the first few harmonics.
        # https://zenodo.org/badge/latestdoi/6309729
        feats.extend(self.calculate_tristimulus(spectrum_magnitudes))
        feats_names.extend([f"{signal_name}_tristimulus_1", f"{signal_name}_tristimulus_2", f"{signal_name}_tristimulus_3"])

        # Spectral Roll-On: The opposite of spectral roll-off, measuring the frequency above which a certain percentage of the total spectral energy is contained.
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_rollon(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_rollon")

        # Spectral Hole Count: The number of significant dips in the spectrum.
        # https://doi.org/10.1103/PhysRevA.104.063111
        feats.extend(self.calculate_spectral_hole_count(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_hole_count")

        # Spectral Auto-correlation: The auto-correlation of the spectrum.
        # https://doi.org/10.48550/arXiv.1702.00105
        feats.append(self.calculate_spectral_autocorrelation(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_autocorrelation")

        # Spectral Variability: Measures the variability of the spectrum.
        # https://doi.org/10.1016/j.dsp.2015.10.011
        feats.extend(self.calculate_spectral_variability(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_variability")

        # Spectral Spread Ratio: The ratio of spectral spread to some reference value.
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_spread_ratio(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_spread_ratio")

        # Spectral Skewness Ratio: The ratio of spectral skewness to some reference value.
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_skewness_ratio(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_skewness_ratio")

        # Spectral Kurtosis Ratio: The ratio of spectral kurtosis to some reference value.
        # https://doi.org/10.1016/j.softx.2020.100456
        feats.extend(self.calculate_spectral_kurtosis_ratio(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_kurtosis_ratio")

        # Spectral Tonal Power Ratio: The ratio of tonal power to total power.
        # https://zenodo.org/badge/latestdoi/6309729
        feats.extend(self.calculate_spectral_tonal_power_ratio(signal))
        feats_names.append(f"{signal_name}_spectral_tonal_power_ratio")

        # Spectral Noise to Harmonics Ratio: The ratio of noise energy to harmonic energy.
        # https://zenodo.org/badge/latestdoi/6309729
        feats.extend(self.calculate_spectral_noise_to_harmonics_ratio(signal))
        feats_names.append(f"{signal_name}_spectral_noise_to_harmonics_ratio")

        # Spectral Even to Odd Harmonic Energy Ratio: The ratio of even harmonic energy to odd harmonic energy.
        # https://zenodo.org/badge/latestdoi/6309729
        # feats.extend(self.calculate_spectral_even_to_odd_harmonic_energy_ratio(signal))
        # feats_names.append(f"{signal_name}_spectral_even_to_odd_harmonic_energy_ratio")

        # Spectral Strongest Frequency Phase: The phase of the strongest frequency component.
        # https://mriquestions.com/phase-v-frequency.html
        feats.extend(self.calculate_spectral_strongest_frequency_phase(freqs_spectrum, spectrum))
        feats_names.append(f"{signal_name}_spectral_strongest_frequency_phase")

        # Spectral Frequency Below Peak: Frequency below the peak frequency.
        # https://doi.org/10.1016/B978-012437552-9/50003-9
        feats.extend(self.calculate_spectral_frequency_below_peak(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_frequency_below_peak")

        # Spectral Frequency Above Peak: Frequency above the peak frequency (next frequency value after the peak).
        # https://doi.org/10.1016/B978-012437552-9/50003-9
        feats.extend(self.calculate_spectral_frequency_above_peak(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_frequency_above_peak")

        # Spectral Cumulative Frequency Below 50% Power: The frequency below which 50% of the spectral power is contained.
        # https://doi.org/10.48550/arXiv.0901.3708
        feats.extend(self.calculate_spectral_cumulative_frequency(freqs_spectrum, spectrum_magnitudes, 0.5))
        feats_names.append(f"{signal_name}_spectral_cumulative_frequency_below_50_percent_power")

        # Spectral Cumulative Frequency Below 75% Power: The frequency below which 75% of the spectral power is contained.
        # https://doi.org/10.48550/arXiv.0901.3708
        feats.extend(self.calculate_spectral_cumulative_frequency(freqs_spectrum, spectrum_magnitudes, 0.75))
        feats_names.append(f"{signal_name}_spectral_cumulative_frequency_below_75_percent_power")

        # Spectral Cumulative Frequency Above 75% Power: The frequency above which 75% of the spectral power is contained.
        # https://doi.org/10.48550/arXiv.0901.3708
        feats.extend(self.calculate_spectral_cumulative_frequency_above(freqs_spectrum, spectrum_magnitudes, 0.75))
        feats_names.append(f"{signal_name}_spectral_cumulative_frequency_above_75_percent_power")

        # Spectral Spread Shift: The change in spectral spread over time.
        # https://docs.twoears.eu/en/latest/afe/available-processors/spectral-features/
        feats.extend(self.calculate_spectral_spread_shift(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_spread_shift")

        # Spectral Entropy Shift: The change in spectral entropy over time.
        # https://doi.org/10.3390/buildings12030310
        feats.extend(self.calculate_spectral_entropy_shift(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_entropy_shift")

        # Spectral Change Vector Magnitude: The magnitude of change in the spectral features over time.
        # https://doi.org/10.3390/rs3112473
        feats.extend(self.calculate_spectral_change_vector_magnitude(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_change_vector_magnitude")

        # Spectral Low Frequency Content: The amount of energy in the low-frequency band.
        # https://resources.pcb.cadence.com/blog/2022-an-overview-of-frequency-bands-and-their-applications
        feats.extend(self.calculate_spectral_low_frequency_content(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_low_frequency_content")

        # Spectral Mid Frequency Content: The amount of energy in the mid-frequency band.
        # https://resources.pcb.cadence.com/blog/2022-an-overview-of-frequency-bands-and-their-applications
        feats.extend(self.calculate_spectral_mid_frequency_content(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_mid_frequency_content")

        # Spectral Peak-to-Valley Ratio: The ratio of peak to valley values in the spectrum.
        # https://doi.org/10.3389/fpsyg.2022.994047
        # https://openlab.help.agilent.com/en/index.htm#t=mergedProjects/DataAnalysis/27021601168830603.htm
        feats.extend(self.calculate_spectral_peak_to_valley_ratio(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_peak_to_valley_ratio")

        # Spectral Valley Depth Mean: The mean depth of valleys in the spectrum.
        # https://doi.org/10.48550/arXiv.1506.04828
        feats.extend(self.calculate_spectral_valley_depth_mean(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_valley_depth_mean")

        # Spectral Valley Depth Standard Deviation: The standard deviation of valley depths.
        # https://doi.org/10.48550/arXiv.1506.04828
        feats.extend(self.calculate_spectral_valley_depth_std(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_valley_depth_std")

        # Spectral Valley Depth Variance: The variance of valley depths.
        # https://doi.org/10.48550/arXiv.1506.04828
        feats.extend(self.calculate_spectral_valley_depth_variance(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_valley_depth_variance")

        # Spectral Valley Width Mode: The most frequent valley width.
        # https://doi.org/10.48550/arXiv.1506.04828
        feats.extend(self.calculate_spectral_valley_width_mode(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_valley_width_mode")

        # Spectral Valley Width Standard Deviation: The standard deviation of valley widths.
        # https://doi.org/10.48550/arXiv.1506.04828
        feats.extend(self.calculate_spectral_valley_width_std(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_valley_width_std")

        # Spectral Subdominant Valley: The second most prominent valley in the spectrum.
        feats.extend(self.calculate_spectral_subdominant_valley(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_subdominant_valley")

        # Spectral Valley Count: The number of valleys in the spectrum.
        # https://doi.org/10.48550/arXiv.1506.04828
        feats.extend(self.calculate_spectral_valley_count(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_valley_count")

        # Spectral Peak Broadness: The width of the spectral peaks.
        # https://terpconnect.umd.edu/~toh/spectrum/PeakFindingandMeasurement.htm
        feats.extend(self.calculate_spectral_peak_broadness(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_peak_broadness")

        # Spectral Valley Broadness: The width of the spectral valleys.
        # https://doi.org/10.48550/arXiv.1506.04828
        feats.extend(self.calculate_spectral_valley_broadness(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_spectral_valley_broadness")

        # Frequency Variance: Variance of the frequencies weighted by amplitude.
        # https://doi.org/10.1016/B978-0-12-811153-6.00003-8
        feats.extend(self.calculate_frequency_variance(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_frequency_variance")

        # Frequency Standard Deviation: Standard deviation of the frequencies weighted by amplitude.
        # https://doi.org/10.1016/B978-0-12-811153-6.00003-8
        feats.extend(self.calculate_frequency_std(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_frequency_std")

        # Frequency Range: Range of the frequencies.
        # https://doi.org/10.1016/B978-0-12-811153-6.00003-8
        feats.extend(self.calculate_frequency_range(freqs_spectrum))
        feats_names.append(f"{signal_name}_frequency_range")

        # Frequency Trimmed Mean: The mean frequency after trimming a percentage of the highest and lowest values.
        # https://doi.org/10.1016/B978-0-12-811153-6.00003-8
        feats.extend(self.calculate_frequency_trimmed_mean(freqs_spectrum, spectrum_magnitudes))
        feats_names.append(f"{signal_name}_frequency_trimmed_mean")

        # Harmonic Product Spectrum: The product of harmonics in the spectrum.
        # 10.1109/MHS.2018.8886911
        feats.extend(self.calculate_harmonic_product_spectrum(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_harmonic_product_spectrum")

        # Smoothness: Measures the smoothness of the spectrum.
        # https://doi.org/10.3390/rs13163196
        feats.extend(self.calculate_smoothness(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_smoothness")

        # Roughness: Measures the roughness of the spectrum.
        feats.extend(self.calculate_roughness(spectrum_magnitudes))
        feats_names.append(f"{signal_name}_roughness")


        # Returning the spectral features and the feature names
        # return np.array(feats), feats_names
        return feats, feats_names

    def calculate_spectral_centroid(self, freqs, magnitudes, order=1):
        """
        Calculate the spectral centroid of a given spectrum.

        The spectral centroid is a measure that indicates where the center of mass of the spectrum is located.
        It is often associated with the perceived brightness of a sound. This function computes the spectral
        centroid by taking the weighted mean of the frequencies, with the magnitudes as weights.

        Parameters:
        ----------
        freqs : np.array
            An array of frequencies corresponding to the spectrum bins.
        magnitudes : np.array
            An array of magnitude values of the spectrum at the corresponding frequencies.
        order : int, optional
            The order of the centroid calculation. Default is 1, which calculates the standard spectral centroid.
            Higher orders can be used for other types of spectral centroid calculations.

        Returns:
        -------
        np.array
            An array containing the calculated spectral centroid. The array is of length 1 for consistency in return type.
            
        Reference:
        ---------
            [1] Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P., Liu, H., Schultz, 
                T., & Gamboa, H. (2020). TSFEL: Time Series Feature Extraction Library. SoftwareX, 11. 
                https://doi.org/10.1016/j.softx.2020.100456
        """
                
        spectral_centroid = np.sum(magnitudes * (freqs ** order)) / np.sum(magnitudes)
        return np.array([spectral_centroid])

    def calculate_spectral_variance(self, freqs, magnitudes):
        """
        Calculate the spectral variance (also known as spectral spread) of a given spectrum.

        The spectral variance is a measure of the spread of the spectrum around its centroid.
        It quantifies how much the frequencies in the spectrum deviate from the spectral centroid.

        Parameters:
        ----------
        freqs : np.array
            An array of frequencies corresponding to the spectrum bins.
        magnitudes : np.array
            An array of magnitude values of the spectrum at the corresponding frequencies.

        Returns:
        -------
        np.array
            An array containing the calculated spectral variance. The array is of length 1 for consistency in return type.
        
        Reference:
        ---------
            https://doi.org/10.1016/j.softx.2020.100456
        """
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        spectral_variance = np.sum(((freqs - mean_frequency) ** 2) * magnitudes) / np.sum(magnitudes)
        return np.array([spectral_variance])

    def calculate_spectral_skewness(self, freqs, magnitudes):
        """
        Calculate the spectral skewness of a given spectrum.

        Spectral skewness is a measure of the asymmetry of the distribution of frequencies in the spectrum
        around the spectral centroid. It indicates whether the spectrum is skewed towards higher or lower
        frequencies.

        Parameters:
        ----------
        freqs : np.array
            An array of frequencies corresponding to the spectrum bins.
        magnitudes : np.array
            An array of magnitude values of the spectrum at the corresponding frequencies.

        Returns:
        -------
        float
            The calculated spectral skewness.
        
        Reference:
        ---------
            https://doi.org/10.1016/j.softx.2020.100456
        """
        mu1 = self.calculate_spectral_centroid(freqs, magnitudes, order=1)
        mu2 = self.calculate_spectral_centroid(freqs, magnitudes, order=2)
        spectral_skewness = np.sum(magnitudes * (freqs - mu1) ** 3) / (np.sum(magnitudes) * mu2 ** 3)
        return spectral_skewness

    def calculate_spectral_kurtosis(self, freqs, magnitudes):
        """
        Calculate the spectral kurtosis of a given spectrum.

        Spectral kurtosis measures the "tailedness" or peakiness of the frequency distribution around the spectral centroid.
        It quantifies how outlier-prone the spectrum is and reflects the degree of concentration of the spectral energy.
        A higher kurtosis value indicates a more peaked distribution with heavy tails.

        Parameters:
        ----------
        freqs : np.array
            An array of frequencies corresponding to the spectrum bins.
        magnitudes : np.array
            An array of magnitude values of the spectrum at the corresponding frequencies.

        Returns:
        -------
        float
            The calculated spectral kurtosis.
        
        Reference:
        ---------
            https://doi.org/10.1016/j.softx.2020.100456
        """
        mu1 = self.calculate_spectral_centroid(freqs, magnitudes, order=1)
        mu2 = self.calculate_spectral_centroid(freqs, magnitudes, order=2)
        spectral_kurtosis = np.sum(magnitudes * (freqs - mu1) ** 4) / (np.sum(magnitudes) * mu2 ** 4)
        return spectral_kurtosis

    def calculate_median_frequency(self, freqs, psd):
        """
        Calculate the cumulative distribution function (CDF) of the PSD

        Parameters:
        ----------
        freqs : np.array
            An array of frequencies corresponding to the PSD bins.
        psd : np.array
            An array of power spectral density values at the corresponding frequencies.

        Returns:
        -------
        np.array
            An array containing the calculated median frequency.
            
        Reference:
        ---------
                https://doi.org/10.1109/iembs.2008.4649357
        """
        cdf = np.cumsum(psd)
        median_freq = freqs[np.searchsorted(cdf, cdf[-1] / 2)]
        return np.array([median_freq])

    def calculate_spectral_flatness(self, magnitudes):
        """
        Calculate the spectral flatness of a given spectrum.

        Spectral flatness quantifies how flat or peaky a spectrum is, indicating how noise-like or tonal a signal is.
        It is often used to distinguish between noise and tonal signals. It can also be referred to as Wiener's entropy.

        Parameters:
        ----------
        magnitudes : np.array
            An array of magnitude values of the spectrum.

        Returns:
        -------
        np.array
            An array containing the calculated spectral flatness.

        Reference:
        --------
            https://doi.org/10.1016/B978-0-12-398499-9.00012-1
        """
        spectral_flatness = np.exp(np.mean(np.log(magnitudes))) / np.mean(magnitudes)
        return np.array([spectral_flatness])

    def calculate_spectral_slope_logarithmic(self, freqs, magnitudes):
        """
        Calculate the logarithmic spectral slope of a given spectrum.

        The logarithmic spectral slope provides a measure of the rate at which the spectrum's magnitude changes
        across frequencies on a logarithmic scale.

        Parameters:
        ----------
        freqs : np.array
            An array of frequencies corresponding to the magnitude spectrum bins.
        magnitudes : np.array
            An array of magnitude values of the spectrum at the corresponding frequencies.

        Returns:
        -------
        np.array
            An array containing the calculated logarithmic spectral slope. The array is of length 1 for consistency in return type.

        Reference:
        ---------
            https://doi.org/10.1016/j.softx.2020.100456
        """
        slope = np.polyfit(freqs, np.log(magnitudes), 1)[0]
        return np.array([slope])

    def calculate_spectral_slope_linear(self, freqs, magnitudes):
        slope = np.polyfit(freqs, magnitudes, 1)[0]
        return np.array([slope])

    def calculate_peak_frequencies(self, freqs, psd):
        peak_frequencies = freqs[np.argsort(psd)[-self.n_dom_freqs:][::-1]]
        return np.array(peak_frequencies)

    def calculate_spectral_edge_frequency(self, freqs, psd):
        # A special case would be roll-off frequency (threshold = .85)
        feats = []
        cumulative_power = np.cumsum(psd) / np.sum(psd)
        for threshold in self.cumulative_power_thresholds:
            feats.append(freqs[np.argmax(cumulative_power >= threshold)])
        return np.array(feats)

    def calculate_band_power(self, freqs, psd):
        # The features array for storing the total power, band absolute powers, and band relative powers
        feats = []
        freq_res = freqs[1] - freqs[0]  # Frequency resolution
        # Calculate the total power of the signal
        try:
            feats.append(simpson(psd, dx=freq_res))
        except:
            feats.append(np.nan)
        # Calculate band absolute and relative power
        for f_band in self.f_bands:
            try:
                # Keeping the frequencies within the band
                idx_band = np.logical_and(freqs >= f_band[0], freqs < f_band[1])
                # Absolute band power by integrating PSD over frequency range of interest
                feats.append(simpson(psd[idx_band], dx=freq_res))
                # Relative band power
                feats.append(feats[-1] / feats[0])
            except:
                feats.extend([np.nan, np.nan])
        return np.array(feats)

    def calculate_spectral_entropy(self, psd):
        try:
            # Formula from Matlab doc
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        except:
            spectral_entropy = np.nan
        return np.array([spectral_entropy])

    def calculate_spectral_contrast(self, freqs, psd):
        feats = []
        for f_band in self.f_bands:
            try:
                idx_band = np.logical_and(freqs >= f_band[0], freqs < f_band[1])
                peak = np.max(psd[idx_band])
                valley = np.min(psd[idx_band])
                contrast = peak - valley
                feats.append(contrast)
            except:
                feats.append(np.nan)
        return np.array(feats)

    def calculate_spectral_bandwidth(self, freqs, magnitudes, order):
        # Definition from Librosa library (with normalized magnitudes)
        # The 1st order spectral bandwidth is the same as spectral mean deviation
        # The 2nd order spectral bandwidth is the same as spectral standard deviation
        normalized_magnitudes = magnitudes / np.sum(magnitudes)
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        spectral_bandwidth = ((np.sum(((freqs - mean_frequency) ** order) * normalized_magnitudes)) ** (1 / order))
        return np.array([spectral_bandwidth])

    def calculate_spectral_absolute_deviation(self, freqs, magnitudes, order=1):
        # Definition from Librosa library
        # The even order spectral absolute deviation is the same as spectral bandwidth of the same order
        normalized_magnitudes = magnitudes / np.sum(magnitudes)
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        spectral_absolute_deviation = ((np.sum((np.abs(freqs - mean_frequency) ** order) * normalized_magnitudes)) ** (1 / order))
        return np.array([spectral_absolute_deviation])

    def calculate_spectral_cov(self, freqs, magnitudes):
        mean_frequency = self.calculate_spectral_centroid(freqs, magnitudes)
        frequency_std = self.calculate_spectral_bandwidth(freqs, magnitudes, 2)
        coefficient_of_variation = (frequency_std / mean_frequency) * 100
        return coefficient_of_variation

    def calculate_spectral_flux(self, magnitudes, order=2):
        spectral_flux = (np.sum(np.abs(np.diff(magnitudes)) ** order)) ** (1 / order)
        return np.array([spectral_flux])
    
    def calculate_spectral_rolloff(self, freqs, magnitudes, roll_percent=0.85):
        cumulative_magnitudes = np.cumsum(magnitudes)
        rolloff_frequency = np.min(freqs[np.where(cumulative_magnitudes >= roll_percent * cumulative_magnitudes[-1])])
        return np.array([rolloff_frequency])

    def calculate_harmonic_ratio(self, signal):
        harmonic_ratio = librosa.effects.harmonic(signal).mean()
        return np.array([harmonic_ratio])

    def calculate_fundamental_frequency(self, signal):
        f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
        return np.array([np.mean(f0)])

    def calculate_spectral_crest_factor(self, magnitudes):
        crest_factor = np.max(magnitudes) / np.mean(magnitudes)
        return np.array([crest_factor])

    def calculate_spectral_decrease(self, freqs, magnitudes):
        k = np.arange(1, len(magnitudes) + 1)
        spectral_decrease = np.sum((magnitudes[1:] - magnitudes[0]) / k[1:])
        return np.array([spectral_decrease])

    def calculate_spectral_irregularity(self, magnitudes):
        irregularity = np.sum(np.abs(magnitudes[1:] - magnitudes[:-1])) / (len(magnitudes) - 1)
        return np.array([irregularity])

    def calculate_mean_frequency(self, freqs, magnitudes):
        mean_freq = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        return np.array([mean_freq])

    def calculate_frequency_winsorized_mean(self, freqs, magnitudes, limits=(0.05, 0.95)):
        sorted_indices = np.argsort(magnitudes)
        lower_limit = int(limits[0] * len(magnitudes))
        upper_limit = int(limits[1] * len(magnitudes))
        trimmed_indices = sorted_indices[lower_limit:upper_limit]
        winsorized_mean = np.mean(freqs[trimmed_indices])
        return np.array([winsorized_mean])

    def calculate_total_harmonic_distortion(self, signal, harmonics=5):
        f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
        fundamental_freq = np.mean(f0)
        harmonic_frequencies = [(i+1) * fundamental_freq for i in range(harmonics)]
        harmonic_power = sum([np.sum(np.abs(np.fft.rfft(signal * np.sin(2 * np.pi * harmonic_freq * np.arange(len(signal)) / self.fs)))) for harmonic_freq in harmonic_frequencies])
        total_power = np.sum(np.abs(np.fft.rfft(signal))**2)
        thd = harmonic_power / total_power
        return np.array([thd])

    # def calculate_inharmonicity(self, signal):
    #     f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    #     fundamental_freq = np.mean(f0)
    #     harmonics = [(i+1) * fundamental_freq for i in range(1, int(self.fs/(2*fundamental_freq)))]
    #     inharmonicity = sum([np.abs(harmonic - fundamental_freq * (i+1)) for i, harmonic in enumerate(harmonics)]) / len(harmonics)
    #     return np.array([inharmonicity])

    def calculate_tristimulus(self, magnitudes):
        if len(magnitudes) < 3:
            return np.array([np.nan, np.nan, np.nan])
        t1 = magnitudes[0] / np.sum(magnitudes)
        t2 = magnitudes[1] / np.sum(magnitudes)
        t3 = np.sum(magnitudes[2:]) / np.sum(magnitudes)
        return np.array([t1, t2, t3])

    def calculate_spectral_rollon(self, freqs, magnitudes, roll_percent=0.85):
        cumulative_magnitudes = np.cumsum(magnitudes)
        rollon_frequency = np.min(freqs[np.where(cumulative_magnitudes >= roll_percent * cumulative_magnitudes[-1])])
        return np.array([rollon_frequency])

    def calculate_spectral_hole_count(self, magnitudes, threshold=0.05):
        peaks, _ = find_peaks(magnitudes, height=threshold)
        dips, _ = find_peaks(-magnitudes, height=-threshold)
        return np.array([len(dips)])

    def calculate_spectral_autocorrelation(self, magnitudes):
        autocorrelation = np.correlate(magnitudes, magnitudes, mode='full')
        return autocorrelation[autocorrelation.size // 2:]

    def calculate_spectral_variability(self, magnitudes):
        variability = np.var(magnitudes)
        return np.array([variability])

    def calculate_spectral_spread_ratio(self, freqs, magnitudes, reference_value=1.0):
        spread = np.sqrt(np.sum((freqs - np.mean(freqs))**2 * magnitudes) / np.sum(magnitudes))
        spread_ratio = spread / reference_value
        return np.array([spread_ratio])

    def calculate_spectral_skewness_ratio(self, freqs, magnitudes, reference_value=1.0):
        mean_freq = np.mean(freqs)
        skewness = np.sum((freqs - mean_freq)**3 * magnitudes) / (len(freqs) * (np.std(freqs)**3))
        skewness_ratio = skewness / reference_value
        return np.array([skewness_ratio])

    def calculate_spectral_kurtosis_ratio(self, freqs, magnitudes, reference_value=1.0):
        mean_freq = np.mean(freqs)
        kurtosis = np.sum((freqs - mean_freq)**4 * magnitudes) / (len(freqs) * (np.std(freqs)**4)) - 3
        kurtosis_ratio = kurtosis / reference_value
        return np.array([kurtosis_ratio])

    def calculate_spectral_tonal_power_ratio(self, signal):
        harmonic_power = np.sum(librosa.effects.harmonic(signal)**2)
        total_power = np.sum(signal**2)
        tonal_power_ratio = harmonic_power / total_power
        return np.array([tonal_power_ratio])

    def calculate_spectral_noise_to_harmonics_ratio(self, signal):
        harmonic_part = librosa.effects.harmonic(signal)
        noise_part = signal - harmonic_part
        noise_energy = np.sum(noise_part**2)
        harmonic_energy = np.sum(harmonic_part**2)
        noise_to_harmonics_ratio = noise_energy / harmonic_energy
        return np.array([noise_to_harmonics_ratio])

    # def calculate_spectral_even_to_odd_harmonic_energy_ratio(self, signal):
    #     f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    #     fundamental_freq = np.mean(f0)
    #     even_harmonics = [(2 * i + 2) * fundamental_freq for i in range(int(self.fs / (2 * fundamental_freq)))]
    #     odd_harmonics = [(2 * i + 1) * fundamental_freq for i in range(int(self.fs / (2 * fundamental_freq)))]
    #     even_energy = sum([np.sum(np.abs(np.fft.rfft(signal * np.sin(2 * np.pi * harmonic * np.arange(len(signal)) / self.fs)))) for harmonic in even_harmonics])
    #     odd_energy = sum([np.sum(np.abs(np.fft.rfft(signal * np.sin(2 * np.pi * harmonic * np.arange(len(signal)) / self.fs)))) for harmonic in odd_harmonics])
    #     even_to_odd_ratio = even_energy / odd_energy
    #     return np.array([even_to_odd_ratio])

    def calculate_spectral_strongest_frequency_phase(self, freqs, spectrum):
        strongest_frequency_index = np.argmax(np.abs(spectrum))
        phase = np.angle(spectrum[strongest_frequency_index])
        return np.array([phase])

    def calculate_spectral_frequency_below_peak(self, freqs, magnitudes):
        peak_index = np.argmax(magnitudes)
        frequency_below_peak = freqs[max(0, peak_index - 1)]
        return np.array([frequency_below_peak])

    def calculate_spectral_frequency_above_peak(self, freqs, magnitudes):
        peak_index = np.argmax(magnitudes)
        frequency_above_peak = freqs[min(len(freqs) - 1, peak_index + 1)]
        return np.array([frequency_above_peak])

    def calculate_spectral_cumulative_frequency(self, freqs, magnitudes, threshold):
        cumulative_power = np.cumsum(magnitudes) / np.sum(magnitudes)
        frequency = freqs[np.where(cumulative_power >= threshold)[0][0]]
        return np.array([frequency])

    def calculate_spectral_cumulative_frequency_above(self, freqs, magnitudes, threshold):
        cumulative_power = np.cumsum(magnitudes) / np.sum(magnitudes)
        frequency = freqs[np.where(cumulative_power <= threshold)[-1][-1]]
        return np.array([frequency])

    def calculate_spectral_spread_shift(self, freqs, magnitudes):
        mean_frequency = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        spread = np.sqrt(np.sum((freqs - mean_frequency) ** 2 * magnitudes) / np.sum(magnitudes))
        return np.array([spread])

    def calculate_spectral_entropy_shift(self, magnitudes):
        psd_norm = magnitudes / np.sum(magnitudes)
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        return np.array([entropy])

    def calculate_spectral_change_vector_magnitude(self, magnitudes):
        change_vector_magnitude = np.linalg.norm(np.diff(magnitudes))
        return np.array([change_vector_magnitude])

    def calculate_spectral_low_frequency_content(self, freqs, magnitudes, low_freq_threshold=300):
        low_freq_content = np.sum(magnitudes[freqs < low_freq_threshold])
        return np.array([low_freq_content])

    def calculate_spectral_mid_frequency_content(self, freqs, magnitudes, mid_freq_range=(300, 3000)):
        mid_freq_content = np.sum(magnitudes[(freqs >= mid_freq_range[0]) & (freqs <= mid_freq_range[1])])
        return np.array([mid_freq_content])

    def calculate_spectral_peak_to_valley_ratio(self, magnitudes):
        peaks, _ = find_peaks(magnitudes)
        valleys, _ = find_peaks(-magnitudes)
        if len(peaks) == 0 or len(valleys) == 0:
            return np.array([np.nan])
        peak_to_valley_ratio = np.max(magnitudes[peaks]) / np.min(magnitudes[valleys])
        return np.array([peak_to_valley_ratio])

    def calculate_spectral_valley_depth_mean(self, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        if len(valleys) == 0:
            return np.array([np.nan])
        valley_depth_mean = np.mean(magnitudes[valleys])
        return np.array([valley_depth_mean])

    def calculate_spectral_valley_depth_std(self, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        if len(valleys) == 0:
            return np.array([np.nan])
        valley_depth_std = np.std(magnitudes[valleys])
        return np.array([valley_depth_std])

    def calculate_spectral_valley_depth_variance(self, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        if len(valleys) == 0:
            return np.array([np.nan])
        valley_depth_variance = np.var(magnitudes[valleys])
        return np.array([valley_depth_variance])

    def calculate_spectral_valley_width_mode(self, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        if len(valleys) < 2:
            return np.array([np.nan])
        valley_widths = np.diff(valleys)
        # valley_width_mode = mode(valley_widths).mode[0]
        valley_width_mode = mode(valley_widths)[0]
        return np.array([valley_width_mode])

    def calculate_spectral_valley_width_std(self, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        if len(valleys) < 2:
            return np.array([np.nan])
        valley_widths = np.diff(valleys)
        valley_width_std = np.std(valley_widths)
        return np.array([valley_width_std])

    def calculate_spectral_subdominant_valley(self, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        if len(valleys) < 2:
            return np.array([np.nan])
        sorted_valleys = np.sort(magnitudes[valleys])
        subdominant_valley = sorted_valleys[-2] if len(sorted_valleys) >= 2 else np.nan
        return np.array([subdominant_valley])

    def calculate_spectral_valley_count(self, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        return np.array([len(valleys)])

    def calculate_spectral_peak_broadness(self, freqs, magnitudes):
        peaks, _ = find_peaks(magnitudes)
        if len(peaks) < 2:
            return np.array([np.nan])
        peak_widths = np.diff(peaks)
        peak_broadness = np.mean(peak_widths)
        return np.array([peak_broadness])

    def calculate_spectral_valley_broadness(self, freqs, magnitudes):
        valleys, _ = find_peaks(-magnitudes)
        if len(valleys) < 2:
            return np.array([np.nan])
        valley_widths = np.diff(valleys)
        valley_broadness = np.mean(valley_widths)
        return np.array([valley_broadness])

    def calculate_frequency_variance(self, freqs, magnitudes):
        mean_freq = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        variance = np.sum(((freqs - mean_freq) ** 2) * magnitudes) / np.sum(magnitudes)
        return np.array([variance])

    def calculate_frequency_std(self, freqs, magnitudes):
        mean_freq = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        variance = np.sum(((freqs - mean_freq) ** 2) * magnitudes) / np.sum(magnitudes)
        std_dev = np.sqrt(variance)
        return np.array([std_dev])

    def calculate_frequency_range(self, freqs):
        freq_range = np.max(freqs) - np.min(freqs)
        return np.array([freq_range])

    def calculate_frequency_trimmed_mean(self, freqs, magnitudes, trim_percent=0.1):
        sorted_indices = np.argsort(magnitudes)
        lower_limit = int(trim_percent * len(magnitudes))
        upper_limit = int((1 - trim_percent) * len(magnitudes))
        trimmed_indices = sorted_indices[lower_limit:upper_limit]
        trimmed_mean = np.mean(freqs[trimmed_indices])
        return np.array([trimmed_mean])
    
    def calculate_harmonic_product_spectrum(self, magnitudes):
        hps = np.copy(magnitudes)
        for h in range(2, 5):
            decimated = magnitudes[::h]
            hps[:len(decimated)] *= decimated
        return np.array([np.sum(hps)])

    def calculate_smoothness(self, magnitudes):
        smoothness = np.sum(np.diff(magnitudes)**2)
        return np.array([smoothness])

    def calculate_roughness(self, magnitudes):
        roughness = np.sum(np.abs(np.diff(magnitudes)))
        return np.array([roughness])
    

# features haven't been implemented yet and cannot find reference
# https://musicinformationretrieval.com/spectral_features.html
# Spectral Strongest Frequency Magnitude: Magnitude of the strongest frequency component.
# Spectral Strongest Frequency: The strongest frequency component.
# Spectral Frequency at Median Power: The frequency at which the power is median.
# Spectral Cumulative Frequency Below 25% Power: The frequency below which 25% of the spectral power is contained.
# Spectral Cumulative Frequency Above 25% Power: The frequency above which 25% of the spectral power is contained.
# Spectral Cumulative Frequency Above 50% Power: The frequency above which 50% of the spectral power is contained.
# Spectral Power Ratio (between different bands): The ratio of power between different frequency bands.
# Spectral Centroid Shift: The change in spectral centroid over time.
# Spectral Flux Shift: The change in spectral flux over time.
# Spectral Rolloff Shift: The change in spectral rolloff over time.
# Spectral Energy Vector: Vector of spectral energy values.
# Spectral High Frequency Content: The amount of energy in the high-frequency band.
# Spectral Peak Distribution: The distribution of peaks in the spectrum.
# Spectral Valley Distribution: The distribution of valleys in the spectrum.
# Spectral Valley Depth Median: The median depth of valleys in the spectrum.
# Spectral Valley Depth Mode: The most frequent valley depth.
# Spectral Valley Width Mean: The mean width of valleys.
# Spectral Valley Width Median: The median width of valleys.
# Spectral Valley Width Variance: The variance of valley widths.
# Spectral Dominant Valley: The most prominent valley.
# Spectral Peak Sharpness: The sharpness of the spectral peaks.
# Spectral Valley Sharpness: The sharpness of the spectral valleys.
# Spectral Dominant Peak: The most prominent peak.
# Spectral Subdominant Peak: The second most prominent peak.