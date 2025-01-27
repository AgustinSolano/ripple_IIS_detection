import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt

class SpikeWaveDetectorClass:
    def __init__(self, samplingRate):
        
        # General constants
        #self.samplingRate = 1000
        self.samplingRate = samplingRate
        self.minDistSpikes = 50  # minimal distance for 'different' spikes - in milliseconds

        # Plotting constants
        self.plotBeforeAfter = 5000  # constant for the plotting method - how much to plot before and after the peak, in ms
        self.blockSizePlot = 1

        # Constants for detection based on frequency analysis
        # the thresholds for the standard deviation are based on the papers
        # Andrillon et al (envelope condition) and Starestina et al (other
        # conditions)
        self.SDthresholdEnv = 5  # threshold in standard deviations for the envelope after bandpass (HP)
        self.SDthresholdAmp = 20  # threshold in standard deviations for the amplitude
        self.SDthresholdGrad = 20  # threshold in standard deviations for the gradient
        self.SDthresholdConjAmp = 5  # threshold in standard deviations for the amplitude for the conjunction of amp&grad condition
        self.SDthresholdConjGrad = 5  # threshold in standard deviations for the gradient for the conjunction of amp&grad condition
        self.SDthresholdConjEnv = 5  # threshold in standard deviations for the HP for the conjunction of amp&HP condition
        self.useEnv = True
        self.useAmp = False
        self.useGrad = False
        self.useConjAmpGrad = True
        self.useConjAmpEnv = False
        self.isDisjunction = True
        self.blockSizeSec = 30  # filter and find peaks at blocks of X seconds - based on Andrillon et al

        # the bandpass range is based on Andrillon et al
        # self.lowCut = 50 %low bound for band pass
        # self.highCut = 150 %high bound for band pass
        # self.minLengthSpike = 5 %a spike is detected if there are points for X ms passing the threshold - in ms, based on Andrillon et al

        # The highpass range is based on Staresina et al
        self.lowCut = 250  # low bound for high pass (Staresina)
        self.highCut = np.inf  # high bound for bandpass
        self.minLengthSpike = 5
        self.maxLengthSpike = 70  # mSec

        self.conditionsArrayTrueIfAny = False
        self.percentageOfNansAllowedArounsSpike = 0.1  # how many NaNs are allowed in the vicinity of the spike
        self.HighBandPassScore = 11  # this is for debugging - finding especially high STDs for HP

        # Constants from the original version of frequency analysis detection
        self.nPointsBlockSizeFiltfilt = 2 * 10**6  # This constant is used in the original version of the code

        # Constants for bandpass
        self.defaultFilterOrder = 1
        self.nanWarning = 0.01

        # Constants for detections based on wavelets - taken from the paper West et al
        self.scaleForCWS = np.arange(1, 31)
        self.thresholdForScale1 = 400
        self.thresholdForScale2 = 400
        self.scale1 = 3
        self.scale2 = 7
        self.scaleSlow = 28
        self.c1 = 1
        self.c2 = 1
        self.tau = 0.125  # in seconds

        # Constants for detection based on Taegar energy - taken from the paper Zaveri et al
        self.minFreqT = 10
        self.maxFreqT = 70
        self.derBefore = 8  # ms, based on the paper
        self.derAfter = 12  # ms
        self.threshMah = 200
        self.nanThresh = 0.5

        # Sleep scoring
        self.NREM = 1

    # frequency analysis detection
    def detect_times(self, data, return_peak_stats=False, sleep_scoring_vec=None):
        """
        Detects peaks in the input data using frequency analysis.

        Parameters:
        - data (numpy.ndarray): Input signal.
        - return_peak_stats (bool): If True, returns additional statistics for each detected peak.
        - sleep_scoring_vec (numpy.ndarray or None): Sleep scoring vector.

        Returns:
        - peak_times (numpy.ndarray): Array of peak times.
        - peak_stats (dict): Additional statistics for each detected peak (if return_peak_stats is True).
        """

        if sleep_scoring_vec is None:
            use_sleep_scoring = False
        else:
            use_sleep_scoring = True
        
        peak_times = []
        if return_peak_stats:
            passed_conditions = []
            zscores_per_peaks_max = []

            # TODO: ver bien porque en el codigo de MATLAB origianl estos 4 arrays estan definidos como cell. VERFICIAR
            zscores_per_peaks_env = []
            zscores_per_peaks_amp = []
            zscores_per_peaks_grad = []
            inds_per_peak = []

        # Replace nans by zeros
        original_data = data.copy()
        data[np.isnan(data)] = 0

        # Detect IIS on NREM sleep
        # TODO: esto lo tengo que acomodar porque la idea es mirar durante vigilia, por lo que no estaria esta flag de NREM
        if use_sleep_scoring:
            sleep_scoring = sleep_scoring_vec.copy()
            data[sleep_scoring != self.NREM] = 0

        # Z-score over the entire session
        #TODO: ver esto del axis porque si entran varias señales esto puede cambiar
        zs_amp_all = stats.zscore(data, axis=1)  

        points_in_block = self.blockSizeSec * self.samplingRate
        n_blocks = len(data[0]) // points_in_block
        ind = 1
        #print(points_in_block,n_blocks)
        
        for i_block in range(1, n_blocks + 1):
            # Use 3 conditions: absolute amplitude above a threshold,
            # gradient above threshold, and envelope of the signal after
            # a bandpass above a threshold

            curr_block = data[0][(i_block - 1) * points_in_block:i_block * points_in_block]
            n_curr_block = len(curr_block)

            # Amplitude
            if self.useAmp or self.useConjAmpGrad or self.useConjAmpEnv:
                zs_amp = zs_amp_all[0][(i_block - 1) * points_in_block:i_block * points_in_block]
                points_passed_thresh_amplitude = np.abs(zs_amp) > self.SDthresholdAmp
                points_passed_thresh_amplitude_low_thresh = np.abs(zs_amp) > self.SDthresholdConjAmp
            else:
                if self.isDisjunction:
                    points_passed_thresh_amplitude = np.zeros(n_curr_block, dtype=bool)
                else:
                    points_passed_thresh_amplitude = np.ones(n_curr_block, dtype=bool)

            # Gradient
            if self.useGrad or self.useConjAmpGrad:
                data_gradient = np.concatenate([[0], np.diff(curr_block)])
                zs_grad = stats.zscore(data_gradient)
                points_passed_thresh_gradient = zs_grad > self.SDthresholdGrad
                points_passed_thresh_gradient_low_thresh = zs_grad > self.SDthresholdConjGrad
            else:
                if self.isDisjunction:
                    points_passed_thresh_gradient = np.zeros(n_curr_block, dtype=bool)
                else:
                    points_passed_thresh_gradient = np.ones(n_curr_block, dtype=bool)
                    
            # Bandpass and envelope
            if self.useEnv or self.useConjAmpEnv:
                # First perform bandpass filtering
                filtered_block = self.bandpass(curr_block, self.samplingRate, self.lowCut, self.highCut) 

                # Find envelope
                env_block = np.abs(signal.hilbert(filtered_block))

                # Find points which pass the threshold as set by the number of SDs compared to the current block
                zs_env = stats.zscore(env_block)
                points_passed_thresh_env = zs_env > self.SDthresholdEnv
                points_passed_thresh_env_low_thresh = zs_env > self.SDthresholdConjEnv
            else:
                if self.isDisjunction:
                    points_passed_thresh_env = np.zeros(n_curr_block, dtype=bool)
                else:
                    points_passed_thresh_env = np.ones(n_curr_block, dtype=bool)

            # Conjunction of amplitude & gradient with lower thresholds
            if self.useConjAmpGrad:
                points_passed_thresh_amp_grad_low_thresh = points_passed_thresh_gradient_low_thresh & points_passed_thresh_amplitude_low_thresh
            else:
                points_passed_thresh_amp_grad_low_thresh = np.zeros(n_curr_block, dtype=bool)

            # Conjunction of amplitude & envelope with lower thresholds
            if self.useConjAmpEnv:
                points_passed_thresh_amp_env_low_thresh = points_passed_thresh_env_low_thresh & points_passed_thresh_amplitude_low_thresh
            else:
                points_passed_thresh_amp_env_low_thresh = np.zeros(n_curr_block, dtype=bool)

            # Disjunction condition
            if self.isDisjunction:
                # If isDisjunction is true - points are detected as threshold if any of the conditions is met
                points_passed_thresh = (
                    points_passed_thresh_env |
                    points_passed_thresh_gradient |
                    points_passed_thresh_amplitude |
                    points_passed_thresh_amp_grad_low_thresh |
                    points_passed_thresh_amp_env_low_thresh
                )
            else:
                # If isDisjunction is false - points are detected as threshold if all of the conditions are met
                points_passed_thresh = (
                    points_passed_thresh_env &
                    points_passed_thresh_gradient &
                    points_passed_thresh_amplitude
                )

            #print(np.sum(points_passed_thresh))

            
            # Find sequences to check whether there is a sequence of points passing the threshold
            # (in paper - 5 ms, due to adapatations of code - 1 ms, i.e. no minimal length), and separates between
            # different spikes (merges points which are close
            # together to one spike)
            if np.sum(points_passed_thresh) > 0:
                if return_peak_stats:
                    points_passed_thresh_numeric = points_passed_thresh.astype(int) # convert to numeric in order to count length of sequences
                    curr_peaks, all_peak_inds = self.find_sequences(curr_block, points_passed_thresh_numeric)
                    n_curr_peaks = len(curr_peaks)
                    curr_passed_conditions = []
                    curr_zscores_per_peaks_max = []
                    curr_inds_per_peak = [None] * n_curr_peaks
                    curr_zscores_per_peaks_env = [None] * n_curr_peaks
                    curr_zscores_per_peaks_amp = [None] * n_curr_peaks
                    curr_zscores_per_peaks_grad = [None] * n_curr_peaks

                    for i_peak in range(n_curr_peaks):
                        #print(i_peak)
                        if self.conditionsArrayTrueIfAny:
                            curr_passed_conditions.append([
                                any(points_passed_thresh_env[all_peak_inds[i_peak]]),
                                any(points_passed_thresh_amplitude[all_peak_inds[i_peak]]),
                                any(points_passed_thresh_gradient[all_peak_inds[i_peak]]),
                                any(points_passed_thresh_amp_grad_low_thresh[all_peak_inds[i_peak]]),
                                any(points_passed_thresh_amp_env_low_thresh[all_peak_inds[i_peak]])
                            ])
                        else:
                            seq_to_find = np.ones(self.minLengthSpike)
                            curr_passed_conditions.append([
                                np.any(np.convolve(seq_to_find, points_passed_thresh_env[all_peak_inds[i_peak]], mode='valid')),
                                np.any(np.convolve(seq_to_find, points_passed_thresh_amplitude[all_peak_inds[i_peak]], mode='valid')),
                                np.any(np.convolve(seq_to_find, points_passed_thresh_gradient[all_peak_inds[i_peak]], mode='valid')),
                                np.any(np.convolve(seq_to_find, points_passed_thresh_amp_grad_low_thresh[all_peak_inds[i_peak]], mode='valid')),
                                np.any(np.convolve(seq_to_find, points_passed_thresh_amp_env_low_thresh[all_peak_inds[i_peak]], mode='valid'))
                            ])

                        curr_zscores_per_peaks_max.append([
                            np.max(zs_env[all_peak_inds[i_peak]]),
                            np.max(zs_grad[all_peak_inds[i_peak]]),
                            np.max(zs_amp[all_peak_inds[i_peak]])
                        ])

                        curr_inds_per_peak[i_peak] = all_peak_inds[i_peak]
                        curr_zscores_per_peaks_env[i_peak] = zs_env[all_peak_inds[i_peak]]
                        curr_zscores_per_peaks_amp[i_peak] = zs_amp[all_peak_inds[i_peak]]
                        curr_zscores_per_peaks_grad[i_peak] = zs_grad[all_peak_inds[i_peak]]

                else:
                    curr_peaks = self.find_sequences(curr_block, points_passed_thresh)

                # Remove spikes with too many NaNs around them
                isnan_at_peak = np.zeros(len(curr_peaks), dtype=bool)
                spike_vicinity = round((self.minDistSpikes / 1000) * self.samplingRate / 2)

                for i_peak in range(len(curr_peaks)):
                    orig_peak_ind = curr_peaks[i_peak] + (i_block - 1) * points_in_block

                    # Handle indices at the beginning or end of the original data
                    start_point = max(1, orig_peak_ind - spike_vicinity)
                    end_point = min(len(original_data[0]), orig_peak_ind + spike_vicinity)

                    data_around_spike = original_data[0][start_point - 1:end_point]
                    
                    # Check for NaNs around the peak
                    isnan_at_peak[i_peak] = np.isnan(original_data[0][orig_peak_ind - 1]) or (
                        np.sum(np.isnan(data_around_spike)) / len(data_around_spike) >= self.percentageOfNansAllowedArounsSpike
                    )

                    # Sometimes the data has zeros instead of NaN - next code is to deal with it
                    if not isnan_at_peak[i_peak]:
                        isnan_at_peak[i_peak] = (
                            np.sum(data_around_spike == 0) / len(data_around_spike) >= self.percentageOfNansAllowedArounsSpike
                        )

                # Remove spikes with too many NaNs
                curr_peaks = np.array(curr_peaks)[~isnan_at_peak]

                # Remove the NaN peaks also from the statistics
                if return_peak_stats:
                    #passed_conditions = np.concatenate([passed_conditions, curr_passed_conditions[~isnan_at_peak, :]])
                    passed_conditions = passed_conditions + [curr_passed_conditions[i] for i in range(len(curr_passed_conditions)) if not isnan_at_peak[i]]
                    #zscores_per_peaks_max = np.concatenate([zscores_per_peaks_max, curr_zscores_per_peaks_max[~isnan_at_peak, :]])
                    zscores_per_peaks_max = zscores_per_peaks_max + [curr_zscores_per_peaks_max[i] for i in range(len(curr_zscores_per_peaks_max)) if not isnan_at_peak[i]] 

                    # TODO: ver bien como se guardan estos datos porque en el codigo de MATLAB original ahcen uso de los cell, aca esta distinto, VERIFICAR

                    inds_per_peak = inds_per_peak + [curr_inds_per_peak[i] for i in range(len(curr_inds_per_peak)) if not isnan_at_peak[i]]
                    zscores_per_peaks_env = zscores_per_peaks_env + [curr_zscores_per_peaks_env[i] for i in range(len(curr_zscores_per_peaks_env)) if not isnan_at_peak[i]]
                    zscores_per_peaks_amp = zscores_per_peaks_amp + [curr_zscores_per_peaks_amp[i] for i in range(len(curr_zscores_per_peaks_amp)) if not isnan_at_peak[i]]
                    zscores_per_peaks_grad = zscores_per_peaks_grad + [curr_zscores_per_peaks_grad[i] for i in range(len(curr_zscores_per_peaks_grad)) if not isnan_at_peak[i]]
            else:
                #curr_peaks = [] 
                curr_peaks = np.array([], dtype=int)

            #print(curr_peaks)
            #print(type(curr_peaks))
            #print(len(curr_peaks))
            peak_times = np.concatenate([peak_times, curr_peaks + (i_block - 1) * points_in_block])

        if return_peak_stats:
            peak_stats = {
                'zscores_per_peaks_max': zscores_per_peaks_max,
                'zscores_per_peaks_env': zscores_per_peaks_env,
                'zscores_per_peaks_amp': zscores_per_peaks_amp,
                'zscores_per_peaks_grad': zscores_per_peaks_grad,
                'inds_per_peak': inds_per_peak,
                'passed_conditions': passed_conditions,
            }

        return peak_times, peak_stats
        # FIN de detect_times


## HELP FUNCTIONS

    # Signal Filtering
    def bandpass(self, timecourse, sampling_rate, low_cut, high_cut, filter_order=None):
        """
        Applies bandpass filtering to the input timecourse.

        Parameters:
        - timecourse (numpy.ndarray): Input timecourse.
        - sampling_rate (float): Sampling rate of the signal.
        - low_cut (float): Lower cutoff frequency.
        - high_cut (float): Upper cutoff frequency.
        - filter_order (int or None): Order of the filter. If None, uses the default filter order.

        Returns:
        - BP (numpy.ndarray): Bandpass-filtered signal.
        """
        if filter_order is None:
            filter_order = self.defaultFilterOrder

        # Handle NaN values
        indices = np.isnan(timecourse)
        if np.sum(indices) > self.nanWarning * len(timecourse):
            print('Warning: Many NaN values in filtered signal')

        timecourse[indices] = 0

        # Butterworth filter design
        if high_cut == np.inf:
            b, a = signal.butter(filter_order, (low_cut / sampling_rate) * 2, 'high')
        else:
            b, a = signal.butter(filter_order, [(low_cut / sampling_rate) * 2, (high_cut / sampling_rate) * 2])

        # Apply bandpass filter
        BP = signal.filtfilt(b, a, timecourse)
        BP[indices] = np.nan

        return BP               
    

    def find_sequences(self, data, data_thresh):
        """
        Finds sequences of ones in the binary data vector.

        Parameters:
        - data (numpy.ndarray): Input data vector.
        - data_thresh (numpy.ndarray): Binary thresholded data vector.

        Returns:
        - seq_peaks (numpy.ndarray): Peak times for each sequence.
        - peak_all_inds (list): List containing indices for each sequence.

        %help function for frequency analysis detection -
            %receives the vector of binary values and finds sequence of ones in length obj.minLengthSpike, which are
            %separated from each other by at least obj.minDistSpikes, returns peak
            %times where each. Remove sequences that are above
            %obj.maxLengthSpike.
            
            %obj.minLengthSpike and minDistSpikes are in ms - translate to number of data
            %points
            numConsSpikes = round(obj.minLengthSpike*obj.samplingRate/1000);
            distSpikePoints = round(obj.minDistSpikes*obj.samplingRate/1000);
            maxLength = round(obj.maxLengthSpike*obj.samplingRate/1000);
        """

        # Translate time-related parameters to data points
        num_cons_spikes = round(self.minLengthSpike * self.samplingRate / 1000) # min length spikes
        dist_spike_points = round(self.minDistSpikes * self.samplingRate / 1000) # minimun distance between spikes
        max_length = round(self.maxLengthSpike * self.samplingRate / 1000) # max length spikes

        # Finding sequences of 1's in length minLengthSpike
        acc_data_block = np.copy(data_thresh)
        thresh_inds = np.where(data_thresh)[0]
        for i_seq_length in range(1, num_cons_spikes):
            acc_data_block = acc_data_block[:-1] + data_thresh[i_seq_length:]

        seq_inds = np.where(acc_data_block >= num_cons_spikes)[0]
        seq_peaks = np.array([], dtype=int)
        peak_all_inds = []

        # Merge peaks which are close together into one spike
        if len(seq_inds) > 0:
            seq_dists = np.diff(seq_inds)
            distinct_seq = np.where(seq_dists > dist_spike_points)[0] + 1
            seq_beginnings = np.concatenate(([seq_inds[0]], seq_inds[distinct_seq]))
            n_seqs = len(seq_beginnings)
            seq_peaks = np.zeros(n_seqs, dtype=int)
            rmvdet = np.array([], dtype=int)

            for i_peak in range(n_seqs):
                lower_limit = seq_beginnings[i_peak]
                if (len(seq_beginnings)-1) > i_peak:
                    upper_limit = seq_inds[distinct_seq[i_peak] - 1] + num_cons_spikes - 1
                else:
                    upper_limit = seq_inds[-1] + num_cons_spikes - 1
                
                curr_inds = np.arange(lower_limit, upper_limit + 1)

                if len(curr_inds) > max_length:
                    rmvdet = np.append(rmvdet,i_peak)

                peak_all_inds.append(curr_inds)
                # The point defining a peak is chosen to be the point with the maximal data value among the points
                # that passed the threshold
                max_point = np.argmax(data[curr_inds])
                seq_peaks[i_peak] = curr_inds[max_point]
            
            seq_peaks = np.delete(seq_peaks, rmvdet)
            peak_all_inds = [x for i, x in enumerate(peak_all_inds) if i not in rmvdet]

        return seq_peaks, peak_all_inds
    

    def plot_spike_waves(self, data, peak_times, block_size_to_plot, peak_stats=None, plot_z_scores=False):
        # plots the peak times
        # receives the data, the timing of the spikes, and how many
        # blocks (spikes) to plot in each subplot (recommended is 5,
        # default is 1)
        # peak_stats - the output of detectTimes with statistics on
        # peaks. If provided, for each peak the title will include
        # information on which detection condition it passed and the
        # maximal zscore for each parameter (envelope of HP, amplitude,
        # gradient)

        if block_size_to_plot is None:
            block_size_to_plot = self.blockSizePlot

        plot_conditions_data = peak_stats is not None

        if not plot_conditions_data or plot_z_scores is False:
            plot_z_scores = False

        n_peaks = len(peak_times)
        ind_block = 1

        block_size_data = self.blockSizeSec * self.samplingRate

        for i_peak in range(10):#range(n_peaks): 
        #for i_peak in np.random.randint(0, len(peak_stats['zscores_per_peaks_max'])-1, size=15):
            if plot_z_scores:
                plt.subplot(4, block_size_to_plot, ind_block)
            else:
                plt.subplot(1, block_size_to_plot, ind_block)

            # plot the block in which the peak was detected
            block_num = int(np.floor(peak_times[i_peak] / block_size_data) + 1)
            min_point = (block_num - 1) * block_size_data + 1
            max_point = block_num * block_size_data
            peak_point = peak_times[i_peak] % block_size_data

            curr_data = data[0][min_point:max_point]
            plt.plot(curr_data)
            #plt.hold(True)
            
            if not plot_conditions_data:
                plt.plot(peak_point, min(curr_data) * 2, marker='*', color='k')
            else:
                curr_inds = peak_stats['inds_per_peak'][i_peak]
                aux_min_curr_data = [min(curr_data) * 2] * len(curr_inds)
                #plt.plot(curr_inds, min(curr_data) * 2, marker='*', color='k')
                plt.plot(curr_inds, aux_min_curr_data, marker='*', color='k')

            if plot_conditions_data:
                title_str = f"spike #{i_peak} passed conditions {peak_stats['passed_conditions'][i_peak]} " 
                #f"spike #{i_peak} passed conditions {peak_stats['passed_conditions'][i_peak, :]} max zscores: HP(red) = {peak_stats['zscores_per_peaks_max'][i_peak, 0]} Amp(blue) = {peak_stats['zscores_per_peaks_max'][i_peak, 2]} Grad(green) = {peak_stats['zscores_per_peaks_max'][i_peak, 1]}"
                plt.title(title_str)

            #plt.hold(False)

            if plot_z_scores:
                curr_data = data[0][min_point:max_point]

                zs_amp = stats.zscore(curr_data) 
                #zs_amp = (curr_data - np.mean(curr_data)) / np.std(curr_data)

                data_gradient = np.concatenate([[0], np.diff(curr_data)])
                zs_grad = stats.zscore(data_gradient)
                #zs_grad = (data_gradient - np.mean(data_gradient)) / np.std(data_gradient)

                filtered_block = self.bandpass(curr_data, self.samplingRate, self.lowCut, self.highCut)
                # find envelope
                # env_block = np.abs(np.fft.hilbert(filtered_block))
                env_block = np.abs(signal.hilbert(filtered_block))
                #zs_hp = (env_block - np.mean(env_block)) / np.std(env_block)
                zs_hp = stats.zscore(env_block) 

                plt.subplot(4, block_size_to_plot, ind_block + block_size_to_plot)
                plt.plot(zs_hp, color='r')
                plt.title('HP zscore')
                #plt.hold(True)
                plt.axhline(y=self.SDthresholdEnv, color='k')
                plt.axhline(y=self.SDthresholdConjEnv, color='k')
                #plt.hold(False)

                plt.subplot(4, block_size_to_plot, ind_block + block_size_to_plot * 2)
                plt.plot(zs_amp, color='b')
                plt.title('Amplitude zscore')
                #plt.hold(True)
                plt.axhline(y=self.SDthresholdAmp, color='k')
                plt.axhline(y=self.SDthresholdConjAmp, color='k')
                #plt.hold(False)

                plt.subplot(4, block_size_to_plot, ind_block + block_size_to_plot * 3)
                plt.plot(zs_grad, color='g')
                plt.title('Gradient zscore')
                #plt.hold(True)
                plt.axhline(y=self.SDthresholdGrad, color='k')
                plt.axhline(y=self.SDthresholdConjGrad, color='k')
                #plt.hold(False)

            ind_block += 1
            if ind_block > block_size_to_plot or i_peak == n_peaks - 1:
                ind_block = 1
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                plt.show()
                #plt.pause(0.1)
                plt.waitforbuttonpress()
                plt.close()