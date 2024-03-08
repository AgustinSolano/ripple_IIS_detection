import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class RippleDetectorClass:
    def __init__(self):

        self.samplingRate = 1000
        # TODO: ver esto del prefijo
        self.dataFilePrefix = 'CSC'

        # all the parameters are from Staresina et al (and Zhang et al is identical)
        self.minFreq = 80
        self.maxFreq = 100

        self.RMSWindowDuration = 20  # ms
        self.rippleThreshPercentile = 99
        self.minDurationAboveThresh = 38  # ms
        self.minNumOfExtreme = 3  # min number of peaks / troughs in a ripple event
        self.NPointsToAverage = 3
        self.minPercNaNAllowed = 0.1
        self.minDistBetweenRipples = 20  # ms

        # IIS removal constants
        self.windowAroundIIS = 500  # ms

        # STIMULATION removal constants TODO: eso no lo uso
        self.windowAroundSTIM = 200  # ms
        
        # sleep scoring parameters TODO: eso no lo uso
        self.scoringEpochDuration = 0.001  # How many seconds represented by one individual value in the scoring vector [scalar].
        self.sleepEpochs = [1]  # all the values in the scoring vector which represent sleep stages for which we want to perform the analysis (like NREM/REM/transitions) [1D vector].
        
        # filtering constants
        self.defaultFilterOrder = 3
        self.nanWarning = 0.01
        
        # plot params
        self.secondBefAfter = 0.5  # seconds
        self.subplotSizeY = 4
        self.subplotSizeX = 3
        self.nInPlotMicro = 4

        # TODO: ver como afecto esto, no va en mi caso
        self.spikeMultiUnits = True
        
        # control params for spike rate around ripples
        self.minDistControl = 300  # ms
        self.maxDistControl = 1300  # ms

        self.firingRateWinSize = 10  # ms
        self.windowSpikeRateAroundRip = 500  # ms
        self.windowForSignificance = 100  # ms
        
        # params for spike rate around stimulations
        self.windowSpikeRateAroundStim = 500
        self.windowSpikeRateForComparison = 500  # ms - for comparing between stim and control
        self.controlDistForStim = 1000  # ms

        self.avgRippleBeforeAfter = 1  # second
        self.freqoiForAvgSpec = np.arange(0, 10.5, 0.5)
        self.freqRangeForAvgSpec = np.arange(1, 251)
        self.timeBeforeAfterEventRipSpec = 1  # second
        self.timeForBaselineRip = 1  # second
        self.minNCycles = 5
        self.minWinSizeSpec = 100  # ms
        self.minNripples = 5

        # micro constants
        self.ripplesDistMicrChanForMerge = 15  # ms
        self.minNripplesForAnalysis_perChannel = 20  # dropping channels with low ripple rate
        
        # plotting constants
        self.nBinsHist = 10
        self.maxLinesInFigureRipSpike = 4
        self.maxColumnsInFigureDataMicro = 4
        
        self.winFromLastSpike = 1000  # ms
        self.shortTimeRangeAfterStim = 3  # seconds
        self.midTimeRangeAfterStim = 60  # seconds
        self.stimulusDuration = 50  # ms
        self.minSpikeRateToIncludeUnit = 1
        
        self.freqRangeForShowingSpindles = list(range(5, 31))
        self.xaxisForRipSp = list(range(-500, 501))
        self.specStartPointRipSp = 500
        self.freqRangeSpRip = list(range(50, 151))
        self.nBinsPolar = 18

    # ripple detection function
    def detect_ripple(self, data, sleep_scoring=None, IIS_times=None, stim_times=None):
        
        # Convert sizes according to the sampling rate
        RMS_window_duration = int(self.RMSWindowDuration * self.samplingRate / 1000)
        min_duration_above_thresh = int(self.minDurationAboveThresh * self.samplingRate / 1000)
        min_dist_between_ripples = int(self.minDistBetweenRipples * self.samplingRate / 1000)

        if sleep_scoring is None:
            sleep_scoring = []

        remove_IIS = True if IIS_times is not None else False
        remove_STIM_artifacts = True if stim_times is not None else False

        # Filter data to required range
        filtered_data = self.bandpass(data[0], self.samplingRate, self.minFreq, self.maxFreq)

        # ESTO NO VA PORQUE LO MIO ES EN VIGILIA
        # If sleep scoring is provided, leave only the segments with the desired sleep stage
        '''if sleep_scoring:
            seg_length = self.scoringEpochDuration * self.sampling_rate
            is_sleep = np.zeros(len(sleep_scoring) * seg_length)
            for i_epoch in range(len(sleep_scoring)):
                if sleep_scoring[i_epoch] in self.sleepEpochs:
                    is_sleep[i_epoch * seg_length:(i_epoch + 1) * seg_length] = 1
            if len(is_sleep) > len(filtered_data):
                is_sleep = is_sleep[:len(filtered_data)]
            elif len(is_sleep) < len(filtered_data):
                filtered_data = filtered_data[:len(is_sleep)]
                data = data[:len(is_sleep)]
            filtered_data[~is_sleep.astype(bool)] = np.nan
            data[~is_sleep.astype(bool)] = np.nan'''

        # ESTO NO VA PORQUE NO TENEMOS ESTIMULACION
        # Remove window around stimulation artifacts
        '''if remove_STIM_artifacts:
            win_around_STIM = int(self.windowAroundSTIM * self.samplingRate / 1000)
            for time in stim_times:
                points_before = min(time, win_around_STIM)
                points_after = min(len(filtered_data) - time, win_around_STIM)
                filtered_data[time - points_before:time + points_after] = np.nan
                data[time - points_before:time + points_after] = np.nan'''

        # Remove window around IIS artifacts
        if remove_IIS:
            win_around_IIS = int(self.windowAroundIIS * self.samplingRate / 1000)
            for time in IIS_times:
                points_before = min(time, win_around_IIS)
                points_after = min(len(filtered_data) - time, win_around_IIS)
                filtered_data[int(time - points_before):int(time + points_after)] = np.nan
                data[int(time - points_before):int(time + points_after)] = np.nan

        # Calculate the root mean squared signal for windows of length RMSWindowDuration
        rms_signal = np.zeros(len(filtered_data) - RMS_window_duration + 1)
        for i_point in range(len(filtered_data) - RMS_window_duration + 1):
            rms_signal[i_point] = np.sqrt(np.mean(filtered_data[i_point:i_point + RMS_window_duration - 1] ** 2))

        # Calculate the threshold as the rippleThreshPercentile percentile of the rms signal
        ripple_thresh = np.nanpercentile(rms_signal, self.rippleThreshPercentile)

        # Find windows that pass the threshold
        did_pass_thresh = rms_signal >= ripple_thresh

        # Find segments that pass the threshold for a duration longer than minDurationAboveThresh milliseconds
        ripple_segs = []
        ind = 0
        indRipple = 0
        while ind <= len(did_pass_thresh) - min_duration_above_thresh:
            if all(did_pass_thresh[ind:ind+min_duration_above_thresh]):
                ripple_segs.append([0, 0])
                ripple_segs[indRipple][0] = ind
                endSeg = ind + np.where(did_pass_thresh[ind:] == 0)[0][0] - 1
                ripple_segs[indRipple][1] = endSeg + RMS_window_duration
                indRipple += 1
                ind = endSeg + 1
            else:
                ind += 1

        # Merge ripples that are close
        ripple_segs_merged = []
        if ripple_segs:
            ripple_segs_arr = np.array(ripple_segs)
            # Diferencia entre el final de un ripple y el inicio del siguiente
            ripple_diffs_small = (ripple_segs_arr[1:, 0] - ripple_segs_arr[:-1, 1]) < min_dist_between_ripples
            #ripple_diffs_small = np.diff(ripple_segs, axis=0) < min_dist_between_ripples

            ind_old = 0
            ind_new = -1
            while ind_old < (len(ripple_segs)-1):
                ind_new += 1
                if not ripple_diffs_small[ind_old]:
                    ripple_segs_merged.append(ripple_segs[ind_old])
                    ind_old += 1
                else:
                    next_merge = np.argmax(~ripple_diffs_small[ind_old + 1:]) + ind_old + 1
                    if next_merge == 0:
                        next_merge = len(ripple_segs)
                    ripple_segs_merged.append([ripple_segs[ind_old][0], ripple_segs[next_merge][1]])
                    ind_old = next_merge + 1
                    ind_new += 1
            if not np.any(ripple_diffs_small) or next_merge < len(ripple_segs):
                ripple_segs_merged.append(ripple_segs[-1])

            ripple_segs = ripple_segs_merged

            # Go over ripples and leave only those with minNumOfExtreme extremes in the raw data
            ripple_times = []
            ripple_start_end = []
            is_ripple = False
            for ripple_seg in ripple_segs:
                curr_ripple = data[0][ripple_seg[0]:ripple_seg[1] + 1]
                if np.isnan(curr_ripple).sum() / len(curr_ripple) >= self.minPercNaNAllowed:
                    continue
                curr_ripple = np.convolve(curr_ripple, np.ones(self.NPointsToAverage) / self.NPointsToAverage, mode='valid')
                #local_max = np.where((curr_ripple[1:-1] > curr_ripple[:-2]) & (curr_ripple[1:-1] > curr_ripple[2:]))[0] + 1
                local_max = signal.argrelmax(curr_ripple)[0]
                if len(local_max) >= self.minNumOfExtreme:
                    is_ripple = True
                else:
                    #local_min = np.where((curr_ripple[1:-1] < curr_ripple[:-2]) & (curr_ripple[1:-1] < curr_ripple[2:]))[0] + 1
                    local_min = signal.argrelmin(curr_ripple)[0]
                    if len(local_min) >= self.minNumOfExtreme:
                        is_ripple = True

                if is_ripple:
                    # find the location of the largest peak
                    abs_max_ind = ripple_seg[0] + np.argmax(curr_ripple)
                    ripple_times.append(abs_max_ind)
                    ripple_start_end.append(ripple_seg)

                is_ripple = False
        else:
            ripple_times = []
            ripple_start_end = []

        return ripple_times, ripple_start_end
    
  
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
            b, a = signal.butter(filter_order, [(low_cut / sampling_rate) * 2, (high_cut / sampling_rate) * 2], 'bandpass')

        # Apply bandpass filter
        BP = signal.filtfilt(b, a, timecourse)
        BP[indices] = np.nan

        return BP
    

    # Plot ripples
    def plot_ripples(self, data, ripple_times, folder_to_save=None):
        
        if folder_to_save is None:
            to_save = False
        else:
            to_save = True

        second_bef_after = int(self.secondBefAfter * self.samplingRate)

        # Filter data to required range
        filtered_data = self.bandpass(data[0], self.samplingRate, self.minFreq, self.maxFreq)

        n_ripples = len(ripple_times)
        n_in_plot = self.subplotSizeX * self.subplotSizeY
        n_plots = (n_ripples - 1) // n_in_plot + 1

        ind_ripple = 0
        fig_ind = 1

        #for i_plot in range(n_plots):
        for i_plot in np.random.randint(0, n_plots-1, size=15):

            f, axs = plt.subplots(self.subplotSizeY, self.subplotSizeX, figsize=(12, 8))

            for ax in axs.flatten():
                if ind_ripple < n_ripples:
                    min_ind = max(ripple_times[ind_ripple] - second_bef_after, 0)
                    max_ind = min(ripple_times[ind_ripple] + second_bef_after, len(data[0]))

                    ax.plot(data[0][min_ind:max_ind], label='Original Data')
                    ax.plot(filtered_data[min_ind:max_ind], label='Filtered Data')
                    
                    '''ax.plot(np.arange(-second_bef_after, second_bef_after) / self.samplingRate,
                            data[0][min_ind:max_ind], label='Original Data')
                    ax.plot(np.arange(-second_bef_after, second_bef_after) / self.samplingRate,
                            filtered_data[min_ind:max_ind], label='Filtered Data')'''
                    ax.set_title(f'Ripple time = {ripple_times[ind_ripple] / self.samplingRate / 60} mins')

                    ind_ripple += 1

            plt.tight_layout()

            #if to_save:
                #plt.savefig(os.path.join(folder_to_save, f'all_ripples_{fig_ind}.jpg'))
                #plt.close(f)
            #else:
            plt.show()
            plt.waitforbuttonpress()
            plt.close()

            fig_ind += 1

