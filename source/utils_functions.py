import numpy as np
from scipy import signal, stats
from skimage.measure import label, regionprops

def detect_IEDs_Sch23(signal_in, srate, t_epoch):
    """
    Detects Interictal Epileptiform Discharges (IEDs) in a given signal.
    Parameters:
    signal_in (numpy.ndarray): The input signal from which IEDs are to be detected.
    srate (float): The sampling rate of the input signal.
    t_epoch (tuple): A tuple containing the start and end times of the epoch relative to the detected event.
    Returns:
    numpy.ndarray: A matrix containing information about detected IEDs. The matrix has four columns:
        - Column 0: Location of discharge maximum.
        - Column 1: Trial onset (location of maximum - start time of epoch * sampling rate).
        - Column 2: Trial offset (location of maximum + end time of epoch * sampling rate).
        - Column 3: Offset relative to maximum location (center of epoch * sampling rate).

    INFO: based on Schalkwijk et al. 2023
    """
    
    # 1. Get envelope from raw signal
    amplit_envelope = np.abs(signal.hilbert(signal_in))
    amplit_envelope = stats.zscore(amplit_envelope, axis=1) 

    # 2. Band-pass filtering 
    filter_order = 10
    low_cut = 25 * 0.8 # low cut-off frequency for high-pass filter (20% less to compensate for attenuation)
    b_hp, a_hp = signal.butter(filter_order, (low_cut / srate) * 2, 'high', analog=False)
    high_cut = 80 * 1.2 # high cut-off frequency for low-pass filter (20% less to compensate for attenuation)
    b_lp, a_lp = signal.butter(filter_order, (high_cut / srate) * 2, 'low', analog=False) 

    # apply filtering and get envelope
    filt_signal_in = signal.filtfilt(b_hp, a_hp, signal_in) # HP
    filt_signal_in = signal.filtfilt(b_lp, a_lp, filt_signal_in) #LP

    filt_signal_in_envelope = np.abs(signal.hilbert(filt_signal_in))

    # 3. Normalize
    filt_signal_in_envelope = stats.zscore(filt_signal_in_envelope, axis=1)

    # 4. Detect events above threshold and time criterion
    cutoff  = 2
    min_t   = 0.02 * srate
    max_t   = 0.1 * srate

    # 5. only find consecutive segments 
    aboveThreshold = (filt_signal_in_envelope > cutoff) # - Samples of threshhold
    aboveThreshold[amplit_envelope < cutoff] = 0  # - Remove if raw sd is < cutoff

    # Identify continuous one and apply time criterion
    spanLocs = label(aboveThreshold) # Label connected regions
    regions = regionprops(spanLocs) # Measure region properties (area)
    spanLength = np.array([region.area for region in regions]) # Extract the length (area) of each region
    goodSpans = np.where((spanLength > min_t) & (spanLength < max_t))[0] # Find indices of spans with lengths within the specified range

    # 6. Create trl matrix 
    # [Note:] trl matrix is a 4-column matrix containing the following information for each trial: 
    #   - [:,0]  Location of discharge maximum
    #   - [:,1]  Trial onset: loc_maximum - trange*srate
    #   - [:,2]  Trial offset: loc_maximum - trange*srate
    #   - [:,3]  Offset relative to maximum location (center): trange*srate

    trl = np.zeros((len(goodSpans),4))

    for g in range(len(goodSpans)):

        currentepochidx = np.where(spanLocs == (goodSpans[g]+1))[1]
        currentepochvals = filt_signal_in_envelope[0,currentepochidx]

        # Find the max in this range
        maxloc = np.argmax(currentepochvals)
        maxloc = maxloc + currentepochidx[0]

        # Create the trl
        trl[g,:] = [maxloc, maxloc-np.abs(t_epoch[0])*srate, maxloc+np.abs(t_epoch[1])*srate, np.abs(t_epoch[0])*srate]

    # 7. Eliminate events that are within 1 sec -------------------------------
    tr_el = np.zeros(trl.shape[0])

    for tr in range(1, trl.shape[0]):
        if (trl[tr,0] - trl[tr-1,0]) <= srate:
            tr_el[tr] = 1
        else:
            pass

    #print(tr_el)

    trl = np.delete(trl, np.where(tr_el == 1), axis=0)

    return trl


def detect_IEDs_Norm19(signal_in, srate, t_epoch=[-2.5, 2.5]):

    IED_band = [25, 60] # [Hz] IED band
    thresh_level = [2, 4] # [z-score] threshold level for IED detection
    ripple_band = [70, 180] # [Hz] ripple band
    max_t  = 0.1 * srate # [samples] maximum duration of IED event

    # IED rejection (Gelinas et al. 2016 Nat. Med.):
    # FIltering in the IED Band
    filter_order = 10
    low_cut_IED = IED_band[0] * 0.8 # [Hz] low cut-off frequency for high-pass filter (20% less to compensate for attenuation)
    b_hp_IED, a_hp_IED = signal.butter(filter_order, (low_cut_IED / srate) * 2, btype = 'highpass', analog=False) 
    high_cut_IED = IED_band[1] * 1.2 # [Hz] high cut-off frequency for low-pass filter (20% more to compensate for attenuation)
    b_lp_IED, a_lp_IED = signal.butter(filter_order, (high_cut_IED / srate) * 2, btype = 'lowpass', analog=False)

    # apply filtering and get envelope
    filt_signal_in_IEDband = signal.filtfilt(b_hp_IED, a_hp_IED, signal_in) # HP
    filt_signal_in_IEDband = signal.filtfilt(b_lp_IED, a_lp_IED, filt_signal_in_IEDband) #LP

    # Hilbert transform
    filt_signal_in_IEDband_envelope = np.abs(signal.hilbert(filt_signal_in_IEDband))
    # Squaring:
    filt_signal_in_IEDband_envelope = filt_signal_in_IEDband_envelope**2

    # Smoothing using a lowpass filter with FIR kaiserwin 
    LPcutoff = np.round(np.mean(ripple_band)/np.pi) # in Hz (see Stark et al. 2014 for a similar method)
        #Note: the cutoff frequency is 40 Hz when ripple-band frequency window is 70-180 Hz 
    transition_width = 10 # [Hz]
    b_lp_kaiser = design_kaiser_filter(LPcutoff, transition_width, srate)

    filt_signal_in_IEDband_envelope = signal.filtfilt(b_lp_kaiser, [1], filt_signal_in_IEDband_envelope)

    # Zscore normalization
    filt_signal_in_IEDband_envelope = (filt_signal_in_IEDband_envelope - np.nanmean(filt_signal_in_IEDband_envelope))/np.nanstd(filt_signal_in_IEDband_envelope)

    print('- Identifying IEDs events [-]')

    IED_locs, IED_properties = signal.find_peaks(filt_signal_in_IEDband_envelope[0], height=thresh_level[1])
    
    trl = np.zeros((len(IED_locs),4))

    for i in range(len(IED_locs)):
        tmp_start = np.max((int(IED_locs[i]-max_t/2),0))
        tmp_stop = np.min((int(IED_locs[i]+max_t/2),len(signal_in[0])))
        currentepochvals = signal_in[0,tmp_start:tmp_stop]
        maxloc = np.argmax(currentepochvals)
        maxloc = tmp_start + maxloc
        
        trl[i,:] = [maxloc, maxloc-np.abs(t_epoch[0])*srate, maxloc+np.abs(t_epoch[1])*srate, np.abs(t_epoch[0])*srate]

    # Se detectan valores repetidos del maximo y se eliminan porque refiere al mismo IED
    _, unique_indices = np.unique(trl[:, 0], return_index=True)
    trl = trl[unique_indices,:]

    return trl


def ripple_char(signal_in, locs, maxsamples):
    """
    Analyze ripple characteristics in a given signal.
    Parameters:
    signal_in (numpy.ndarray): The input signal array.
    locs (list or numpy.ndarray): Indices of candidate ripple peaks.
    maxsamples (int): The maximum number of samples to consider around each candidate peak.
    Returns:
    tuple: A tuple containing three numpy arrays:
        - ripplepeaks (numpy.ndarray): Number of peaks in each segment.
        - rippletrough (numpy.ndarray): Indices of the trough closest to each ripple peak.
        - sharptransient (numpy.ndarray): Sharpest transient for later spike removal.
    """

    # - Pre-define output variables -
    # -------------------------------------------------------------------------
    # Pre-define output variables
    ripplepeaks = np.full(len(locs), np.nan)
    rippletrough = np.full(len(locs), np.nan)
    sharptransient = np.full(len(locs), np.nan)

    # - Go through ripple candidates -
    for idx in range(len(locs)):
        
        # 1. Determine window -------------------------------------------------
        s_start = np.round(locs[idx] - maxsamples/2).astype(int) # start sample
        s_end = np.round(locs[idx] + maxsamples/2).astype(int) # stop sample
        datasegm_trial  = signal_in[0,s_start:s_end]
        #datasegm_time = times[s_start:s_end]
        
        # 2. Find all troughs close to candidate ripple peak ------------------
        idxloc, _ = signal.find_peaks((-1)*datasegm_trial)
        pks2 = datasegm_trial[idxloc]
        idxloc = idxloc + s_start # offset to the real indexs
        
        # 3. Save ripple characteristics --------------------------------------
        if len(idxloc) == 0 or len(idxloc) == 1:
            ripplepeaks[idx] = np.nan
            rippletrough[idx] = np.nan
            sharptransient[idx] = np.nan
        else:
            ripplepeaks[idx] = len(idxloc) # - Number of peaks in segment
            rippletrough[idx] = idxloc[np.argmin(np.abs(idxloc-locs[idx]))] # - Datapoint of trough closest to ripple peak idx 

            # 4. Flag sharpest transient ------------------------------------------
            pksz = stats.zscore(pks2)
            pksdiff = pksz[1:]-pksz[:-1]
            sharptransient[idx] = np.max(np.abs(pksdiff))                    # - Mark sharpest transient for later spike removal


    return ripplepeaks, rippletrough, sharptransient

def num_discharges_in_ripple_trial(trial_start, trial_end, discharge_pks):
    """
    Calculate the number of discharges within each ripple trial.
    Parameters:
    trial_start (array-like): An array of start times for each trial.
    trial_end (array-like): An array of end times for each trial.
    discharge_pks (array-like): An array of discharge peak times.
    Returns:
    numpy.ndarray: An array containing the number of discharges for each trial.
    """
    
    nodischarges = np.zeros(len(trial_start))

    for tr_idx in range(len(trial_start)):

        currentepoch_start = trial_start[tr_idx]
        currentepoch_end = trial_end[tr_idx]
        nodischarges[tr_idx] = np.sum((discharge_pks >= currentepoch_start) & (discharge_pks <= currentepoch_end))

    return nodischarges


def detect_ieeg_ripples_Sch23(signal_in, srate, IED_events, t_range = 2.5):
    """
    Detects ripples in an intracranial EEG (iEEG) signal.
    Parameters:
    signal_in (numpy.ndarray): The input iEEG signal.
    srate (int): The sampling rate of the input signal.
    IED_events (numpy.ndarray): Array containing information about detected inter-ictal discharges.
    t_range (float, optional): Time range for ripple detection. Default is 2.5 seconds.
    Returns:
    numpy.ndarray: A matrix containing information about detected ripples.

    INFO: based on Schalkwijk et al. 2023
    """

    # 1. Band-pass filtering 
    print('- Filtering and Hilbert transform [-]')
    filter_order = 10
    low_cut = 80 * 0.8 # [Hz] low cut-off frequency for high-pass filter (20% less to compensate for attenuation)
    b_hp, a_hp = signal.butter(filter_order, (low_cut / srate) * 2, btype = 'highpass', analog=False) 
    high_cut = 120 * 1.2 # [Hz] high cut-off frequency for low-pass filter (20% more to compensate for attenuation)
    b_lp, a_lp = signal.butter(filter_order, (high_cut / srate) * 2, btype = 'lowpass', analog=False)

    # apply filtering and get envelope
    filt_signal_in = signal.filtfilt(b_hp, a_hp, signal_in) # HP
    filt_signal_in = signal.filtfilt(b_lp, a_lp, filt_signal_in) #LP

    # Hilbert transform
    filt_signal_in_envelope = np.abs(signal.hilbert(filt_signal_in))

    b_lp_h, a_lp_h = signal.butter(filter_order, (10*1.2 / srate) * 2, btype = 'lowpass', analog=False) #10 Hz LP on hilbert amplitude
    filt_signal_in_envelope = signal.filtfilt(b_lp_h, a_lp_h, filt_signal_in_envelope) 

    # Normalize with zscore
    filt_signal_in_envelope = stats.zscore(filt_signal_in_envelope, axis=1)


    # - Find candidate ripple events ------------------------------------------
    print('- Looking candidate ripple events [-]')
    # [Notes:]  - Minimum duration = 25 ms
    #           - Maximum duration = 200 ms  
    #           - Minimum peak distance = 500 ms (avoiding overlap)
    # -------------------------------------------------------------------------
    minsamples = np.round((25/1000) * srate) # Minimum duration 25 ms (sampling points) %note: orig. 20ms
    maxsamples = np.round((200/1000) * srate) # Maximum duration 200 ms (sampling points) 
    MPD        = np.round((500/1000) * srate) # Minimum peak distance 500 ms (sampling points)

    # - Determine minimum & maximum peak height & distance - - - - - - - - - - 
    # [Notes:]  - pks = peak values
    #           - locs = locations 
    #           - width = distance between peaks 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Find peaks in the filtered signal envelope
    peaks, properties = signal.find_peaks(filt_signal_in_envelope[0], height=2, width=(minsamples, maxsamples), distance=MPD)
    pks = properties['peak_heights']
    locs = peaks
    width = properties['widths']

    # - Now go through the signal and determine ripple characteristics --------
    print('- Determine ripple characteristics [-]')
    ripplepeaks, rippletrough, sharptransient = ripple_char(signal_in, locs, maxsamples)

    # - Render peak width into secs -------------------------------------------
    width = width / srate

    # - Render ripple peaks per epoch into Hz ---------------------------------
    ripplepeaks = ripplepeaks * (srate/maxsamples) 

    # - Remove NaN values -----------------------------------------------------
    # TODO

    # - Create a basic trl ----------------------------------------------------
    # [Note:] Defines trial centered around ripple trough as follows:
    #   - [:,0]  Trial onset: ripple trough - settings.trange*data.fsample [samples]
    #   - [:,1]  Trial offset: ripple trough + settings.trange*data.fsample [samples]
    #   - [:,2]  Trigger offset: set at -settings.trange*data.fsample [samples]
    #   - [:,3]  Ripple peak amplitude (z-score)
    #   - [:,4]  Ripple frequency [Hz]
    #   - [:,5]  Indication of sharpest transient in trial
    #   - [:,6]  Ripple duration in seconds
    #   - [:,7]  Cross referencing with discharge detection (optional) : amount of detected discharges in trial
    # -------------------------------------------------------------------------  

    # Si hay informacion sobre las descargas
    if len(IED_events)!=0:
        discharge_pks = IED_events[:,0]
        #num_discharges = np.zeros(len(rippletrough))
        num_discharges = num_discharges_in_ripple_trial(rippletrough-t_range*srate, rippletrough+t_range*srate, discharge_pks)
        trl = np.column_stack((rippletrough-t_range*srate, rippletrough+t_range*srate, np.full(len(rippletrough), -2.5*srate), pks, ripplepeaks, sharptransient, width, num_discharges))

    # Si NO hay informacion sobre las descargas
    else:
        trl = np.column_stack((rippletrough-t_range*srate, rippletrough+t_range*srate, np.full(len(rippletrough), -2.5*srate), pks, ripplepeaks, sharptransient, width, np.zeros(len(rippletrough))))


    # - Remove any potential trials that: -------------------------------------
    #       - Start prior to the recording onset (i.e., ripple detected at the very start)
    #       - Exceed the recording (i.e., ripple detected at the very end
    # -------------------------------------------------------------------------
    trl = trl[trl[:,0]>0,:] # - Remove trial prior to recording onset
    trl = trl[trl[:,1]<len(signal_in[0]),:] # - Remove trials that occur past recording end


    # 2. Cleaning up the trl matrix
    print('- Cleaning up the trl matrix according to criteria [-]')

    amp_th = [2,4] # amplitude threshold
    minfreq = 80; # minimum frequency
    sharptransient_th = 2 # sharp transient threshold
    #overlap = 0 # ??
    trange = 2.5
    min_num_ripples = 10

    # Compute logical AND for multiple conditions
    tIndex = np.logical_and.reduce((
        trl[:,3] >= amp_th[0], trl[:,3] <= amp_th[1], # - Amplitude criteria
        trl[:,4] > minfreq, # - Frequency criteria
        trl[:,5] < sharptransient_th, # - Sharp transient criteria
        trl[:,6] > (np.median(trl[:,6]) - np.std(trl[:,6])), # - Duration criteria
        trl[:,7] == 0 # - IED criteria  
    ))

    # Se concatena como una nueva columna cuales son los putative-ripples que pasan el proceso de limpienza (=1) y los que no (=0)
    trl = np.column_stack((trl, tIndex))

    return trl


def compute_z_normalized_erp(epochs, baseline=(-2.5, -2.0)):
    """
    Compute z-normalized ERP relative to a baseline window in MNE Epochs object.

    Parameters:
    - epochs: mne.Epochs object
    - baseline: tuple, defines the time range for baseline normalization (in seconds)
    
    Returns:
    - z_normalized_erp: numpy array of the z-normalized ERP
    """
    # Compute the mean ERP across all epochs
    evoked = epochs.average()

    # Convert Evoked object to NumPy array (channels x time points)
    erp = evoked.data

    # Get the sampling frequency and time points
    times = evoked.times
    
    # Define the baseline time indices
    baseline_idx = np.where((times >= baseline[0]) & (times <= baseline[1]))[0]

    # Compute the baseline mean and std (for each channel)
    baseline_mean = erp[:, baseline_idx].mean(axis=1, keepdims=True)
    baseline_std = erp[:, baseline_idx].std(axis=1, keepdims=True)

    # Z-normalize the ERP relative to the baseline
    z_normalized_erp = (erp - baseline_mean) / baseline_std

    return z_normalized_erp, evoked.times


def design_kaiser_filter(cutoff_freq, transition_width, sampling_rate, att_rejectband_db=60):
    """
    Design a Kaiser filter based on given specifications.
    
    Parameters:
    -----------
    cutoff_freq : float or tuple
        Cutoff frequency in Hz. For bandpass/bandstop, provide a tuple of (low, high)
    transition_width : float
        Width of transition region in Hz
    sampling_rate : float
        Sampling rate of the signal in Hz
    att_rejectband_db : float, optional
        Attenuation in the stop band in dB (default: 60)
        
    Returns:
    --------
    b : ndarray
        FIR filter coefficients
    """
    
    # Normalize frequencies to Nyquist frequency
    nyq_rate = sampling_rate / 2.0
    
    if isinstance(cutoff_freq, tuple):
        # For bandpass/bandstop
        edges = [freq/nyq_rate for freq in cutoff_freq]
    else:
        # For lowpass/highpass
        edges = cutoff_freq/nyq_rate
    
    # Calculate filter length and beta parameter
    width = transition_width/nyq_rate
    N, beta = signal.kaiserord(att_rejectband_db, width)
    
    # Make sure N is odd for Type I linear phase FIR filter
    if N % 2 == 0:
        N += 1
    
    # Design the filter
    if isinstance(cutoff_freq, tuple):
        # Bandpass filter
        b = signal.firwin(N, edges, window=('kaiser', beta), pass_zero=False)
    else:
        # Lowpass filter
        b = signal.firwin(N, edges, window=('kaiser', beta))
    
    return b

def detect_ieeg_ripples_Norm19(signal_in, srate, IED_events, t_range = 2.5):
    
    """
    Detects ripples in intracranial EEG (iEEG) signals based on a specified algorithm.
    Parameters:
    signal_in (numpy.ndarray): The input iEEG signal.
    srate (float): The sampling rate of the iEEG signal.
    IED_events (numpy.ndarray): Array of interictal epileptiform discharge (IED) events.
    t_range (float, optional): Time range for trial windowing, default is 2.5 seconds.
    Returns:
    numpy.ndarray: A trial matrix containing detected ripple events and their characteristics. The columns are:
        - Trial onset: ripple trough - t_range * srate [samples]
        - Trial offset: ripple trough + t_range * srate [samples]
        - Trigger offset: set at -t_range * srate [samples]
        - Ripple peak amplitude (z-score)
        - Ripple frequency [Hz]
        - Indication of sharpest transient in trial
        - Ripple duration in seconds
        - Ripple relative amplitude [dB]
        - Amount of detected discharges in trial based on changes in the IED band
        - Indicator of whether the ripple passed the cleaning criteria (1 if passed, 0 otherwise) 
    """

    thresh_level = [2, 4]
    ripple_band = [70, 180]

    # 1. Band-pass filtering 
    print('- Filtering and Hilbert transform [-]')
    filter_order = 10
    low_cut = ripple_band[0] * 0.8 # [Hz] low cut-off frequency for high-pass filter (20% less to compensate for attenuation)
    b_hp, a_hp = signal.butter(filter_order, (low_cut / srate) * 2, btype = 'highpass', analog=False) 
    high_cut = ripple_band[1] * 1.2 # [Hz] high cut-off frequency for low-pass filter (20% more to compensate for attenuation)
    b_lp, a_lp = signal.butter(filter_order, (high_cut / srate) * 2, btype = 'lowpass', analog=False)

    # apply filtering and get envelope
    filt_signal_in = signal.filtfilt(b_hp, a_hp, signal_in) # HP
    filt_signal_in = signal.filtfilt(b_lp, a_lp, filt_signal_in) #LP

    # Hilbert transform
    filt_signal_in_envelope = np.abs(signal.hilbert(filt_signal_in))

    # To calculate ripple amplitude in dB relative to median
    ENV = filt_signal_in_envelope/np.nanmedian(filt_signal_in_envelope)
    ENV = 10*np.log10(ENV) 

    # Clipping the signal:
    ## Before clipping we stimate robust mean and std
    rbst_mean, rbst_std, inliers, outliers = robust_mean(filt_signal_in_envelope[0], dim=0, k=thresh_level[1], fit=False)
    top_lim = rbst_mean + thresh_level[1]*rbst_std
    top_lim_DEPR = np.mean(filt_signal_in_envelope) + thresh_level[1]*np.std(filt_signal_in_envelope)
    filt_signal_in_envelope_clip = filt_signal_in_envelope.copy()
    filt_signal_in_envelope_clip[filt_signal_in_envelope_clip > top_lim] = top_lim
        
    # Squaring:
    filt_signal_in_sq_envelope_clip = filt_signal_in_envelope_clip**2

    # Smoothing using a lowpass filter with FIR kaiserwin 
    LPcutoff = np.round(np.mean(ripple_band)/np.pi) # in Hz (see Stark et al. 2014 for a similar method)
        #Note: the cutoff frequency is 40 Hz when ripple-band frequency window is 70-180 Hz 
    transition_width = 10 # [Hz]

    b_lp_kaiser = design_kaiser_filter(LPcutoff, transition_width, srate)

    filt_signal_in_sq_envelope_clip = signal.filtfilt(b_lp_kaiser, [1], filt_signal_in_sq_envelope_clip)

    # Compute means and std:
    avg_sqSignal = np.nanmean(filt_signal_in_sq_envelope_clip, axis=1)
    stdev_sqSignal = np.nanstd(filt_signal_in_sq_envelope_clip, axis=1)        

    # Squaring and filtering the UNCLIPPED signal:
    filt_signal_in_envelope = filt_signal_in_envelope**2
    filt_signal_in_envelope = signal.filtfilt(b_lp_kaiser, [1], filt_signal_in_envelope)

    # Normalize with zscore - NOte: using the avergade & std computed with the clipped signal
    filt_signal_in_envelope = (filt_signal_in_envelope - avg_sqSignal)/stdev_sqSignal

    # - Find candidate ripple events ------------------------------------------
    print('- Looking candidate ripple events [-]')
    # [Notes:]  - Minimum duration = 25 ms
    #           - Maximum duration = 200 ms  
    #           - Minimum peak distance = 500 ms (avoiding overlap)
    # -------------------------------------------------------------------------
    minsamples = np.round((20/1000) * srate) # Minimum duration 20 ms (sampling points) 
    maxsamples = np.round((200/1000) * srate) # Maximum duration 200 ms (sampling points) 
    mindistance = np.round((30/1000) * srate) # Merge putative ripples of minimum distante between peaks is 30 ms or less (sampling points) 
    #MPD        = np.round((500/1000) * srate) # Minimum peak distance 500 ms (sampling points)

    # - Determine minimum & maximum peak height & distance - - - - - - - - - - 
    # [Notes:]  - pks = peak values
    #           - locs = locations 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Find peaks in the filtered signal envelope
    peaks, properties = signal.find_peaks(filt_signal_in_envelope[0], height=thresh_level[1])
    pks = properties['peak_heights']
    locs = peaks

    ## 
    '''# IED rejection (Gelinas et al. 2016 Nat. Med.):
    low_cut_IED = IED_band[0] * 0.8 # [Hz] low cut-off frequency for high-pass filter (20% less to compensate for attenuation)
    b_hp_IED, a_hp_IED = signal.butter(filter_order, (low_cut_IED / srate) * 2, btype = 'highpass', analog=False) 
    high_cut_IED = IED_band[1] * 1.2 # [Hz] high cut-off frequency for low-pass filter (20% more to compensate for attenuation)
    b_lp_IED, a_lp_IED = signal.butter(filter_order, (high_cut_IED / srate) * 2, btype = 'lowpass', analog=False)

    # apply filtering and get envelope
    filt_signal_in_IEDband = signal.filtfilt(b_hp_IED, a_hp_IED, signal_in) # HP
    filt_signal_in_IEDband = signal.filtfilt(b_lp_IED, a_lp_IED, filt_signal_in_IEDband) #LP

    # Hilbert transform
    filt_signal_in_IEDband_envelope = np.abs(signal.hilbert(filt_signal_in_IEDband))
    # Squaring:
    filt_signal_in_IEDband_envelope = filt_signal_in_IEDband_envelope**2
    filt_signal_in_IEDband_envelope = signal.filtfilt(b_lp_kaiser, [1], filt_signal_in_IEDband_envelope)
    # Zscore normalization
    filt_signal_in_IEDband_envelope = (filt_signal_in_IEDband_envelope - np.nanmean(filt_signal_in_IEDband_envelope))/np.nanstd(filt_signal_in_IEDband_envelope)

    print('- Rejecting IEDs events [-]')
    # Ignore IED-ripples events (event that coincide within 50 ms):
    IED_locs, IED_properties = signal.find_peaks(filt_signal_in_IEDband_envelope[0], height=thresh_level[1])
    IED_pks = IED_properties['peak_heights']
    IED_rej_count = 0
    pre_rej_count = len(IED_locs)'''

    IED_rej_count = 0
    IED_locs = IED_events[:,0]
    pre_rej_count = len(IED_locs)

    for i in range(len(IED_locs)):
        tmp = (np.where(np.abs(locs-IED_locs[i]) < (0.05*srate)))
        if len(tmp[0]) > 0:
            locs = np.delete(locs, tmp)
            pks = np.delete(pks, tmp)
            IED_rej_count += 1
    print(f'*** rejected {IED_rej_count} / {pre_rej_count} events based on IED band analysis')

    ##

    # Finding ripple boundaries considering that event duration was expanded until ripple power fell below 2 SD
    boundaries = []
    width = []
    ripplepeaks = np.full(len(locs), np.nan)
    rippletrough = np.full(len(locs), np.nan)
    sharptransient = np.full(len(locs), np.nan)
    relativeampl = np.full(len(locs), np.nan) # relative amplitude of the ripple peak
        
    #for k in locs:
    for k, count in zip(locs, range(len(locs))):
        # Find the starting point of the peak
        start = k
        while True:
            if start == 0:  # Handle edge case at start of signal
                break
            
            start = start - 1
            if filt_signal_in_envelope[0,start] < thresh_level[0]:
                break
        
        # Find the ending point of the peak
        end = k
        while True:
            if end == len(filt_signal_in_envelope[0]) - 1:  # Handle edge case at end of signal
                break
            
            end = end + 1
            if filt_signal_in_envelope[0,end] < thresh_level[0]:
                break
        
        width.append(end - start)
        boundaries.append((start, end))

        # Detect negative peak position for each ripple (closest to ripple's power peak)
        minpos,_ = signal.find_peaks((-1)*filt_signal_in[0,start:end]) # on the band-pass filtered signal

        if len(minpos) == 0:
            rippletrough[count] = np.nan
            ripplepeaks[count] = np.nan
            sharptransient[count] = np.nan
            relativeampl[count] = np.nan 
        else:
            maxamp = np.argmax(ENV[0,start:end])
            minpos = minpos - 1
            maxamp = maxamp - 1
            tmp = np.argmin(np.abs(minpos - maxamp))
            min_index = minpos[tmp]
            peak_position = np.min([(start + min_index), len(signal_in[0])])
            
            rippletrough[count] = peak_position # save ripple trhoug position
            ripplepeaks[count] = len(minpos) # save number of peaks in each segment
            sharptransient[count] = np.max(np.abs(np.diff(signal_in[0,start:end]))) # save sharpest transient for later spike removal
            relativeampl[count] = ENV[0, peak_position] # save relative amplitude of the ripple peak

    # Merge ripples if inter-ripple period is less than minDistance:
    reject_merged = np.zeros(len(boundaries)) # Initialize a vector to reject merged ripples
    for k in range(1, len(boundaries)):
        if (boundaries[k][0] - boundaries[k-1][1]) < mindistance:
            # Merge
            boundaries[k-1] = (boundaries[k-1][0], boundaries[k][1])
            width[k-1] = boundaries[k-1][1] - boundaries[k-1][0]
            #boundaries[k] = (0,0)
            reject_merged[k] = 1
            width[k-1] = boundaries[k-1][1] - boundaries[k-1][0]

    # Filter width according to minsamples and merges ripples
    locs = locs[np.logical_and.reduce((reject_merged != 1, np.array(width) >= minsamples, np.array(width) <= maxsamples))]
    pks = pks[np.logical_and.reduce((reject_merged != 1, np.array(width) >= minsamples, np.array(width) <= maxsamples))]
    width_filt = np.array(width)[np.logical_and.reduce((reject_merged != 1, np.array(width) >= minsamples, np.array(width) <= maxsamples))]
    rippletrough = rippletrough[np.logical_and.reduce((reject_merged != 1, np.array(width) >= minsamples, np.array(width) <= maxsamples))]
    ripplepeaks = ripplepeaks[np.logical_and.reduce((reject_merged != 1, np.array(width) >= minsamples, np.array(width) <= maxsamples))]
    sharptransient = sharptransient[np.logical_and.reduce((reject_merged != 1, np.array(width) >= minsamples, np.array(width) <= maxsamples))]
    relativeampl = relativeampl[np.logical_and.reduce((reject_merged != 1, np.array(width) >= minsamples, np.array(width) <= maxsamples))]

    # - Now go through the signal and determine ripple characteristics --------
    #print('- Determine ripple characteristics [-]')
    #ripplepeaks, rippletrough, sharptransient = ripple_char(signal_in, locs, maxsamples)

    # - Render peak width into secs -------------------------------------------
    width_filt = width_filt / srate

    # - Render ripple peaks per epoch into Hz ---------------------------------
    ripplepeaks = ripplepeaks * (srate/maxsamples) 

    # - Remove NaN values -----------------------------------------------------
    # TODO

    # - Create a basic trl ----------------------------------------------------
    # [Note:] Defines trial centered around ripple trough as follows:
    #   - [:,0]  Trial onset: ripple trough - settings.trange*data.fsample [samples]
    #   - [:,1]  Trial offset: ripple trough + settings.trange*data.fsample [samples]
    #   - [:,2]  Trigger offset: set at -settings.trange*data.fsample [samples]
    #   - [:,3]  Ripple peak amplitude (z-score)
    #   - [:,4]  Ripple frequency [Hz]
    #   - [:,5]  Indication of sharpest transient in trial
    #   - [:,6]  Ripple duration in seconds
    #   - [:,7]  Ripple relative amplitude [dB]
    #   - [:,8]  Cross referencing with discharge detection (optional) : amount of detected discharges in trial
    # -------------------------------------------------------------------------  

    #NOTA: voy a mantener la misma estructura de salida entre los distintos algoritmos para homogeneizar pero siguiendo los
    #criterios de cada uno 

    # Si hay informacion sobre las descargas
    if len(IED_locs)!=0:
        num_discharges = num_discharges_in_ripple_trial(rippletrough-t_range*srate, rippletrough+t_range*srate, IED_locs)
        trl = np.column_stack((rippletrough-t_range*srate, rippletrough+t_range*srate, np.full(len(rippletrough), -2.5*srate), pks, ripplepeaks, sharptransient, width_filt, relativeampl, num_discharges))

    # Si NO hay informacion sobre las descargas
    else:
        trl = np.column_stack((rippletrough-t_range*srate, rippletrough+t_range*srate, np.full(len(rippletrough), -2.5*srate), pks, ripplepeaks, sharptransient, width_filt, relativeampl, np.zeros(len(rippletrough))))


    # - Remove any potential trials that: -------------------------------------
    #       - Start prior to the recording onset (i.e., ripple detected at the very start)
    #       - Exceed the recording (i.e., ripple detected at the very end
    # -------------------------------------------------------------------------
    trl = trl[trl[:,0]>0,:] # - Remove trial prior to recording onset
    trl = trl[trl[:,1]<len(signal_in[0]),:] # - Remove trials that occur past recording end


    # 2. Cleaning up the trl matrix
    #print('- Cleaning up the trl matrix according to criteria [-]')

    tIndex = np.ones(len(trl))

    '''
    amp_th = [2,4] # amplitude threshold
    minfreq = 80; # minimum frequency
    sharptransient_th = 2 # sharp transient threshold
    #overlap = 0 # ??
    trange = 2.5
    min_num_ripples = 10

    # Compute logical AND for multiple conditions
    tIndex = np.logical_and.reduce((
        trl[:,3] >= amp_th[0], trl[:,3] <= amp_th[1], # - Amplitude criteria
        trl[:,4] > minfreq, # - Frequency criteria
        trl[:,5] < sharptransient_th, # - Sharp transient criteria
        trl[:,6] > (np.median(trl[:,6]) - np.std(trl[:,6])), # - Duration criteria
        trl[:,7] == 0 # - IED criteria  
    ))
    '''
    # Se concatena como una nueva columna cuales son los putative-ripples que pasan el proceso de limpienza (=1) y los que no (=0)
    trl = np.column_stack((trl, tIndex))

    return trl


"""
Determines if a peak in a signal is prominent based on its amplitude.
A peak is considered prominent if its amplitude is significantly higher 
than the amplitudes at the beginning and end of the signal.
Args:
    signal (list or numpy.ndarray): The signal data as a list or array of numerical values.
    peak_index (int): The index of the peak in the signal to be evaluated.
Returns:
    bool: True if the peak is prominent, False otherwise.
"""
def is_prominent_peak(signal, peak_index):
    peak_amplitude = signal[peak_index]
    threshold_amplitude = 0.8 * peak_amplitude
    
    # Check if amplitude at the beginning and end is lower than threshold
    beginning_amplitude = signal[0]
    end_amplitude = signal[-1]
    
    return beginning_amplitude <= threshold_amplitude and end_amplitude <= threshold_amplitude


def robust_mean(data, dim=None, k=3, fit=False):
    """
    Calculates mean and standard deviation discarding outliers.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array
    dim : int, optional
        Dimension along which to calculate mean (default: first non-singleton dimension)
    k : float, optional
        Number of sigmas for outlier cutoff (default: 3)
    fit : bool, optional
        Whether to use fitting to robustly estimate mean (default: False)

    Returns
    -------
    tuple: (finalMean, stdSample, inlierIdx, outlierIdx)
        - finalMean: Robust mean of the data
        - stdSample: Standard deviation of the data
        - inlierIdx: Indices of inliers
        - outlierIdx: Indices of outliers
    """
    # Convert to float to handle NaN
    original_dtype = data.dtype
    data_float = data.astype(float)
    
    # Input validation
    if data_float.size == 0:
        raise ValueError('Please supply non-empty data to robust_mean')
    
    # Determine dimension
    if dim is None:
        if data_float.ndim == 2 and (data_float.shape[0] == 1 or data_float.shape[1] == 1):
            dim = 0 if data_float.shape[0] == 1 else 1
        else:
            dim = 0
    
    # Check for insufficient data
    finite_count = np.sum(np.isfinite(data_float))
    if finite_count < 4:
        print('Warning: Less than 4 data points!')
        return (np.nanmean(data_float, axis=dim).astype(original_dtype), 
                np.full(np.delete(data_float.shape, dim), np.nan), 
                np.where(np.isfinite(data_float))[0], 
                np.array([]))
    
    # Fitting for 1D data if requested
    if fit:
        if data_float.ndim > 1 and max(data_float.shape) > 1:
            raise ValueError('Fitting is currently only supported for 1D data')
        
        # Define objective function for minimization
        def obj_func(x):
            return np.median(np.abs(data_float - x))
        
        # Use scipy.optimize.minimize to find robust mean
        res = minimize(obj_func, np.median(data_float), method='Nelder-Mead')
        median_data = res.x[0]
    else:
        # Use numpy's median along specified dimension
        median_data = np.nanmedian(data_float, axis=dim)
    
    # Expand median_data to match data shape
    expanded_median = np.expand_dims(median_data, axis=dim)
    
    # Magic numbers from original implementation
    magic_number2 = 1.4826**2
    
    # Calculate squared residuals
    res2 = (data_float - expanded_median)**2
    med_res2 = np.maximum(np.nanmedian(res2, axis=dim), np.finfo(float).eps)
    
    # Test value for outlier detection
    test_value = res2 / (magic_number2 * med_res2[(slice(None),) * dim + (np.newaxis,) * (data_float.ndim - dim)])
    
    # Identify inliers and outliers
    inlier_mask = test_value <= k**2
    inlier_idx = np.where(inlier_mask)
    outlier_idx = np.where(~inlier_mask)
    
    # Calculate standard deviation for inliers
    if finite_count > 4:
        # For single dimensional data
        if data_float.ndim == 1 or max(data_float.shape) == 1:
            flat_res2 = res2[inlier_mask]
            std_sample = np.sqrt(np.sum(flat_res2) / (len(flat_res2) - 4))
        else:
            # For multi-dimensional data
            res2_masked = res2.copy()
            res2_masked[~inlier_mask] = np.nan
            n_inliers = np.sum(~np.isnan(res2_masked), axis=dim)
            good_idx = np.sum(np.isfinite(res2_masked), axis=dim) > 4
            
            std_sample = np.full(np.delete(data_float.shape, dim), np.nan)
            std_sample[good_idx] = np.sqrt(np.nansum(res2_masked[good_idx], axis=dim) / (n_inliers[good_idx] - 4))
    else:
        std_sample = np.full(np.delete(data_float.shape, dim), np.nan)
    
    # Calculate final mean
    final_mean = np.nanmean(np.where(inlier_mask, data_float, np.nan), axis=dim)
    
    return final_mean, std_sample, inlier_idx[0], outlier_idx[0]
'''
def robust_mean(data, dim=None, k=3, fit=False):
    """
    Calculates mean and standard deviation discarding outliers.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array
    dim : int, optional
        Dimension along which to calculate mean (default: first non-singleton dimension)
    k : float, optional
        Number of sigmas for outlier cutoff (default: 3)
    fit : bool, optional
        Whether to use fitting to robustly estimate mean (default: False)

    Returns
    -------
    tuple: (finalMean, stdSample, inlierIdx, outlierIdx)
        - finalMean: Robust mean of the data
        - stdSample: Standard deviation of the data
        - inlierIdx: Indices of inliers
        - outlierIdx: Indices of outliers
    """

    data = data[0]

    # Convert to float to handle NaN
    original_dtype = data.dtype
    data_float = data.astype(float)
    
    # Input validation
    if data_float.size == 0:
        raise ValueError('Please supply non-empty data to robust_mean')
    
    # Determine dimension
    if dim is None:
        if data_float.ndim == 2 and (data_float.shape[0] == 1 or data_float.shape[1] == 1):
            dim = 0 if data_float.shape[0] == 1 else 1
        else:
            dim = 0
    
    # Check for insufficient data
    finite_count = np.sum(np.isfinite(data_float))
    if finite_count < 4:
        print('Warning: Less than 4 data points!')
        return (np.nanmean(data_float, axis=dim).astype(original_dtype), 
                np.full(np.delete(data_float.shape, dim), np.nan), 
                np.where(np.isfinite(data_float))[0], 
                np.array([]))
    
    # Fitting for 1D data if requested
    if fit:
        if data_float.ndim > 1 and max(data_float.shape) > 1:
            raise ValueError('Fitting is currently only supported for 1D data')
        
        # Define objective function for minimization
        def obj_func(x):
            return np.median(np.abs(data_float - x))
        
        # Use scipy.optimize.minimize to find robust mean
        res = minimize(obj_func, np.median(data_float), method='Nelder-Mead')
        median_data = res.x[0]
    else:
        # Use numpy's median along specified dimension
        median_data = np.nanmedian(data_float, axis=dim)
    
    # Expand median_data to match data shape
    expanded_median = np.expand_dims(median_data, axis=dim)
    
    # Magic numbers from original implementation
    magic_number2 = 1.4826**2
    
    # Calculate squared residuals
    res2 = (data_float - expanded_median)**2
    med_res2 = np.maximum(np.nanmedian(res2, axis=dim), np.finfo(float).eps)
    
    # Test value for outlier detection
    test_value = res2 / (magic_number2 * med_res2[(slice(None),) * dim + (np.newaxis,) * (data_float.ndim - dim)])
    
    # Identify inliers and outliers
    inlier_mask = test_value <= k**2
    inlier_idx = np.where(inlier_mask)
    outlier_idx = np.where(~inlier_mask)
    
    # Calculate standard deviation for inliers
    if finite_count > 4:
        # For single dimensional data
        if data_float.ndim == 1 or max(data_float.shape) == 1:
            flat_res2 = res2[inlier_mask]
            std_sample = np.sqrt(np.sum(flat_res2) / (len(flat_res2) - 4))
        else:
            # For multi-dimensional data
            res2_masked = res2.copy()
            res2_masked[~inlier_mask] = np.nan
            n_inliers = np.sum(~np.isnan(res2_masked), axis=dim)
            good_idx = np.sum(np.isfinite(res2_masked), axis=dim) > 4
            
            std_sample = np.full(np.delete(data_float.shape, dim), np.nan)
            std_sample[good_idx] = np.sqrt(np.nansum(res2_masked[good_idx], axis=dim) / (n_inliers[good_idx] - 4))
    else:
        std_sample = np.full(np.delete(data_float.shape, dim), np.nan)
    
    # Calculate final mean
    final_mean = np.nanmean(np.where(inlier_mask, data_float, np.nan), axis=dim)
    
    return final_mean, std_sample, inlier_idx[0], outlier_idx[0]'''