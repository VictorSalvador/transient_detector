# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:00:07 2023

@author: vicsalcas

Automatic detection of attack transients on the A string

  -> main_path : where your .mat files are

IMPORTANT:
  Slip = peak
  If detecting a big number of transients, set display = 'OFF' (time consuming)

"""

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy.io
import glob


display = 'OFF'
savefile = 'ON'
csv_filename = 'm1_spf-semitone_std-06_Astring'
main_path = os.path.expanduser('~') + '/Desktop/Villefavard 2022' \
                                    + '/Tests musiciens/Attaques musicien 1'

string = 'A'  # 'A' or 'C'
string_frequency = {'A': 220, 'C': 65.4}

# Parameters for sliding window algorithms
hop_size = 128
win_len = 2048

# Peak-picking algorithm parameters pp_parameters
pp_prominences = {'A': 0.013, 'C': 0.013}
pp_distances = {'A': 20, 'C': 20}
pp_height_factor = 1/3


# Threshold for Sliding Phase Frequency (bandwidth centered at string natural
# frequency):
#   spf_th_low = string_frequency[string]/2**(1/spf_th_wide)
#   spf_th_up = string_frequency[string]*2**(1/spf_th_wide)
# Common values:
#  24 -> semitone bandwidth
#  12 -> tone bandwidth
spf_th_wide = 24

# Threshold for the maximum standard deviation of the sliding phases timestamps
std_th = 0.6


os.chdir(main_path)


def detect_onsets_from_peaks(abs_sig_diff, peaks, sr, lagtime_peak=0.1):
    lagtime_peak_samples = int(lagtime_peak*sr)
    onset_frames = np.array([])
    # Only a peak can be an onset
    for i in range(len(peaks)):
        # If the peaks are in the first lagtime_peak seconds, we can not know
        # if it is an onset
        if peaks[i] < lagtime_peak_samples:
            continue
        # Number of peaks detected lagtime_peak seconds before current peak
        arethey_peaks_before = len(np.where(
            np.logical_and(peaks >= peaks[i] - lagtime_peak_samples,
                           peaks < peaks[i]))[0])
        # If there are no peaks lagtime_peak seconds before current peak
        # --> ONSET DETECTED
        if arethey_peaks_before == 0:
            onset_frames = np.append(onset_frames, peaks[i])
    return onset_frames/sr


count = 0

# Datatype of the transient dataset FOR BOXPLOT
dtype = np.dtype([('Transient number', int, 1),
                  ('Bow part', np.unicode_, 512),
                  ('Dynamics', np.unicode_, 512),
                  ('Direction', np.unicode_, 512),
                  ('Musician', np.unicode_, 512),
                  ('Bow', np.unicode_, 512),
                  ('Transient start', int, 1),
                  ('Transient end', int, 1)])
# Create empty transient array that will contain all transients of all signals
all_transients_array = np.ndarray(shape=1,  # first row to delete
                                  dtype=dtype)

# --- Progress bar

# Count number of .mat files
nb_files = 0
for filename in glob.glob('*' + string + 'string.mat'):
    nb_files += 1

# Create progress bar
pvar = tqdm(desc='Transient detection in progress... ', total=nb_files)


# --- Main loop

# For every signal (.mat file) in main_path
for filename in glob.glob('*' + string + 'string.mat'):

    # -- Load signal and sampling rate
    file_path = main_path + "/" + filename
    print('File: ' + filename)
    sr = int(scipy.io.loadmat(file_path)['freq'])
    sig = scipy.io.loadmat(file_path)['indata'][:, 0]

    # -- Calculate first difference of the signal in absolute value
    abs_sig_diff = np.abs(np.diff(sig))

    # -- Detect peaks from abs_sig_diff, i.e., detect slipping phases

    # Peak-picking algorithm parameters (pp): see pp_parameters section

    # Minimum height of the peaks: Adaptative threshold varying with
    # abs_sig_diff envelope
    # --> Minimum height = pp_height = maximum peak of the windowed
    #                      abs_sig_diff signal times pp_height_factor

    pp_height = np.zeros_like(abs_sig_diff)

    for i in range(len(abs_sig_diff)-win_len):
        pp_height[int(win_len/2)+i] = np.max(abs_sig_diff[i:win_len+i]) \
            * pp_height_factor

    # Adjust initial and final height of signal (if not, height=0)
    pp_height[0:int(win_len/2)] = pp_height[int(win_len/2)]
    pp_height[-int(win_len/2):] = pp_height[-int(win_len/2)-1]

    # Detect peaks
    peaks, _ = scipy.signal.find_peaks(abs_sig_diff, height=pp_height,
                                       prominence=pp_prominences[string],
                                       distance=pp_distances[string])

    # -- Detect onsets from peaks (first peak after 0.1 seconds of no peaks)
    file_onsets = detect_onsets_from_peaks(abs_sig_diff, peaks, sr,
                                           lagtime_peak=0.1)

    # -- Calculate number of peaks from abs_sig_diff signal, i.e. :
    # nb_peaks = number of slips in the temporal window of length = win_len
    nb_peaks = np.zeros_like(abs_sig_diff)
    for i in range(len(abs_sig_diff)-win_len):
        nb_peaks[int(win_len/2)+i] = len(np.where(
            np.logical_and(peaks >= i, peaks <= (win_len+i)))[0])

    # -- ONSETS LOOP

    # Transient number index (transients are saved with an index for every
    # signal, starting from 0)
    transient_number = 0

    # Array to store Slipping Phase (pseudo)Frequency (spf) values in Hz
    spf_array = np.zeros_like(abs_sig_diff)

    # Array to store standard deviation of the peak distribution in the
    # windowed abs_sig_diff signal
    std_peaks_array = np.zeros_like(abs_sig_diff)

    # For every onset detected (and then transient candidate):
    for i in range(len(file_onsets)):

        # -- Calculate transient length
        # How? -> parsing signal from current onset to next onset and
        # analysing mean and standard deviation

        # If last onset, then next onset = end of signal
        if i+1 == len(file_onsets):
            next_onset = len(abs_sig_diff)
        else:
            next_onset = file_onsets[i+1]

        # End of transient, set to -1 in case of no transient detected
        transient_end = -1

        # nb_peaks_cut = part of the current nb_peaks function (from current
        # onset to next onset or end of signal)
        nb_peaks_cut = nb_peaks[int(file_onsets[i]*sr):int(next_onset*sr)]

        # -- ANALYSIS OF THE FREQUENCY AND STD OF THE SLIPS TO DETECT HELMHOLTZ
        # Sliding window algorithm to analyse both spf_array and str_peaks
        # functions, both with thresholds to detect the beginning of the
        # Helmholtz motion and thus the end of the transient
        for j in range(len(nb_peaks_cut)-win_len):
            first_sample = int(file_onsets[i]*sr)+j
            last_sample = int(file_onsets[i]*sr)+j+win_len
            # Mean number of peaks in the current windowed nb_peaks function
            mean_peaks = np.mean(nb_peaks[first_sample:last_sample])
            # Slipping Phase (pseudo)Frequency (spf) in Hz
            spf = sr*mean_peaks/win_len
            # Std
            std_peaks = np.std(nb_peaks[first_sample:last_sample])
            # store both spf and std in respective arrays
            spf_array[first_sample] = spf
            std_peaks_array[first_sample] = std_peaks

            # - MAIN DETECTING CONDITION:
            # Helmholtz regime is reached if:
            #   - frequency = frequency(string) in some bandwidth threshold
            #                   (spf_th_wide)
            #   - std < std_th
            # THEN --> break loop and go to next detected onset

            # Lower and upper threshold for slipping phase frequency (spf)
            spf_th_low = string_frequency[string]/2**(1/spf_th_wide)
            spf_th_up = string_frequency[string]*2**(1/spf_th_wide)
            if spf_th_low < spf < spf_th_up:
                if std_peaks < std_th:
                    transient_end = int(first_sample)  # in samples
                    break

        # If Helmholtz regime never reached, transient ommited (probably caused
        # by false onset)
        if transient_end == -1:
            continue

        transients_array = np.ndarray(shape=1, dtype=dtype)
        transients_array[0][0] = transient_number
        transients_array[0][1] = filename.split('_')[1]
        transients_array[0][2] = filename.split('_')[2]
        transients_array[0][3] = filename.split('_')[3]
        transients_array[0][4] = filename.split('_')[4]
        transients_array[0][5] = filename.split('_')[5]
        transients_array[0][6] = int(file_onsets[i]*sr)
        transients_array[0][7] = transient_end

        # transient_lims = np.append(transient_lims,
        #                            np.array([filename, int(transient_number),
        #                                      int(file_onsets[i]*sr),
        #                                      transient_end]))

        # add transients detected to dataset
        all_transients_array = np.append(all_transients_array,
                                         transients_array)

        transient_number += 1

    if display == 'ON':
        fig, ax = plt.subplots(nrows=5, sharex=True)
        ax[0].plot(np.arange(len(sig))/sr, sig, label='Signal')
        ax[0].set(title=filename)
        ax[1].plot(np.arange(len(abs_sig_diff))/sr, abs_sig_diff,
                   label='First derivative signal (absolute value)')
        ax[1].scatter((np.arange(len(np.diff(sig)))/sr)[peaks],
                      np.abs(np.diff(sig))[peaks], label='Detected peaks',
                      c='r', marker='*')
        ax[1].plot(np.arange(len(pp_height))/sr, pp_height, color='g',
                   label='Maximum height for peak detection')
        ax[2].plot(np.arange(len(nb_peaks))/sr, nb_peaks, color='g',
                   label='Number of peaks (nb_peaks)')
        ax[0].vlines(file_onsets, -5, 5, color='r', alpha=0.9,
                     linestyle='--', label='Detected onsets')
        ax[1].vlines(file_onsets, 0, 0.6, color='r', alpha=0.9,
                     linestyle='--')
        ax[2].vlines(file_onsets, 0, 40, color='r', alpha=0.9,
                     linestyle='--')
        ax[0].legend(frameon=False)
        ax[1].legend(frameon=False)
        ax[2].legend(frameon=False)

        for k in range(transient_number):
            ax[0].axvspan(int(all_transients_array[-k-1][6])/sr,
                          int(all_transients_array[-k-1][7])/sr, facecolor='g',
                          alpha=0.5)
        ax[3].plot(np.arange(len(spf_array))/sr, spf_array, color='orange',
                   label='Slipping phase frequency (SPF)')
        ax[3].hlines(string_frequency[string]/2**(1/24), 0, 10, color='g',
                     alpha=0.9, linestyle='--',
                     label='Upper and lower SPF thresholds')
        ax[3].hlines(string_frequency[string]*2**(1/24), 0, 10, color='g',
                     alpha=0.9, linestyle='--')
        ax[3].legend(frameon=False)
        ax[4].plot(np.arange(len(std_peaks_array))/sr, std_peaks_array,
                   color='orange', label='Standard deviation (STD)')
        ax[4].hlines(0.6, 0, 10, color='g', alpha=0.9,
                     linestyle='--', label='Upper STD threshold')
        ax[4].legend(frameon=False)
        ax[3].vlines(file_onsets, 0, 440, color='r', alpha=0.9,
                     linestyle='--')
        ax[4].vlines(file_onsets, 0, 10, color='r', alpha=0.9,
                     linestyle='--')
        ax[4].set_xlabel('Time (s)')
        ax[4].set_ylabel('STD (Slips)')
        ax[3].set_ylabel('SPF (Hz)')
        ax[2].set_ylabel('Slips')
        ax[1].set_ylabel('Amplitude')
        ax[0].set_ylabel('Amplitude')

    # For testing just a bunch of signals
    # if count == 3:
    #     break
    
    count += 1
    pvar.update(1)

pvar.close()

# delete empty first row
all_transients_array = np.delete(all_transients_array, 0, 0)

# --- SAVE FILE WITH DETECTED TRANSIENTS FOR BOXPLOT

if savefile == 'ON':
    np.savetxt(csv_filename + '.csv', all_transients_array, delimiter=';',
               fmt='%s')
