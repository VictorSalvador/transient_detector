# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:30:10 2023

@author: vicsalcas

Script to detect transients by hand

AND

Boxplot

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import glob
import os
import librosa
import librosa.display
import time

#%%
'''
 *** Transient detector from clicking from a signal plot ***

Returns a dictionary containing the transients defined by clicking the
beginning and the end on a transient plot

filenames have the following structure:
    'DVC_pointe_piano_pousse_musician1_BowB_Astring.mat'

It may not work well, yet...

I recommend testing it with 2 or 3 different .mat files in your test folder
(i.e. file_path string). If too many, it may be too long and tiring.

Or even better, just detect transients on which you are interested
  - Put the -mat files in your file_path folder
  - run this cutie
  - save your data by activating savefile

'''

############################# ----------------> IMPORTANT PARAMETERS

# File path where your data is located.
file_path = 'C:/Users/vicsalcas/Desktop/Villefavard 2022/' + \
            'Tests musiciens/Attaques musicien 1/'

# ON if you want to save the transient dataset in path_file
savefile = 'OFF'

#############################

# Datatype of the transient dataset
dtype = np.dtype([('Transient number', int, 1),
                    ('Bow part', np.unicode_, 512),
                    ('Dynamics', np.unicode_, 512),
                    ('Direction', np.unicode_, 512),
                    ('Musician', np.unicode_, 512),
                    ('Bow', np.unicode_, 512),
                    ('Transient start', int, 1),
                    ('Transient end', int, 1)])

# Message plotter during clicking process
def tellme(s, color):
    plt.suptitle(s, fontsize=16, color=color)
    plt.draw()
def textbox(text, xmin, xmax):
    textbox = plt.text(int((xmin+xmax)/2),
                       0, text, size=16,
     ha="center", va="center",
     bbox=dict(boxstyle="round",
               ec=(1., 0.5, 0.5),
               fc=(1., 0.8, 0.8),
               )
     )
    plt.show()
    aborted = plt.waitforbuttonpress() # True if key is pressed
    textbox.remove()
    
    return aborted

# Go to file directory
os.chdir(file_path)

# Create empty transient array that will contain all transients of all signals
all_transients_array = np.ndarray(shape=1, # first row to delete
                               dtype=dtype)


for filename in glob.glob('*Astring*.mat'):
    print(filename)
    # Load signal and sampling frequency
    mat = scipy.io.loadmat(file_path + filename)
    sr = int(mat['freq'])
    sig = mat['indata'][:, 0]
    
    # Lag correction for attack onset detection
    onset_lag_adjust = 1
    
    # Peak picking algorithm parameters
    delta = 0.4
    wait = 25
    
    # Calculate signal onsets from spectral envelope
    o_env = librosa.onset.onset_strength(y=sig, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr,
                                              delta=delta, wait=wait)
    onset_samples = (times[onset_frames]*sr).astype(int)
    
    # Click detect
    
    # Create empty transient array that will contain all transients of current
    # signal
    transients_array = np.ndarray(shape=len(onset_samples), 
                                   dtype=dtype)
    
    for onset in range(len(onset_samples)):
        
        # Plot signal, show signal around detected onset
        plt.figure(figsize=(18,10))
        plt.plot(sig)
        xlim_ini = onset_samples[onset]-2000
        xlim_fin = onset_samples[onset]+12000
        plt.xlim((xlim_ini, xlim_fin))
        
        # Initial message
        if onset == 0:
            aborted = textbox("You will define the limits of SEVERAL transients. \
                              \n Click to begin, press key to exit.",
                              xlim_ini, xlim_fin)
            if aborted == True:
                plt.close('all')
                is_break = True
                break
            
        # While loop until transient is detected
        while True:
            pts = []
            while len(pts) < 2:
                tellme('Select beginning and end of transient', 'b')
                pts = np.asarray(plt.ginput(2, timeout=-1))
                if len(pts) < 2:
                    tellme('Too few points, starting over', 'r')
                    time.sleep(1)  # Wait a second
        
            fill = plt.axvspan(pts[0,0], pts[1,0], facecolor='r', alpha=0.3)
        
            tellme('Happy? Key press for yes, mouse click for no', 'b')
        
            if plt.waitforbuttonpress():
                transients_array[onset][0] = onset
                transients_array[onset][1] = filename.split('_')[1]
                transients_array[onset][2] = filename.split('_')[2]
                transients_array[onset][3] = filename.split('_')[3]
                transients_array[onset][4] = filename.split('_')[4]
                transients_array[onset][5] = filename.split('_')[5]
                transients_array[onset][6] = pts[0,0].astype(int)
                transients_array[onset][7] = pts[1,0].astype(int)
                print(transients_array[onset])
                break
            
            fill.remove()
    
    # add transients detected to dataset
    all_transients_array = np.append(all_transients_array,
                                     transients_array)
    
# delete empty first row
all_transients_array = np.delete(all_transients_array, 0, 0)

# This does not work I don't know why...
tellme('All transients detected!!! Closing figures...', 'g')
time.sleep(2) # Pause 2 seconds

# Close all figures
plt.close('all')

if savefile == 'ON':
    np.savetxt('Attack_transients_Astring_Musician1_allBows_pointe_piano.csv', all_transients_array, delimiter=';', fmt='%s')


#%% Boxplot

transients = np.genfromtxt('array_edit.csv', dtype=dtype, delimiter=';')

bows = ['BowA', 'BowB', 'Bowmusician']
sr = 51200

# Define dictionaries for boxplotting
transients_dict = {bows[0]:[],
                   bows[1]:[],
                   bows[2]:[]}

transients_dict_stroketype = {bows[0]:[],
                              bows[1]:[],
                              bows[2]:[]}
# for scatter with boxplot
transient_color_dynamics = {'forte': 'r', 'piano': 'g'}
transient_marker_direction = {'pousse': '.', 'tire': 'x'}

all_durations = np.array([])


# assign transient durations to a bow for the boxplot
for bow in bows:
    transient_durations = np.array([])
    for i in range(len(transients)):
        if transients[i][5] == bow:
            print(transients[i])
            transient_duration = np.abs(transients[i][-1] - transients[i][-2])
            all_durations = np.append(all_durations, transient_duration)
            print(str(transient_duration))
            transients_dict[bow].append(1000*transient_duration/sr)
            transients_dict_stroketype[bow].append([1000*transient_duration/sr,
                                                    transient_color_dynamics[transients[i][2]],
                                                    transient_marker_direction[transients[i][3]]])

# calculate median of all durations in order to plot it as a horizontal line
median = 1000*np.median(all_durations)/sr

# Boxplot
fig, ax = plt.subplots()
ax.boxplot(transients_dict.values(), showfliers=False)

for i, bow in enumerate(bows):
    y = np.array(transients_dict_stroketype[bow])[:,0].astype(float)
    y_colors = np.array(transients_dict_stroketype[bow])[:,1].astype(str)
    y_markers = np.array(transients_dict_stroketype[bow])[:,2]
    x = np.random.normal(i+1, 0.05, size=len(y)) # Add random "jitter" to x-ax
    for j in range(len(y)): # Didn't came up with a better solution.......
        ax.scatter(x[j], y[j], c=y_colors[j], marker=y_markers.tolist()[j], alpha=0.4)

ax.set_xticklabels(['Bow A', 'Bow B', 'Own bow'], fontsize=15)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.hlines(median, 0, 4, color='blue', linestyle='--', alpha=0.5, label='Median = ' + str(round(median,1)) + ' ms')
ax.set_xlim([0.5,3.5])
ax.set_ylabel('Transient duration (ms)', fontsize=15)
ax.grid(axis='y')
ax.legend(frameon=False, fontsize=13)


# fig.savefig(os.getcwd() + '/analysis_transients/Figures/m1_pointe_handselect_delete transients over 200ms.svg', format='svg', dpi=1200)

