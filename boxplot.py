# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:36:14 2023

@author: Víctor Salvador Castrillo
victor.salvador_castrillo@sorbonne-universite.fr

Boxplot to show transient lengths for each violoncello bow

  - save your data by activating savefile = 'ON'

LOAD:
    filename:
        'your_csvfile_with_transients_detected_by_hand.csv'

    datatype:
        [('Transient number', int, 1),
        ('Bow part', np.unicode_, 512),
        ('Dynamics', np.unicode_, 512),
        ('Direction', np.unicode_, 512),
        ('Musician', np.unicode_, 512),
        ('Bow', np.unicode_, 512),
        ('Transient start', int, 1),
        ('Transient end', int, 1)]

OUTPUT: save your data by activating savefile = 'ON'
    filename:
        'your_boxplot_of_transient_lengths.svg'

"""

import matplotlib.pyplot as plt
import numpy as np

savefile = 'OFF'

# Datatype of the transient dataset
dtype = np.dtype([('Transient number', int, 1),
                 ('Bow part', np.unicode_, 512),
                 ('Dynamics', np.unicode_, 512),
                 ('Direction', np.unicode_, 512),
                 ('Musician', np.unicode_, 512),
                 ('Bow', np.unicode_, 512),
                 ('Transient start', int, 1),
                 ('Transient end', int, 1)])

transients = np.genfromtxt(
             'your_csvfile_with_transients_detected_by_hand.csv',
             dtype=dtype, delimiter=';')

bows = ['BowA', 'BowB', 'Bowmusician']
sr = 51200

# Define dictionaries for boxplotting
transients_dict = {bows[0]: [],
                   bows[1]: [],
                   bows[2]: []}

transients_dict_stroketype = {bows[0]: [],
                              bows[1]: [],
                              bows[2]: []}
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
            transients_dict_stroketype[bow].append([
                1000*transient_duration/sr,
                transient_color_dynamics[transients[i][2]],
                transient_marker_direction[transients[i][3]]])

# calculate median of all durations in order to plot it as a horizontal line
median = 1000*np.median(all_durations)/sr

# Boxplot
fig, ax = plt.subplots()
ax.boxplot(transients_dict.values(), showfliers=False)

for i, bow in enumerate(bows):
    y = np.array(transients_dict_stroketype[bow])[:, 0].astype(float)
    y_colors = np.array(transients_dict_stroketype[bow])[:, 1].astype(str)
    y_markers = np.array(transients_dict_stroketype[bow])[:, 2]
    x = np.random.normal(i+1, 0.05, size=len(y))  # Add random "jitter" to x-ax
    for j in range(len(y)):  # Didn't came up with a better solution...
        ax.scatter(x[j], y[j], c=y_colors[j],
                   marker=y_markers.tolist()[j], alpha=0.4)

ax.set_xticklabels(['Bow A', 'Bow B', 'Own bow'], fontsize=15)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.hlines(median, 0, 4, color='blue', linestyle='--',
          alpha=0.5, label='Median = ' + str(round(median, 1)) + ' ms')
ax.set_xlim([0.5, 3.5])
ax.set_ylabel('Transient duration (ms)', fontsize=15)
ax.grid(axis='y')
ax.legend(frameon=False, fontsize=13)


if savefile == 'ON':
    fig.savefig('your_boxplot_of_transient_lengths.svg',
                format='svg', dpi=1200)
