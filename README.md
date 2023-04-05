# transient_detection
Algorithm to detect transient regimes in bowed-string force signals on a cello bridge.

## Method

Automatic detection works as follows:

### Calculate slips from first difference of signal --> First slip = Transient onset

![screenshot](screenshots/slips_detected.png "")

### Count number of slips for each window

![screenshot](screenshots/count_nb_slips.png "")

### Calculate mean and std of the peak density function

![screenshot](screenshots/calculate_mean_std.png "")

### Detect end of transient from mean and std

![screenshot](screenshots/transient_detected.png "")