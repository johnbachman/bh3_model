"""Code to load and calculate mean and average of the BH3 profiling
titration data."""

import numpy
import matplotlib.pyplot as plt

# Prepare the data
# ================
#wt = numpy.loadtxt('data/wt_mef_raw.csv', skiprows=1, delimiter=',')
#data = numpy.loadtxt('data/baxko_mef_raw.csv', skiprows=1, delimiter=',')
#data = numpy.genfromtxt('data/baxko_mef_raw.csv', skiprows=1, delimiter=',',
#                        missing_values='', filling_values=numpy.nan)
data = numpy.genfromtxt('data/bakko_mef_raw.csv', skiprows=1, delimiter=',',
                        missing_values='', filling_values=numpy.nan)
#wt = numpy.loadtxt('data/bakko_mef_raw.csv', skiprows=1, delimiter=',')

# Globals
# =======
NUM_PTS = 37
NUM_REPLICATES = 3
NUM_CONCS = 9
NUM_COLS = 21
TIME = 0
DMSO = 1
FCCP = 2
BIM_START = 3
BID_START = 12
BIM_RANGE = range(BIM_START, BIM_START + NUM_CONCS)
BID_RANGE = range(BID_START, BID_START + NUM_CONCS)
#BIM_RANGE = range(BIM_START, BIM_START + 5)
#BID_RANGE = range(BID_START, BID_START + 5)

concs = numpy.array([100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01])

def normalize_to_absolute_max():
    """Normalize all values in the dataset to within the range
    [0, 1], where 0 and 1 are defined by the min and max of the dataset.

    After all values have been normalized, calulate the mean and standard
    deviation of each column and return as a tuple of arrays.

    Returns a tuple (data_mean, data_sd).
    """

    # Normalize the data to the range [0, 1]
    # ======================================
    norm_data = numpy.zeros(numpy.shape(data))

    # Ignore time column and any nans when finding absolute max and min
    raw_max = numpy.nanmax(data[:,1])
    raw_min = numpy.nanmin(data[:,1:])

    # Copy the time column over into the normalized data array
    norm_data[:,0] = data[:,0]

    for row in range(0, NUM_PTS * NUM_REPLICATES):
        # Don't normalize the time column (i.e., column 0)
        for col in range(1, NUM_COLS):
            raw_val = data[row, col]
            norm_data[row, col] = (raw_val - raw_min) / (raw_max - raw_min)

    # Calculate the mean and SD for each column
    # =========================================
    data_mean = numpy.zeros((NUM_PTS, NUM_COLS))
    data_sd = numpy.zeros((NUM_PTS, NUM_COLS))

    data_mean[:,0] = data[0:NUM_PTS,0] # Copy time column

    for col_index in range(1,NUM_COLS): # Leave out time column
        for time_index in range(0, NUM_PTS):
            # Initialize total and squared error to 0
            total = 0
            sq_err = 0
            count = 0

            # Calculate the mean
            for rep_index in range(0, NUM_REPLICATES):
                val = norm_data[(rep_index * NUM_PTS) + time_index, col_index]
                if not numpy.isnan(val):
                    total += val
                    count += 1 
            avg = float(total) / float(count)

            # Calculate the SD
            if count > 1:
                for rep_index in range(0, NUM_REPLICATES):
                    val = norm_data[(rep_index * NUM_PTS) + time_index, col_index]
                    if not numpy.isnan(val):
                        sq_err += (val - avg)**2
                sd = numpy.sqrt( (1./(count-1)) * sq_err)
            else:
                sd = numpy.nan

            # Fill in the entries of the mean and SD arrays
            data_mean[time_index, col_index] = avg
            data_sd[time_index, col_index] = sd

    return (data_mean, data_sd)

def normalize_to_relative_max(bim_reference_col, bid_reference_col):
    """Normalize all values in the dataset to within the range
    [0, 1], where 1 is defined as the signal from a reference
    curve (e.g., DMSO or the lowest BH3 peptide concentration).
    Zero is defined as the lowest value in the dataset (usually FCCP,
    but not necessarily).

    The reference curve is first averaged across its three replicates and
    the averaged raw values are used to perform the normalization.

    Once all values have been normalized, caclulate the mean and standard
    deviation of each column and return as a tuple of arrays.

    Returns a tuple (data_mean, data_sd)
    """


    # Calculate the mean of the replicates of the reference curve
    # ===========================================================
    # Check args
    if (bim_reference_col not in BIM_RANGE or
        bid_reference_col not in BID_RANGE):
        raise Exception("Reference column index not in appropriate range.")

    # For Bim:
    bim_reference_mean = numpy.zeros(NUM_PTS)
    
    for time_index in range(0, NUM_PTS):
        total = 0
        count = 0
        for rep_index in range(0, NUM_REPLICATES):
            ref_val = data[(rep_index * NUM_PTS) + time_index,
                           bim_reference_col]
            if not numpy.isnan(ref_val):
                total += ref_val
                count += 1
        avg = float(total) / float(count)
        bim_reference_mean[time_index] = avg

    # For Bid:
    bid_reference_mean = numpy.zeros(NUM_PTS)
    
    for time_index in range(0, NUM_PTS):
        total = 0
        count = 0
        for rep_index in range(0, NUM_REPLICATES):
            ref_val = data[(rep_index * NUM_PTS) + time_index,
                           bid_reference_col]
            if not numpy.isnan(ref_val):
                total += ref_val
                count += 1
        avg = float(total) / float(count)
        bid_reference_mean[time_index] = avg

    # Normalize the data to the range [0, 1]
    # ======================================
    norm_data = numpy.zeros(numpy.shape(data))

    # Ignore time column and any nans when finding absolute min
    raw_min = numpy.nanmin(data[:,1:])

    # Copy the time column over into the normalized data array
    norm_data[:,0] = data[:,0]

    for time_index in range(0, NUM_PTS):
        for rep_index in range(0, NUM_REPLICATES):
            # Don't normalize the time column (i.e., column 0)
            for col in range(1, NUM_COLS):
                row_index = (rep_index * NUM_PTS) + time_index
                raw_val = data[row_index, col]

                if col in BIM_RANGE:
                    ref_val = bim_reference_mean[time_index]
                else: # Use Bid ref by default
                    ref_val = bid_reference_mean[time_index]

                if not numpy.isnan(ref_val):
                    norm_data[row_index, col] = (raw_val - raw_min) / \
                                                (ref_val - raw_min)
                else:
                    raise Exception("Reference val is NaN.")

    # Calculate the mean and SD for each column
    # =========================================
    data_mean = numpy.zeros((NUM_PTS, NUM_COLS))
    data_sd = numpy.zeros((NUM_PTS, NUM_COLS))

    data_mean[:,0] = data[0:NUM_PTS,0] # Copy time column

    for col_index in range(1,NUM_COLS): # Leave out time column
        for time_index in range(0, NUM_PTS):
            # Initialize total and squared error to 0
            total = 0
            sq_err = 0
            count = 0

            # Calculate the mean
            for rep_index in range(0, NUM_REPLICATES):
                val = norm_data[(rep_index * NUM_PTS) + time_index, col_index]
                if not numpy.isnan(val):
                    total += val
                    count += 1 
            avg = float(total) / float(count)

            # Calculate the SD
            if count > 1:
                for rep_index in range(0, NUM_REPLICATES):
                    val = norm_data[(rep_index * NUM_PTS) + time_index, col_index]
                    if not numpy.isnan(val):
                        sq_err += (val - avg)**2
                sd = numpy.sqrt( (1./(count-1)) * sq_err)
            else:
                sd = numpy.nan

            # Fill in the entries of the mean and SD arrays
            data_mean[time_index, col_index] = avg
            data_sd[time_index, col_index] = sd

    return (data_mean, data_sd)

def plot_data(data_mean, data_sd, time_col, col_range, logy=False):
    plt.ion()
    plt.figure()
    for col_index in col_range:
        if logy:
            plt.plot(data_mean[:, time_col],
                     numpy.log(data_mean[:, col_index]))
        else:
            plt.errorbar(data_mean[:, time_col], data_mean[:, col_index],
                    yerr=data_sd[:, col_index])
    #if dmso_col is not None:
    #    plt.plot(data_mean[:, time_col], data_mean[:, dmso_col], 'ro')
    #    plt.errorbar(data_mean[:, time_col], data_mean[:, dmso_col],
    #            yerr=data_sd[:, dmso_col])

    plt.show()

"""
MAX_TIME_INDEX = 37
exp_tspan = data_mean[0:MAX_TIME_INDEX, TIME]
fccp_avgs = data_mean[0:MAX_TIME_INDEX, FCCP]
fccp_stds = data_sd[0:MAX_TIME_INDEX, FCCP]
dmso_avgs = data_mean[0:MAX_TIME_INDEX, DMSO]
dmso_stds = data_sd[0:MAX_TIME_INDEX, DMSO]

#conc_pick = 0 # range from [0, 8]
#bim_avgs = data_mean[0:MAX_TIME_INDEX, BIM_START + conc_pick]
#bim_stds = data_sd[0:MAX_TIME_INDEX, BIM_START + conc_pick]
#bid_avgs = data_mean[0:MAX_TIME_INDEX, BID_START + conc_pick]
#bid_stds = data_sd[0:MAX_TIME_INDEX, BID_START + conc_pick]

bim_avgs = data_mean[0:MAX_TIME_INDEX, BIM_RANGE]
bim_stds = data_sd[0:MAX_TIME_INDEX, BIM_RANGE]
bid_avgs = data_mean[0:MAX_TIME_INDEX, BID_RANGE]
bid_stds = data_sd[0:MAX_TIME_INDEX, BID_RANGE]

"""
