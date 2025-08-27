import matplotlib.pyplot as plt
import os
import numpy as np
import sys

def reject_outliers(data, m=10.0):
    # Filter out NaN values
    not_nan_mask = ~np.isnan(data)
    data = data[not_nan_mask]

    # Calculate deviations from median and identify outliers
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0

    # Filter out the outliers
    non_outliers_mask = s < m
    final_mask = not_nan_mask.copy()
    final_mask[not_nan_mask] = non_outliers_mask

    return data[non_outliers_mask], final_mask

# Check if the filename is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python read_columns.py <filename>")
    sys.exit(1)

# Get the filename from the command-line argument
filename = sys.argv[1]

# Read the file and skip the header
try:
    data = np.genfromtxt(filename, skip_header=1)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Split data into individual arrays based on columns
E_TS = data[:, 0]
E_R = data[:, 1]
E_P = data[:, 2]
dE = data[:, 3]
E_DFT = data[:, 4]
E_CCSD_T = data[:, 5]

# Function to calculate and print stats
def print_stats(label, array):
    avg = np.mean(array)
    std = np.std(array)
    count = len(array)
    print(f"SEM {label} = {avg:.6f} +/- {std/np.sqrt(count):.6f} (Data points: {count})")
    print(f"STD {label} = {avg:.6f} +/- {std:.6f} (Data points: {count})")
# Print stats before rejecting outliers
print("Before rejecting outliers:")
print_stats("E_TS", E_TS)
print_stats("E_R", E_R)
print_stats("E_P", E_P)
print_stats("dE", dE)
print_stats("E_DFT", E_DFT)
print_stats("E_CCSD_T", E_CCSD_T)

## Reject outliers from each array
#E_TS_filtered, _ = reject_outliers(E_TS)
#E_R_filtered, _ = reject_outliers(E_R)
#E_P_filtered, _ = reject_outliers(E_P)
#dE_filtered, _ = reject_outliers(dE)
#E_DFT_filtered, _ = reject_outliers(E_DFT)
#E_CCSD_T_filtered, _ = reject_outliers(E_CCSD_T)

# Reject outliers from each array
E_TS_filtered, mask = reject_outliers(E_TS)
E_R_filtered = E_R[mask]
E_P_filtered = E_P[mask]
dE_filtered = dE[mask]
E_DFT_filtered = E_DFT[mask]
E_CCSD_T_filtered = E_CCSD_T[mask]

# Print stats after rejecting outliers
print("\nAfter rejecting outliers:")
print_stats("E_TS", E_TS_filtered)
print_stats("E_R", E_R_filtered)
print_stats("E_P", E_P_filtered)
print_stats("dE", dE_filtered)
print_stats("E_DFT", E_DFT_filtered)
print_stats("E_CCSD_T", E_CCSD_T_filtered)

## Plot histograms for filtered data
#fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#
## Histogram for filtered E_TS
#axs[0].hist(E_TS_filtered, bins=10, color='blue', alpha=0.7)
#axs[0].set_title('Histogram of Filtered E_TS')
#axs[0].set_xlabel('E_TS values')
#axs[0].set_ylabel('Frequency')
#axs[0].get_xaxis().get_major_formatter().set_useOffset(False)
#
## Histogram for filtered E_CCSD_T
#axs[1].hist(E_CCSD_T_filtered, bins=10, color='green', alpha=0.7)
#axs[1].set_title('Histogram of Filtered E_CCSD_T')
#axs[1].set_xlabel('E_CCSD_T values')
#axs[1].set_ylabel('Frequency')
#axs[1].get_xaxis().get_major_formatter().set_useOffset(False)
#
## Adjust layout
#plt.tight_layout()
#
## Save the plot to the same directory as the input file
#output_filename = os.path.splitext(filename)[0] + "_filtered_histograms.png"
#plt.savefig(output_filename)
#
#print(f"Filtered histogram saved as {output_filename}")
