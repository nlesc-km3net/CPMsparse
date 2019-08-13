#!/usr/bin/env python

import sys
import os
sys.path.append(os.getcwd())

import cpmsparse.util as util
from cpmsparse.kernels import PatternSparse, CpmSparse
import pandas as pd
import numpy as np

group_id = 'g8788'  # event hits: 130


print("We're good to go!")

context, cc = util.init_pycuda()

# Load data
N, x, y, z, ct = util.get_real_input_data(f"sample_data/sample_{group_id}.csv")
print("Read", N, "hits from file")

# Load Pattern Matrix
pattern_file = 'pattern_mtx_50_50_100_100.csv'
pattern = np.loadtxt(open(pattern_file, "rb"), delimiter=",")
pattern = np.array(pattern.flatten()).astype(np.float32)

# Correlation criterion
pattern_threshold = 0.01        # the pattern correlation threshold [percent]
pattern_time_bin_count = 50     # number of bins for time domain
pattern_dist_bin_count = 50     # number of bins for distance domain
pattern_time_limit = 100        # pattern time limit - the time multiplied by the speed of light [meters]
pattern_dist_limit = 100        # pattern distance limit [meters]
pattern_kernel = PatternSparse(N, cc=cc,
                            ptt_threshold=pattern_threshold,
                            ptt_time_bin_count=pattern_time_bin_count,
                            ptt_dist_bin_count=pattern_dist_bin_count,
                            ptt_time_limit=pattern_time_limit,
                            ptt_dist_limit=pattern_dist_limit)
d_col_idx, d_prefix_sums, d_degrees, total_hits = pattern_kernel.compute(N, x, y, z, ct, pattern)

# Community detection and classification
cpm_resolution = 0.1            # CPM resolution parameter
cpm_kernel = CpmSparse(N, cc=cc)
classified_hits = cpm_kernel.compute(d_col_idx, d_prefix_sums, cpm_resolution)
print("Number of classified hits: ", classified_hits)
