#!/usr/bin/env python

import gpu_louvain
import time
import numpy as np
import pandas as pd
import pycuda.driver as drv
from km3net.kernels import QuadraticDifferenceSparse, Match3BSparse, PatternSparse
import km3net.util as util
import sys
import os
sys.path.append(os.getcwd())


# group_id = "g8788"    # hits: 130
# group_id = "g101"       # hits: 63
# group_id = "g16194"     # hits: 42
# group_id = "g3119"      # hits: 31
# group_id = "g819"       # hits: 20
# group_id = "g4077"      # hits: 14
# group_id = "bck_1m"
# group_id = 'mixed'
# group_id = 'full_data'
# group_id = 'rand1000'
# group_id = 'grandfull'
# group_id = 'all_groups'
# group_id = '1m'

# group_id = '1k'
# group_id = '10k'
# group_id = '100k'
# group_id = '1m'
group_id = '10m'

print("We're good to go!")

# Load data
# N, x, y, z, ct, hit, tot, dom = util.get_real_input_data(f"louvain_data/sample_{group_id}.csv")
N, x, y, z, ct, hit, tot, dom = util.get_real_input_data(f"louvain_data/sample_{group_id}_window.csv")
print("Read", N, "hits from file")

# window_size = int(0.5 * 1e5)
window_size = N
window_step_size = N
context, cc = util.init_pycuda()

cpm_resolution = 0.1
pattern_threshold = 0.01

pattern_file = 'full_data_pattern_mtx_50_50_100_100.csv'
pattern = np.loadtxt(open(pattern_file, "rb"), delimiter=",")
pattern = np.array(pattern.flatten()).astype(np.float32)

pattern_kernel = PatternSparse(window_size, sliding_window_width=100, cc=cc, ptt_threshold=pattern_threshold)
gpuKernel = gpu_louvain.KernelWrapper()

current_step = 0

result_list = []
while True:

    info_progress = (current_step/N) * 100

    slice_x = x[current_step:current_step+window_size]
    slice_y = y[current_step:current_step+window_size]
    slice_z = z[current_step:current_step+window_size]
    slice_ct = ct[current_step:current_step + window_size]
    slice_hit = hit[current_step:current_step + window_size]
    slice_dom = dom[current_step:current_step + window_size]
    slice_N = len(slice_ct)

    # brak if slice empty
    if slice_N == 0:
        break

    # run correlation
    d_col_idx, d_prefix_sums, d_degrees, d_total_hits = pattern_kernel.compute(slice_N, slice_x, slice_y, slice_z, slice_ct, pattern)

    degrees = np.zeros(slice_N).astype(np.int32)
    drv.memcpy_dtoh(degrees, d_degrees)

    prefix_sums = np.zeros(slice_N).astype(np.int32)
    drv.memcpy_dtoh(prefix_sums, d_prefix_sums)

    total_hits = degrees.sum()
    col_idx = np.zeros(total_hits).astype(np.int32)
    drv.memcpy_dtoh(col_idx, d_col_idx)

    # run CPM
    gpuKernel.run(info_progress, slice_N, col_idx, prefix_sums, degrees, cpm_resolution, 0.02)

    # calculate L1 hits for given slice
    l1_hits = set([])
    for i in range(slice_N-1):
        time_i = slice_ct[i]
        idx_off = 0

        while True:
            idx_off += 1
            j = i + idx_off

            if j > (slice_N - 1):
                break
                
            time_j = slice_ct[j]
            if time_j - time_i < 3:
                if slice_dom[i] == slice_dom[j]:
                    l1_hits.add(i)
                    l1_hits.add(j)
            else:
                break

    #append result
    result_slice = {}
    result_slice['max_comm_size'] = max(gpuKernel.get_community_sizes())
    result_slice['sum_hit_class'] = sum(gpuKernel.get_hit_classes())
    result_slice['act_class'] = 1 if 1 in slice_hit else 0
    result_slice['act_hits'] = sum(slice_hit)
    result_slice['L1_hits'] = len(l1_hits)
    result_list.append(result_slice)
    #adjust next slice
    current_step += window_step_size

# df = pd.DataFrame(result_list)
# output_filename = f'louvain_output/out_{str(group_id)}_window_eval_res{cpm_resolution}_thres{pattern_threshold}'
# df.to_csv(f'{output_filename}.csv')
