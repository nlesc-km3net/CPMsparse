from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from cpmsparse.util import *

class PatternSparse(object):

    def __init__(self, N, sliding_window_width=1500, cc='52',
                ptt_threshold=0.01,
                ptt_time_bin_count=50,
                ptt_dist_bin_count=50,
                ptt_time_limit=100,
                ptt_dist_limit=100):
                
        kernel_name = "pattern_full"
        block_size_x = 512

        self.N = np.int32(N)
        self.sliding_window_width = np.int32(sliding_window_width)
        self.threads = (block_size_x, 1, 1)
        self.grid = (int(np.ceil(N/float(block_size_x))), 1)

        with open(get_kernel_path()+'correlate_full.cu', 'r') as f:
            kernel_string = f.read()

        prefix = ("#define block_size_x " + str(block_size_x) + "\n" +
                "#define window_width " + str(sliding_window_width) + "\n" +
                "#define ptt_threshold " + str(ptt_threshold) + "\n" +
                "#define ptt_time_bin_count " + str(ptt_time_bin_count) + "\n" +
                "#define ptt_dist_bin_count " + str(ptt_dist_bin_count) + "\n" +
                "#define ptt_time_limit " + str(ptt_time_limit) + "\n" +
                "#define ptt_dist_limit " + str(ptt_dist_limit) + "\n")
                
        kernel_string = prefix + kernel_string

        compiler_options = ['-Xcompiler=-Wall', '--std=c++11', '-O3']

        self.compute_sums = SourceModule("#define write_sums 1\n" + kernel_string, options=compiler_options,
                                         arch='compute_' + cc, code='sm_' + cc,
                                         cache_dir=False, no_extern_c=True).get_function(kernel_name)
        self.compute_sparse_matrix = SourceModule("#define write_spm 1\n" + kernel_string, options=compiler_options,
                                                  arch='compute_' + cc, code='sm_' + cc,
                                                  cache_dir=False, no_extern_c=True).get_function(kernel_name)

    def compute(self, n_count, x, y, z, ct, pattern):
        n_count = np.int32(n_count)

        d_x = ready_input(x)
        d_y = ready_input(y)
        d_z = ready_input(z)
        d_ct = ready_input(ct)
        d_pattern = ready_input(pattern)

        #run the first kernel
        row_idx = np.zeros(10).astype(np.int32)
        col_idx = np.zeros(10).astype(np.int32)
        prefix_sums = np.zeros(10).astype(np.int32)
        degrees = np.zeros(n_count).astype(np.int32)

        d_degrees = allocate_and_copy(degrees)
        d_row_idx = allocate_and_copy(row_idx)
        d_col_idx = allocate_and_copy(col_idx)
        d_prefix_sums = allocate_and_copy(prefix_sums)

        args_list = [d_row_idx, d_col_idx, d_prefix_sums, d_degrees, n_count, self.sliding_window_width, d_x, d_y, d_z, d_ct, d_pattern]
        self.compute_sums(*args_list, block=self.threads, grid=self.grid, stream=None, shared=0)

        #allocate space to store sparse matrix
        drv.memcpy_dtoh(degrees, d_degrees)
        total_correlated_hits = degrees.sum()
        col_idx = np.zeros(total_correlated_hits).astype(np.int32)
        prefix_sums = np.cumsum(degrees).astype(np.int32)

        d_col_idx = allocate_and_copy(col_idx)
        d_prefix_sums = allocate_and_copy(prefix_sums)

        args_list2 = [d_row_idx, d_col_idx, d_prefix_sums, d_degrees, n_count, self.sliding_window_width, d_x, d_y, d_z, d_ct, d_pattern]
        self.compute_sparse_matrix(*args_list2, block=self.threads, grid=self.grid, stream=None, shared=0)

        return d_col_idx, d_prefix_sums, d_degrees, total_correlated_hits


class CpmSparse(object):

    def __init__(self, N, cc='52'):
        block_size_x = 512

        self.N = np.int32(N)
        self.threads = (block_size_x, 1, 1)
        self.grid = (int(np.ceil(N/float(block_size_x))), 1)

        with open(get_kernel_path()+'cpm.cu', 'r') as f:
            kernel_string = f.read()

        compiler_options = ['-Xcompiler=-Wall', '--std=c++11', '-O3']

        kernel_module = SourceModule(kernel_string, options=compiler_options,
                                        arch='compute_' + cc, code='sm_' + cc,
                                        cache_dir=False, no_extern_c=True)

        self.move_nodes = kernel_module.get_function("move_nodes")

        self.community_internal_edges = kernel_module.get_function("calculate_community_internal_edges")

        self.calculate_part_cpm = kernel_module.get_function("calculate_part_cpm")

        self.classify_communities = kernel_module.get_function("classify_communities")

        self.classify_hits = kernel_module.get_function("classify_hits")

    def compute(self, col_idx, prefix_sums, cpm_resolution):

        # check input
        cpm_resolution = np.float32(cpm_resolution)
        d_col_idx = ready_input(col_idx)
        d_prefix_sums = ready_input(prefix_sums)

        # prepare GPU memory:

        # array for storing community assignment per node
        community_idx = np.arange(self.N).astype(np.int32)
        d_community_idx = allocate_and_copy(community_idx)
        d_tmp_community_idx = allocate_and_copy(community_idx)

        # array for storing community sizes
        community_sizes = np.ones(self.N).astype(np.int32)
        d_community_sizes = allocate_and_copy(community_sizes)
        d_tmp_community_sizes = allocate_and_copy(community_sizes)

        # array for storing community inter connecting edges
        community_inter = np.zeros(self.N).astype(np.int32)
        d_community_inter = allocate_and_copy(community_inter)
        d_tmp_community_inter = allocate_and_copy(community_inter)

        # array for storing partial cpm for each community
        part_cpm = np.zeros(self.N).astype(np.float32)
        d_part_cpm = allocate_and_copy(part_cpm)

        # config
        iterations_limit = 15   # limit of cpm iterations
        cpm_thresh = 0.02       # threshold ratio supports decision if another cpm iteration iss needed

        iterations = 0      # counter
        iter_cpm_score = 0  # current cpm score

        # CPM execution:
        while True:

            # calclate best community assignment
            args_move_nodes = [self.N, d_col_idx, d_prefix_sums, d_community_idx, d_community_sizes, d_tmp_community_idx, d_tmp_community_sizes, cpm_resolution]
            self.move_nodes(*args_move_nodes, block=self.threads, grid=self.grid, stream=None, shared=0)

            # memory reset
            d_tmp_community_inter = allocate_and_copy(np.zeros(self.N).astype(np.int32))
            d_part_cpm = allocate_and_copy(np.zeros(self.N).astype(np.float32))

            # calculate interconnecting edges per community
            args_community_internal_edges = [self.N, d_col_idx, d_prefix_sums, d_tmp_community_idx, d_tmp_community_inter]
            self.community_internal_edges(*args_community_internal_edges, block=self.threads, grid=self.grid, stream=None, shared=0)

            # calculate partial cpm per community
            args_calculate_part_cpm =  [self.N, d_tmp_community_inter, d_tmp_community_sizes, d_part_cpm, cpm_resolution]
            self.calculate_part_cpm(*args_calculate_part_cpm, block=self.threads, grid=self.grid, stream=None, shared=0)

            # calculate overall cpm score
            drv.memcpy_dtoh(part_cpm, d_part_cpm)
            current_cpm_score = sum(part_cpm)

            # check cpm improvement for given iteration
            if iter_cpm_score != 0:
                cpm_diff = abs((current_cpm_score - iter_cpm_score) / iter_cpm_score)
            else:
                cpm_diff = 1
            
            if cpm_diff <= cpm_thresh or iterations > iterations_limit:
                # terminate if improvement below threshold or iteration limit reached
                break
            else:
                # prepare next iteration
                iterations += 1
                iter_cpm_score = current_cpm_score

                # copy temporary results of iteration
                drv.memcpy_dtod(d_community_idx, d_tmp_community_idx, community_idx.nbytes)
                drv.memcpy_dtod(d_community_sizes, d_tmp_community_sizes, community_idx.nbytes)
                drv.memcpy_dtod(d_community_inter, d_tmp_community_inter, community_idx.nbytes)

        # classify communities
        community_class = np.zeros(self.N).astype(np.int32)
        d_community_class = allocate_and_copy(community_class)
        args_classify_communities = [self.N, d_community_inter, d_community_sizes, d_community_class]
        self.classify_communities(*args_classify_communities, block=self.threads, grid=self.grid, stream=None, shared=0)

        # classify hits 
        hit_class = np.zeros(self.N).astype(np.int32)
        d_hit_class = allocate_and_copy(hit_class)
        args_classify_hits = [self.N, d_community_idx, d_community_class, d_hit_class]
        self.classify_hits(*args_classify_hits, block=self.threads, grid=self.grid, stream=None, shared=0)

        # get classified hits 
        drv.memcpy_dtoh(hit_class, d_hit_class)
        classified_hits = sum(hit_class)

        return classified_hits