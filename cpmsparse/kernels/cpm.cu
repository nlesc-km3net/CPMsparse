#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <cmath>

extern "C" {
    __global__ void move_nodes(int n_tot, int *d_col_idx, int *d_prefix_sums, int *d_community_idx, int *d_community_sizes, int *d_tmp_community_idx, int *d_tmp_community_sizes, float resolution);
    __global__ void calculate_community_internal_edges(int n_tot, int *d_col_idx, int *d_prefix_sums, int *d_tmp_community_idx, int *d_tmp_community_inter);
    __global__ void calculate_part_cpm(int n_tot, int *d_tmp_community_inter, int *d_tmp_community_sizes, float *d_part_cpm, float resolution);
    __global__ void classify_communities(int n_tot, int *d_community_inter, int *d_community_sizes, int *d_community_class);
    __global__ void classify_hits(int n_tot, int *d_community_idx, int *d_community_class, int *d_hit_class);
}

__global__ void move_nodes(int n_tot, int *d_col_idx, int *d_prefix_sums, int *d_community_idx, int *d_community_sizes, int *d_tmp_community_idx, int *d_tmp_community_sizes, float resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {

        //define neighbour range
        int start = 0;
        if (i>0) {
            start = d_prefix_sums[i-1];
        }
        int end = d_prefix_sums[i];

        //CPM
        int current_comm = d_community_idx[i];
        int new_comm = current_comm;
        int n_i = 1;

        bool local_set = false;
        float local_q = 0;
        float max_q = 0;

        //iterate over neighbours of i 
        for(int j = start; j < end; j++) {

            int col = d_col_idx[j];

            //get community of neighbour
            int col_comm = d_community_idx[col];

            int n_comm = d_community_sizes[col_comm];

            int k_i_comm = 0;   //sum of weights of edges joining i with community
            //search for other neighbors from this community
            for(int n = start; n < end; n++) {
                int col_n = d_col_idx[n];
                //check if its from the same community
                if(d_community_idx[col_n] != col_comm) {
                    continue;
                }

                k_i_comm++;
            }

            local_q = - ( 2*k_i_comm - (2 * n_i * resolution * n_comm) );

            if(!local_set || local_q <= max_q) {
                if(local_set && local_q == max_q && new_comm < col_comm) {
                    //do nothing
                } else {

                    local_set = true;
                    new_comm = col_comm;
                    max_q = local_q;
                }
            }
        }

        d_tmp_community_idx[i] = new_comm;   
        atomicAdd(&d_tmp_community_sizes[new_comm], 1); 
        atomicSub(&d_tmp_community_sizes[current_comm], 1); 
    }
}

__global__ void calculate_community_internal_edges(int n_tot, int *d_col_idx, int *d_prefix_sums, int *d_tmp_community_idx, int *d_tmp_community_inter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        int inter_count = 0;
       
        //define neighbour range
        int start = 0;
        if (i>0) {
            start = d_prefix_sums[i-1];
        }
        int end = d_prefix_sums[i];
        int current_comm = d_tmp_community_idx[i];

        //iterate over neighbours of i 
        for (int j = start; j < end; j++) {
            int col = d_col_idx[j];
            if (d_tmp_community_idx[col] == current_comm) {
                inter_count++;
            }
        }

        atomicAdd(&d_tmp_community_inter[current_comm], inter_count);
    }
}

__global__ void calculate_part_cpm(int n_tot, int *d_tmp_community_inter, int *d_tmp_community_sizes, float *d_part_cpm, float resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        float ec = (float) (d_tmp_community_inter[i] / 2.0);
        float nc = (float) (d_tmp_community_sizes[i]);
        d_part_cpm[i] = - ( ec - (resolution * nc * nc) );
    }
}


#define class_size_limit 20.0
#define class_dens_limit 0.4


__global__ void classify_communities(int n_tot, int *d_community_inter, int *d_community_sizes, int *d_community_class) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        float ec = (float) (d_community_inter[i] / 2.0);
        float nc = (float) (d_community_sizes[i]);
        float density = ec / ((nc*(nc-1.0)) / 2.0);
        if (isnan(density)) {
            density = 0.0;
        }

        int comm_class = 0;
        if (nc > class_size_limit && density > class_dens_limit) {
            comm_class = 1;
        }
        d_community_class[i] = comm_class;
    }
}

__global__ void classify_hits(int n_tot, int *d_community_idx, int *d_community_class, int *d_hit_class) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        int comm_idx = d_community_idx[i];
        d_hit_class[i] = d_community_class[comm_idx];
    }
}