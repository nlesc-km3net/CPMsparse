#include <stdio.h>
#include <inttypes.h>
#include <float.h>

#ifndef tile_size_x
  #define tile_size_x 1
#endif

#ifndef block_size_x
  #define block_size_x 512
#endif

#ifndef window_width
  #define window_width 1500
#endif

#define USE_READ_ONLY_CACHE read_only
#if USE_READ_ONLY_CACHE == 1
#define LDG(x, y) __ldg(x+y)
#elif USE_READ_ONLY_CACHE == 0
#define LDG(x, y) x[y]
#endif

#ifndef write_sums
  #define write_sums 0
#endif

#ifndef write_spm
  #define write_spm 0
#endif

#ifndef write_rows
  #define write_rows 0
#endif

#ifndef use_shared
  #define use_shared 0
#endif

extern "C" {
__global__ void pattern_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, int sliding_window_width, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct, float *__restrict__ pattern);
}

template<typename F>
__device__ void correlate_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct, F criterion, float *__restrict__ pattern);

__forceinline__ __device__ bool pattern_criterion(float x1, float y1, float z1, float ct1, float x2, float y2, float z2, float ct2, float *__restrict__ pattern);

/*
 * This is the kernel used for computing correlations in both directions using the match 3b criterion
 */
__global__ void pattern_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, int sliding_window_width, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct, float *__restrict__ pattern) {

    correlate_full(row_idx, col_idx, prefix_sums, sums, N, x, y, z, ct, pattern_criterion, pattern);

}


/*
 * This function fills shared memory will values from global memory
 *
 * The area loaded is equal to the working set of this thread block (block_size_x * tile_size_x) plus the window_width
 *
 * The threads in a thread block will load values in global memory from their global index 'i' up to block_size_x*tile_size_x+window_width
 * It is possible to modify which values from global memory are loaded by using the parameter 'offset'
 * The threads can skip the first x elements of shared memory by using a non zero value for 'start'
 * N is the total number of hits in the input, used to guard out-of-bound accesses
 *
 * first loading phase, start=0, offset=bx-window_width
 * second loading phase, start=block_size_x*tile_size_x, offset=bx
 */
__forceinline__ __device__ void fill_shared_memory(float *sh_ct, float *sh_x, float *sh_y, float* sh_z,
                                       const float *ct, const float *x, const float *y, const float *z,
                                                         int bx, int i, int start, int offset, int N) {
    #pragma unroll
    for (int k=start+i; k < block_size_x*tile_size_x+window_width; k+=block_size_x) {
        if (k+offset >= 0 && k+offset < N) {
            sh_ct[k] = LDG(ct,k+offset);
            sh_x[k] = LDG(x,k+offset);
            sh_y[k] = LDG(y,k+offset);
            sh_z[k] = LDG(z,k+offset);
        } else {
            sh_ct[k] = (float) NAN;  //this value ensures out-of-bound hits won't be correlated
            sh_x[k] = 0.0f;
            sh_y[k] = 0.0f;
            sh_z[k] = 0.0f;
        }
    }
}


/*
 * This function is responsible for looping over the iteration space of each thread
 * For each correlation to be computed it will call the criterion and either
 * store the number of correlations or the coordinates of the correlated hit.
 */
template<typename F>
__forceinline__ __device__ void correlate(int *row_idx, int *col_idx, int *sum, int *offset, int bx, int i,
                float *l_x, float *l_y, float *l_z, float *l_ct, float *sh_x, float *sh_y, float *sh_z, float *sh_ct, int col_offset, int it_offset, F criterion, float *__restrict__ pattern) {
    for (int j=it_offset; j < window_width+it_offset; j++) {

        #pragma unroll
        for (int ti=0; ti<tile_size_x; ti++) {

            bool condition = criterion(l_x[ti], l_y[ti], l_z[ti], l_ct[ti],
                    sh_x[i+ti*block_size_x+j], sh_y[i+ti*block_size_x+j],
                    sh_z[i+ti*block_size_x+j], sh_ct[i+ti*block_size_x+j], pattern);

            if (condition) {
                #if write_spm == 1
                #if write_rows
                row_idx[offset[ti]] = bx+i+ti*block_size_x; 
                #endif
                col_idx[offset[ti]] = bx+i+ti*block_size_x+j+col_offset;
                offset[ti] += 1;
                #endif
                #if write_sums == 1
                sum[ti] += 1;
                #endif
            }

        }

    }

}



/*
 * This function computes the correlated hits of hits no more than 'window_width' apart in both directions.
 * It does this using a 1-dimensional mapping of threads and thread blocks to hits in this time slice.
 *
 * This function supports the usual set of optimizations, including tiling, read-only cache. 
 * Tuning parameters supported are 'read_only' [0,1], 'tile_size_x' any low number, and 'block_size_x' multiple of 32.
 *
 * 'write_sums' can be set to [0,1] to enable the code that outputs the number of correlated hits per hit
 * This number is used to compute the offsets into the sparse matrix representation of the correlations table.
 * 
 * 'write_spm' can be set to [0,1] to enable the code that outputs the sparse matrix
 * 'write_rows' can be set to [0,1] to enable also writing the row_idx, only effective when write_spm=1
 *
 */
template<typename F>
__device__ void correlate_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct, F criterion, float *__restrict__ pattern) {

    int i = threadIdx.x;
    int bx = blockIdx.x * block_size_x * tile_size_x;

    __shared__ float sh_ct[block_size_x * tile_size_x + window_width];
    __shared__ float sh_x[block_size_x * tile_size_x + window_width];
    __shared__ float sh_y[block_size_x * tile_size_x + window_width];
    __shared__ float sh_z[block_size_x * tile_size_x + window_width];

    //the first loading phase
    fill_shared_memory(sh_ct, sh_x, sh_y, sh_z, ct, x, y, z, bx, i, 0, bx-window_width, N);

    #if write_spm == 1
    int offset[tile_size_x];
    if (bx+i==0) {
        offset[0] = 0;
    }
    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        if (bx+i+ti*block_size_x-1 >= 0 && bx+i+ti*block_size_x-1 < N) {
            offset[ti] = prefix_sums[bx+i+ti*block_size_x-1];
        }
    }
    #else
    int *offset = (int *)0;
    #endif

    __syncthreads();

    //start of the the computations phase
    float l_ct[tile_size_x];
    float l_x[tile_size_x];
    float l_y[tile_size_x];
    float l_z[tile_size_x];
    #if write_sums == 1
    int sum[tile_size_x];
    #else
    int *sum = (int *)0;
    #endif

    //keep the most often used values in registers
    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        l_ct[ti] = sh_ct[i+ti*block_size_x+window_width];
        l_x[ti] = sh_x[i+ti*block_size_x+window_width];
        l_y[ti] = sh_y[i+ti*block_size_x+window_width];
        l_z[ti] = sh_z[i+ti*block_size_x+window_width];
        #if write_sums == 1
        sum[ti] = 0;
        #endif
    }

    //first loop computes correlations with earlier hits
    correlate(row_idx, col_idx, sum, offset, bx, i, l_x, l_y, l_z, l_ct,
                    sh_x, sh_y, sh_z, sh_ct, -window_width, 0, criterion, pattern);

    //make sure all threads are done with phase-1
     __syncthreads();

    //start load phase-2
    //fill the first part of shared memory with data already in registers
    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        sh_ct[i+ti*block_size_x] = l_ct[ti];
        sh_x[i+ti*block_size_x] = l_x[ti];
        sh_y[i+ti*block_size_x] = l_y[ti];
        sh_z[i+ti*block_size_x] = l_z[ti];
    }

    //the first block_size_x*tile_size_x part has already been filled
    fill_shared_memory(sh_ct, sh_x, sh_y, sh_z, ct, x, y, z, bx, i, block_size_x*tile_size_x, bx, N);

    __syncthreads();

    //the next loop computes correlations with hits later in time
    correlate(row_idx, col_idx, sum, offset, bx, i, l_x, l_y, l_z, l_ct,
                    sh_x, sh_y, sh_z, sh_ct, 0, 1, criterion, pattern);

    #if write_sums == 1
    for (int ti=0; ti<tile_size_x; ti++) {
        if (bx+i+ti*block_size_x < N) {
            sums[bx+i+ti*block_size_x] = sum[ti];
        }
    }
    #endif
}



#ifndef ptt_threshold
    #define ptt_threshold 0.01f
#endif

#ifndef ptt_time_limit
    #define ptt_time_limit 100.0f
#endif

#ifndef ptt_time_bin_count
    #define ptt_time_bin_count 50.0f
#endif

#ifndef ptt_dist_limit
    #define ptt_dist_limit 100.0f
#endif

#ifndef ptt_dist_bin_count
    #define ptt_dist_bin_count 50.0f
#endif

#define ptt_dist_bin_size (ptt_dist_limit/ptt_dist_bin_count)
#define ptt_time_bin_size (ptt_time_limit/ptt_time_bin_count)

/*
 * This function implements the pattern criterion
 */
__forceinline__ __device__ bool pattern_criterion(float x1, float y1, float z1, float ct1, float x2, float y2, float z2, float ct2, float *__restrict__ pattern) {
    float time_diff = fabsf(ct1 - ct2);
    if (isnan(time_diff)) {
        return false;
    }

    if (time_diff >= ptt_time_limit) {
        return false;
    }

    float diffx  = x1 - x2;
    float diffy  = y1 - y2;
    float diffz  = z1 - z2;

    float dist_diff = sqrtf( (diffx * diffx) + (diffy * diffy) + (diffz * diffz) );

    if (dist_diff >= ptt_dist_limit) {
        return false;
    }

    int time_bin_idx = (int) (time_diff / ptt_time_bin_size);
    int dist_bin_idx = (int) (dist_diff / ptt_dist_bin_size);

    int bin_idx = (ptt_time_bin_count * dist_bin_idx + time_bin_idx);

    float pattern_score = pattern[bin_idx];

    if (pattern_score > ptt_threshold) {
        return true;
    } else {
        return false;
    }
}