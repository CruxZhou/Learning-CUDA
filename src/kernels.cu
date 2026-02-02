#include <vector>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "../tester/utils.h"

// ---------宏定义部分---------

// 错误检查宏
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr<<"CUDA error at"<<__FILE__<<":"<<__LINE__        \
                 <<"-"<<cudaGetErrorString(err)<<"\n";             \
        exit(1);                                                   \
    } \
} while(0)
#endif

// ---------通用函数实现部分---------

/**
 * @brief 归约求和函数。
 * 利用 __shfl_down_sync 指令完成整个 warp 的 sum(val)。
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val){
#pragma unroll
  for(int offset = 16; offset > 0; offset >>= 1){
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---------核函数与调用部分---------

template <typename T>
__global__ void trace_warp_shuffle_kernel(const T* __restrict__ input, T* __restrict__ output, const size_t diag, const size_t cols, const size_t step) {
  
  extern __shared__ unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);

  const unsigned int tid = threadIdx.x;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + tid;

  T sum = (T)0;
#pragma unroll
  for (size_t i = idx; i < diag; i += step) { // gird-stride loop
    sum += input[i * cols + i];
  }

  T warp_sum = warp_reduce_sum(sum);
  if(tid % 32 == 0){
    smem[tid / 32] = warp_sum;
  }
  __syncthreads();

  int num_warps = (blockDim.x + 31) / 32;
  if(tid < 32){
    T block_sum = (tid < num_warps) ? smem[tid] : (T)0;
    block_sum = warp_reduce_sum(block_sum);
    if(tid == 0){
      atomicAdd(output, block_sum);
    }
  }
}

template <typename T>
void trace_calculate(const T* input, T* output, size_t rows, size_t cols, const dim3& block_dim, const dim3& grid_dim) {
  const size_t smem_size = (size_t)block_dim.x * sizeof(T);
  const size_t step = block_dim.x * grid_dim.x;

  const size_t diag = (rows < cols) ? rows : cols;
  trace_warp_shuffle_kernel<T><<<grid_dim, block_dim, smem_size>>>(input, output, diag, cols, step);
}


/**
 * @brief 基于 Cooperative Groups 的全网格协同 trace 计算核函数。
 *
 * 该核函数计算矩阵迹，并通过两级归约避免原子操作：
 *   1. **块内归约**：每个线程块使用 grid-stride loop 分配对角线索引，
 *      通过 warp shuffle 和 shared memory 归约得到块局部和，存入 partial[blockIdx.x]；
 *   2. **全局归约**：由 blockIdx.x == 0 的线程块读取所有 partial[i]，
 *      再次进行块内归约，最终将总和写入 partial[0]。
 */
template <typename T>
__global__ void trace_cooperative_reduce_kernel(const T* __restrict__ input, T* __restrict__ partial, // 长度至少 gridDim.x，最终结果写 partial[0]
                                                const size_t diag, const size_t cols, const size_t step) {
    extern __shared__ unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();

    const unsigned int tid = threadIdx.x;
    const size_t idx  = blockIdx.x * blockDim.x + tid;

    T sum = (T)0;
#pragma unroll
    for (size_t i = idx; i < diag; i += step) { // gird-stride loop
        sum += input[i * cols + i];
    }

    T warp_sum = warp_reduce_sum(sum); // warp 内归约

    if ((tid & 31) == 0) {
        smem[tid >> 5] = warp_sum;
    }
    block.sync();

    const int num_warps = (blockDim.x + 31) / 32;
    if (tid < 32) {
        T block_sum = (tid < (unsigned)num_warps) ? smem[tid] : (T)0;
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            partial[blockIdx.x] = block_sum;
        }
    }
    grid.sync();

    if (blockIdx.x == 0) {
        T final_sum = (T)0;
        for (size_t i = tid; i < gridDim.x; i += blockDim.x) {
            final_sum += partial[i];
        }

        T warp_val = warp_reduce_sum(final_sum);
        if ((tid & 31) == 0) {
            smem[tid >> 5] = warp_val;
        }
        block.sync();

        if (tid < 32) {
            T total = (tid < (unsigned)num_warps) ? smem[tid] : (T)0;
            total = warp_reduce_sum(total);
            if (tid == 0) {
                partial[0] = total;
            }
        }
    }
}

template <typename T>
void trace_cooperative_kernel_launch(const T* input, T* partial, size_t rows, size_t cols, const dim3& block_dim, const dim3& grid_dim) {
  const size_t smem_size = ((block_dim.x + 31) / 32) * sizeof(T);
  const size_t step = block_dim.x * grid_dim.x;
  const size_t diag = (rows < cols) ? rows : cols;

  void* args[] = {
    (void*)&input,    
    (void*)&partial,  
    (void*)&diag,       
    (void*)&cols,       
    (void*)&step        
  };

  CUDA_CHECK(cudaLaunchCooperativeKernel(
    (void*)trace_cooperative_reduce_kernel<T>,
    grid_dim, block_dim, args, smem_size, nullptr));

}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  const size_t n_elem = rows * cols;
  if (n_elem == 0) return (T)0;

  // ---------prop查询---------
  int device_id = 0;
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

  const int max_threads_per_block = device_prop.maxThreadsPerBlock;     
  const int multi_processor_count = device_prop.multiProcessorCount;    
  const int warp_size = device_prop.warpSize;                           

  // ---------并行参数配置---------
  int threads = max_threads_per_block;
  // 当设置threads为自定义值时下面两行起作用
  if (threads % warp_size != 0) threads = (threads / warp_size) * warp_size;
  if (threads < warp_size) threads = warp_size;
  if (threads > 1024) threads = 1024;

  const size_t diag = (rows < cols) ? rows : cols;
  int blocks = (int)((diag + (size_t)threads - 1) / (size_t)threads);
  if (blocks < 1) blocks = 1;

  // 防止atomicAdd次数太多，限制最大blocks数量
  // 不进行atomicAdd时可以不考虑限制
  const int max_reasonable_blocks = multi_processor_count * 32;
  if (blocks > max_reasonable_blocks) blocks = max_reasonable_blocks;

  dim3 block_dim(threads);
  dim3 grid_dim(blocks);

  // ---------Host端数据处理----------
  T* d_input = nullptr;
  T* d_partial = nullptr;
  //T* d_out   = nullptr;

  const size_t size_bytes = n_elem * sizeof(T);
  CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
  CUDA_CHECK(cudaMalloc(&d_partial, grid_dim.x * sizeof(T)));
  //CUDA_CHECK(cudaMalloc(&d_out, sizeof(T)));

  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_partial, 0, grid_dim.x * sizeof(T)));
  //CUDA_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  // ---------核函数处理---------
  trace_cooperative_kernel_launch<T>(d_input, d_partial, rows, cols, block_dim, grid_dim);
  //trace_calculate<T>(d_input, d_partial, rows, cols, block_dim, grid_dim);
  //CUDA_CHECK(cudaGetLastError());
  //CUDA_CHECK(cudaDeviceSynchronize());

  // ---------GPU结果拷回CPU---------
  T h_out = (T)0;
  CUDA_CHECK(cudaMemcpy(&h_out, d_partial, sizeof(T), cudaMemcpyDeviceToHost)); //结果在d_partial[0]

  // ---------内存释放---------
  if(d_input) CUDA_CHECK(cudaFree(d_input));
  if(d_partial) CUDA_CHECK(cudaFree(d_partial));
  d_input = d_partial = nullptr;

  return h_out;
}


/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    return;
}



// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);