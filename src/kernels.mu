#include <vector>
#include <musa_fp16.h>

#include "../tester/utils.h"

// ---------------- Error Check ----------------
#ifndef MUSA_CHECK
#define MUSA_CHECK(call)                                                     \
  do {                                                                       \
    musaError_t err = (call);                                                \
    if (err != musaSuccess) {                                                \
      std::cerr << "error at " << __FILE__ << ":" << __LINE__      \
                << " - " << musaGetErrorString(err) << std::endl;            \
      std::exit(1);                                                         \
    }                                                                        \
  } while (0)
#endif

// ---------------- Warp Reduce Sum ----------------
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

// ---------------- Kernel ----------------
template <typename T>
__global__ void trace_warp_reduce_kernel(const T* __restrict__ input,
                                        T* __restrict__ out,
                                        size_t diag, size_t cols, size_t step) {
  extern __shared__ unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);

  const unsigned int tid = threadIdx.x;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + tid;

  // grid-stride sum of diagonal
  T sum = (T)0;
  for (size_t i = idx; i < diag; i += step) {
    sum += input[i * cols + i];
  }

  // warp-level reduce
  T wsum = warp_reduce_sum(sum);

  // write warp sums to shared
  if ((tid & 31) == 0) {
    smem[tid >> 5] = wsum; // lane0 of each warp
  }
  __syncthreads();

  // first warp reduces all warp sums
  const int num_warps = (blockDim.x + 31) >> 5;
  if (tid < 32) {
    T bsum = (tid < (unsigned)num_warps) ? smem[tid] : (T)0;
    bsum = warp_reduce_sum(bsum);
    if (tid == 0) {
      atomicAdd(out, bsum);
    }
  }
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

  const size_t diag = (rows < cols) ? rows : cols;

  // ---- device props ----
  int device_id = 0;
  musaDeviceProp prop{};
  MUSA_CHECK(musaGetDeviceProperties(&prop, device_id));

  const int warp_size = prop.warpSize;
  int threads = prop.maxThreadsPerBlock;
  if (threads > 1024) threads = 1024;
  if (threads % warp_size != 0) threads = (threads / warp_size) * warp_size;
  if (threads < warp_size) threads = warp_size;

  int blocks = (int)((diag + (size_t)threads - 1) / (size_t)threads);
  if (blocks < 1) blocks = 1;

  // limit blocks to avoid too many atomicAdd calls
  const int max_reasonable_blocks = prop.multiProcessorCount * 32;
  if (blocks > max_reasonable_blocks) blocks = max_reasonable_blocks;

  dim3 block_dim(threads);
  dim3 grid_dim(blocks);

  const size_t step = (size_t)block_dim.x * (size_t)grid_dim.x;
  const size_t num_warps = ((size_t)threads + 31) / 32;
  const size_t smem_bytes = num_warps * sizeof(T);

  // ---- alloc & copy ----
  T* d_input = nullptr;
  T* d_out = nullptr;

  MUSA_CHECK(musaMalloc(&d_input, n_elem * sizeof(T)));
  MUSA_CHECK(musaMalloc(&d_out, sizeof(T)));

  MUSA_CHECK(musaMemcpy(d_input, h_input.data(), n_elem * sizeof(T),
                        musaMemcpyHostToDevice));
  MUSA_CHECK(musaMemset(d_out, 0, sizeof(T)));

  // ---- launch ----
  trace_warp_reduce_kernel<T><<<grid_dim, block_dim, smem_bytes>>>(
      d_input, d_out, diag, cols, step);
  MUSA_CHECK(musaGetLastError());
  MUSA_CHECK(musaDeviceSynchronize());

  // ---- copy back ----
  T h_out = (T)0;
  MUSA_CHECK(musaMemcpy(&h_out, d_out, sizeof(T), musaMemcpyDeviceToHost));

  // ---- free ----
  MUSA_CHECK(musaFree(d_input));
  MUSA_CHECK(musaFree(d_out));

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
