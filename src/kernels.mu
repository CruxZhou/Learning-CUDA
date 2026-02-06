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


// ---------通用函数实现部分---------

// ********类型转换函数********
// 把任意 T 转成 float
template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ float to_float<half>(half x) { return __half2float(x); }

// 把 float 转回 T
template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ half from_float<half>(float x) { return __float2half(x); }

// ********归约函数********
__device__ __forceinline__ float warp_reduce_sum_float(float v) { // 归约求和
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max_float(float v) { // 归约最大值
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

// ---------核函数与调用部分---------


/**
* @brief Flash Attention 核函数实现：
* 优化思路：
* 1）分块加载 Q/K/V，利用共享内存和 warp shuffle 实现高效的online softmax 计算
* 2）一个 block 处理 Qm 个 query token，实现复用共享内存中加载的 K/V tile，减少全局内存访问次数
* 3）针对 half 和 float 输入分别优化计算S的方式，保证精度和性能
*/
template <typename T, int KV_tilesize, int Bl, int Qm>
__global__ void flash_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    T* __restrict__ O)
{
    // --------- 预处理 ----------
    const float softmax_scale = 1.0f / sqrtf((float)head_dim); // 适配过程发现host端无法计算

    // 每个block处理Qm个query token
    const int t_block = blockIdx.x;  
    const int q_head = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;

    const int t0 = t_block * Qm; // 起始位置
    if (t0 >= target_seq_len) return;

    // GQA
    const int group_size = query_heads / kv_heads;
    const int kv_head_id = q_head / group_size;

    // strides
    const size_t Q_b_stride  = (size_t)target_seq_len * query_heads * head_dim;
    const size_t Q_t_stride  = (size_t)query_heads * head_dim;
    const size_t Q_h_stride  = (size_t)head_dim;

    const size_t KV_b_stride = (size_t)src_seq_len * kv_heads * head_dim;
    const size_t KV_s_stride = (size_t)kv_heads * head_dim;
    const size_t KV_h_stride = (size_t)head_dim;

    // offset
    const size_t kv_offset = (size_t)batch_idx * KV_b_stride + (size_t)kv_head_id * KV_h_stride;
    
    // 线程/warp 划分
    const int lane = tid & 31; 
    const int wid  = tid >> 5;
    const int num_warps = (Bl + 31) / 32; //blockDim.x

    // --------- shared memory切分 ----------
    extern __shared__ float smem[];
    float* smem_Qm   = smem; 
    float* smem_K    = smem_Qm + (size_t)Qm * head_dim; 
    float* smem_V    = smem_K  + (size_t)KV_tilesize * (head_dim + 1);
    float* smem_S     = smem_V  + (size_t)KV_tilesize * (head_dim + 1);
    float* smem_warpbuf   = smem_S   + KV_tilesize;
    float* smem_O = smem_warpbuf + 32;
    float* smem_row_m_s = smem_O + (size_t)Qm * head_dim;
    float* smem_row_l_s = smem_row_m_s + Qm;
    // smem_Qm [Qm, D] 存当前block负责的Qm个query token的向量
    // smem_K    [KV_tilesize, D+1] 存本次KV tile的K。padding了1列以避免bank conflict
    // smem_V    [KV_tilesize, D+1] 存本次KV tile的V。padding了1列以避免bank conflict
    // smem_S     [KV_tilesize] 存本次tile的score
    // smem_warpbuf   [32] 归约临时缓冲 一个block最多32个warp
    // smem_O [Qm, D] softmax过程没有除以分母的累加结果
    // smem_row_m_s [Qm] 当前query行的max
    // smem_row_l_s [Qm] 当前query行的softmax分母


    // --------- 加载 Q tile ----------

    // 加载Qm个query向量到smem_Qmi
    for (int q = 0; q < Qm; q++) {
        const int tq = t0 + q; // t0~t0+Qm-1
        const bool valid_q = (tq < target_seq_len);

        // 加载Q向量
        if (valid_q) {
            const size_t q_offset = (size_t)batch_idx * Q_b_stride + (size_t)tq * Q_t_stride + (size_t)q_head * Q_h_stride;
            for (int x = tid; x < head_dim; x += Bl) { 
            // 并行加载 head_dim 维向量
            // 每个block有Bl个线程
                smem_Qm[(size_t)q * head_dim + x] = to_float<T>(Q[q_offset + x]);
                smem_O[(size_t)q * head_dim + x] = 0.0f;
            }
        } else {
            for (int x = tid; x < head_dim; x += Bl) {
                smem_Qm[(size_t)q * head_dim + x] = 0.0f;
                smem_O[(size_t)q * head_dim + x] = 0.0f;
            }
        }
        
        if (tid == 0) {
            smem_row_m_s[q] = -1e38f;
            smem_row_l_s[q] = 0.0f;
        }
        __syncthreads();
    }

    // 确定 block 遍历 KV 的全局上限
    const int t_last = min(t0 + Qm - 1, target_seq_len - 1);
    const int s_end_block = is_causal ? min(t_last + 1, src_seq_len) : src_seq_len;

    // --------- 遍历 KV tile ----------
    for (int j = 0; j < s_end_block; j += KV_tilesize) {
        const int s_block_end = min(j + KV_tilesize, s_end_block);
        const int KV_valid_block = s_block_end - j;

        // 把Kj和Vj tile加载到共享内存
        const int kv_elems = KV_valid_block * head_dim;
        for (int i = tid; i < kv_elems; i += Bl) {
            const int y = i / head_dim; // y 表示 tile 内第几行
            const int x = i - y * head_dim; // x 表示该行向量中的第几列
            const int s = j + y; // s 是全局 src 序列中的 token 下标

            const size_t kv_idx = kv_offset + (size_t)s * KV_s_stride + x; // K/V [batch, s, kv_head, x]
            smem_K[(size_t)y * (head_dim + 1) + x] = to_float<T>(K[kv_idx]);
            smem_V[(size_t)y * (head_dim + 1) + x] = to_float<T>(V[kv_idx]);
        }
        __syncthreads();


        // 当前KV tile对本blcok中的Qm个query依次做一次FA更新
        for (int q = 0; q < Qm; ++q) { 
            const int tq = t0 + q;
            if (tq >= target_seq_len) continue;
            const int s_end_q = is_causal ? min(tq + 1, src_seq_len) : src_seq_len;
            const int KV_valid_q = max(0, min(j + KV_tilesize, s_end_q) - j);
            if (KV_valid_q <= 0) continue;

            float row_m = smem_row_m_s[q];
            float row_l = smem_row_l_s[q];

            // 计算 S = Qm . Kj^T 
            if (std::is_same<T, float>::value){
                for (int y = tid; y < KV_valid_q; y += Bl) { 
                    // 每个线程负责若干个 y（tile 的行），计算该 query 对这些 key 的 score
                    // float类型的test对精度要求更高，用下面的算法过不了精度
                    float sum = 0.0f;
                    #pragma unroll 
                    for (int x = 0; x < head_dim; ++x) {
                        
                        sum += smem_Qm[(size_t)q * head_dim + x] *  
                            smem_K[(size_t)y * (head_dim + 1) + x];
                    }
                    smem_S[y] = sum * softmax_scale;
                }
                __syncthreads();
            }else if (std::is_same<T, half>::value){ 
                // 如果输入是half，用 warp 更高效
                for (int y = wid; y < KV_valid_q; y += num_warps) { 
                    // 以 warp 为单位分配
                    float sum = 0.0f;

                    // lane-strided dot over head_dim
                    for (int x = lane; x < head_dim; x += 32) {
                        sum += smem_Qm[(size_t)q * head_dim + x] * 
                            smem_K[(size_t)y * (head_dim + 1) + x];
                    }
                    sum = warp_reduce_sum_float(sum);// warp内归约
                    if (lane == 0) {
                        smem_S[y] = sum * softmax_scale;
                    }
                }
                __syncthreads();
            }

            float local_max = -1e38f;
            for (int y = tid; y < KV_valid_q; y += Bl) { // block-strided over y
                local_max = fmaxf(local_max, smem_S[y]);
            }

            float wmax = warp_reduce_max_float(local_max);// 先在 warp 内求最大值
            if (lane == 0) smem_warpbuf[wid] = wmax;// 每个 warp 的 lane0 把本 warp 的最大值写到 smem_warpbuf[wid]
            __syncthreads();

            if (wid == 0) {
                float v = (lane < num_warps) ? smem_warpbuf[lane] : -1e38f;
                float bmax = warp_reduce_max_float(v); //warp0再做一次warp_max，得到全 block 的最大值
                if (lane == 0) smem_warpbuf[0] = bmax;
            }
            __syncthreads();
            const float block_max = smem_warpbuf[0];

            // online softmax 部分
            const float row_m_new  = fmaxf(row_m, block_max);
            const float scale_prev = (isinf(row_m) && row_m < 0.0f) ? 0.0f : __expf(row_m - row_m_new);
            const float scale_curr = __expf(block_max - row_m_new);

            float local_sum = 0.0f;
            if (lane == 0) {
                for (int y = wid; y < KV_valid_q; y += num_warps) { 
                    float p = __expf(smem_S[y] - block_max);
                    smem_S[y] = p;
                    local_sum += p;
                } 
            }

            float wsum = warp_reduce_sum_float((lane == 0) ? local_sum : 0.0f);
            if (lane == 0) smem_warpbuf[wid] = wsum;
            __syncthreads();

            if (wid == 0) {
                float v = (lane < num_warps) ? smem_warpbuf[lane] : 0.0f;
                float bsum = warp_reduce_sum_float(v);
                if (lane == 0) smem_warpbuf[0] = bsum;
            }
            __syncthreads();
            const float block_sum = smem_warpbuf[0];

            const float row_l_new = row_l * scale_prev + block_sum * scale_curr;   
            // online softmax 更新分母：l_new = l*exp(m-m_new) + sum_tile*exp(m_tile-m_new)

            // 迭代计算 O
            for (int x = wid; x < head_dim; x += num_warps) {
                float v = 0.0f;
                if (lane < KV_valid_q) { 
                    v = smem_S[lane] * smem_V[(size_t)lane * (head_dim + 1) + x]; 
                }
                float pv = warp_reduce_sum_float(v);
                if (lane == 0) {
                    float old = smem_O[(size_t)q * head_dim + x]; 
                    smem_O[(size_t)q * head_dim + x] = old * scale_prev + pv * scale_curr;
                }
            }
            __syncthreads();


            if (tid == 0) {
                smem_row_m_s[q] = row_m_new;
                smem_row_l_s[q] = row_l_new;
            } 
            __syncthreads();
        } // q loop
    } // j loop


    // 最后把计算结果写回全局内存
    for (int q = 0; q < Qm; ++q) {
        const int tq = t0 + q;
        if (tq >= target_seq_len) continue;

        const float inv_l = (smem_row_l_s[q] > 0.0f) ? (1.0f / smem_row_l_s[q]) : 0.0f;

        const size_t q_offset = (size_t)batch_idx * Q_b_stride + (size_t)tq * Q_t_stride + (size_t)q_head * Q_h_stride;

        for (int x = tid; x < head_dim; x += Bl) {
            float out = smem_O[(size_t)q * head_dim + x] * inv_l;
            O[q_offset + x] = from_float<T>(out);
        }
    }
}

// ---------核函数调用接口部分---------
template <typename T>
void flash_attention_kernel_launch(
    const T* d_q, const T* d_k, const T* d_v,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal,
    T* d_o)
{
    
    
      const int Bl = 256;
      const int KV_tilesize = 32;
      const int Qm = 16;

      size_t smem_floats =
            (size_t)Qm * head_dim +
            (size_t)KV_tilesize * (head_dim + 1) +
            (size_t)KV_tilesize * (head_dim + 1) +
            (size_t)KV_tilesize +
            (size_t)32 +
            (size_t)Qm * head_dim +
            (size_t)Qm +
            (size_t)Qm;

      size_t smem_bytes = smem_floats * sizeof(float);

      size_t grid_x = (target_seq_len + Qm - 1) / Qm;
      dim3 grid_dim(grid_x, query_heads, batch_size);
      dim3 block_dim(Bl);

      flash_attention_kernel<T, KV_tilesize, Bl, Qm>
            <<<grid_dim, block_dim, smem_bytes>>>(
                d_q, d_k, d_v,
                batch_size, target_seq_len, src_seq_len,
                query_heads, kv_heads, head_dim,
                is_causal, d_o);
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
    // 计算各个tensor的元素数量
    size_t q_elems = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t k_elems = (size_t)batch_size * src_seq_len    * kv_heads    * head_dim;
    size_t v_elems = (size_t)batch_size * src_seq_len    * kv_heads    * head_dim;
    

    // ---------GPU内存分配和数据拷贝---------
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
   
    MUSA_CHECK(musaMalloc(&d_q, q_elems * sizeof(T)));
    MUSA_CHECK(musaMalloc(&d_k, k_elems * sizeof(T)));
    MUSA_CHECK(musaMalloc(&d_v, v_elems * sizeof(T)));
    MUSA_CHECK(musaMalloc(&d_o, q_elems * sizeof(T)));

    MUSA_CHECK(musaMemcpy(d_q, h_q.data(), q_elems * sizeof(T), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_k, h_k.data(), k_elems * sizeof(T), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_v, h_v.data(), v_elems * sizeof(T), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemset(d_o, 0, q_elems * sizeof(T)));

    // ---------并行参数配置和核函数处理---------
    flash_attention_kernel_launch<T>(
        d_q, d_k, d_v,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal,
        d_o);

    // ---------GPU结果拷回CPU---------
    MUSA_CHECK(musaMemcpyAsync(h_o.data(), d_o, q_elems * sizeof(T), musaMemcpyDeviceToHost));

    // ---------内存释放---------
    MUSA_CHECK(musaFree(d_q));
    MUSA_CHECK(musaFree(d_k));
    MUSA_CHECK(musaFree(d_v));
    MUSA_CHECK(musaFree(d_o));
    d_q = d_k = d_v = d_o = nullptr;  
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
