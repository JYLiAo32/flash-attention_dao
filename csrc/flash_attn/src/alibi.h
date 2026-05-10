/**
 * ALiBi (Attention with Linear Biases) Flash Attention 实现
 *
 * 为什么需要 ALiBi：传统的位置编码（正弦、学习型）在处理比训练时更长的序列时
 * 外推能力有限。ALiBi 通过添加与 token 间距离成正比的惩罚来解决这个问题，
 * 使模型能够处理训练时未见过的更长序列。
 *
 * 参考文献："Train Short, Test Long: Attention with Linear Biases Enables Input Length
 * Extrapolation" (Press et al., 2021)
 *
 * 偏置公式：score(q, k) += -slope * |q_pos - k_pos|
 * 其中 slope 跨 attention head 呈几何递减
 */

#include <cmath>

#include "namespace_config.h"
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * ALiBi bias 应用器，用于 GPU kernel 中的 attention score
 *
 * 为什么存在这个类：在 Flash Attention kernel 执行期间，我们需要在 softmax 之前
 * 对 attention score 应用基于位置的 bias。这个类处理将 ALiBi bias 应用于存储在
 * shared memory 或 register 中的 tensor fragment 所需的复杂索引。
 *
 * 模板参数：
 *   Is_causal: 如果为 true，应用简化的 causal mask bias；如果为 false，应用基于
 *              row/column 位置的完整双向 ALiBi bias
 */
template <bool Is_causal>
struct Alibi {
    /**
     * 该 attention head 的 slope 参数
     * 为什么：不同的 head 使用不同的 slope（通常为 m / 2^(i+1)，i 为 head 索引）
     * 以在 attention head 之间创建多样化的位置 bias 模式
     */
    const float alibi_slope;

    /**
     * Key (K) 和 Query (Q) tensor 的最大序列长度
     * 为什么：ALiBi 计算绝对距离，所以我们需要完整的序列长度边界来正确计算
     * 任意 tile 位置的 |q_pos - k_pos|
     */
    const int max_seqlen_k, max_seqlen_q;

    /**
     * 使用指定参数构造 ALiBi 函数对象
     *
     * @param alibi_slope: 位置 bias 的 slope 乘数（每个 head 唯一）
     * @param max_seqlen_k: 最大 key 序列长度（用于距离归一化）
     * @param max_seqlen_q: 最大 query 序列长度（用于距离归一化）
     */
    __forceinline__ __device__ Alibi(const float alibi_slope, const int max_seqlen_k, const int max_seqlen_q)
        : alibi_slope(alibi_slope)
        , max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };


    /**
     * 对 tensor fragment 原地应用 ALiBi 位置 bias
     *
     * 为什么存在此函数：在 Flash Attention 中，attention score 以存储在
     * shared memory 或 register 中的 tiled fragment 形式计算。此函数遍历 fragment
     * 并根据每个元素在 attention matrix 中的全局位置应用 ALiBi bias。
     *
     * Tensor layout 说明：
     *   - 形状：(nrow=(2, MMA_M), ncol=(2, MMA_N))，其中 MMA_M x MMA_N 是
     *     Tensor Core 矩阵乘法 tile 大小（例如 16x16）
     *   - "2" 维度来自 CUDA 的 warp-level MMA 指令 layout
     *
     * @param tensor: attention score tensor fragment（原地修改）
     * @param col_idx_offset_: 此 tile 的基本 column index（key 位置）
     * @param row_idx_offset: 此 tile 的基本 row index（query 位置）
     * @param warp_row_stride: warp 内用于索引的 row stride
     */
    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor 形状为 (nrow=(2, MMA_M), ncol=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");

        // 为什么这样计算 lane_id：warp 中的每个 CUDA thread（lane）处理
        // tensor 的特定元素。lane_id % 4 给出 warp 的 8 元素宽处理中的 column group。
        const int lane_id = threadIdx.x % 32;

        // 根据 thread 的 lane 位置调整 column offset。
        // 为什么：memory layout 分布在各 lane 之间，所以每个 lane 需要知道其 offset
        // 来计算正确的全局 column index。
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

        ////////////////////////////////////////////////////////////////////////////////
        // Causal Attention 情况
        // 为什么更简单：在 causal attention 中（例如 GPT 等仅 decoder 模型），
        // 每个 query 位置只能 attend 之前的 key 位置。
        // ALiBi bias 变为：slope * k_pos（仅依赖 column，不依赖 row）。
        // 这允许我们将相同的 bias vector 添加到所有 row。
        ////////////////////////////////////////////////////////////////////////////////
        if constexpr (Is_causal) {
            // 遍历 column group（MMA_N 维度）
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                // 为什么 * 8：由于 Tensor Core layout，每个 column group 跨越 8 个元素
                const int col_idx_base = col_idx_offset + nj * 8;

                // 在 column group 内遍历
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;

                    // 将 bias 应用到 column 中的所有 row
                    // 为什么只用 col_idx：在 causal 模式下，row_idx 不影响 bias
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                    }
                }
            }
        ////////////////////////////////////////////////////////////////////////////////
        // Bidirectional Attention 情况
        // 为什么更复杂：在 bidirectional attention 中（例如 BERT、encoder 模型），
        // 每个 query 位置可以 attend 所有 key 位置。ALiBi bias 依赖于
        // 绝对距离：slope * |q_pos - k_pos|。
        //
        // 这里使用的公式：row_idx + max_seqlen_k - max_seqlen_q - col_idx
        // 通过将位置归一化到公共坐标系来计算相对距离。
        // 这处理了 Q 和 K 序列长度不同的情况（例如在 cross-attention 或 varlen batching 中）。
        ////////////////////////////////////////////////////////////////////////////////
        } else {
            // 遍历 row group（MMA_M 维度外循环）
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                // 为什么用 warp_row_stride：row 以这个 stride 分布在 warp 中，
                // 需要来计算全局 row index
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;

                // 在 row group 内遍历
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    // 为什么 * 8：在 Tensor Core layout 中每个 row group 跨越 8 个元素
                    const int row_idx = row_idx_base + i * 8;

                    // 遍历 column group（MMA_N 维度）
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;

                        // 在 column group 内遍历
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;

                            // 应用 ALiBi bias：-slope * |relative_distance|
                            // 相对距离公式考虑了 Q 和 K 之间的序列长度差异。
                            tensor(make_coord(i, mi), make_coord(j, nj)) -=
                                alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                        }
                    }
                }
            }
        }
    }

};

}  // namespace FLASH_NAMESPACE
