# FlashAttention 学习路径

## 第一阶段：理解注意力机制

### 前置知识
1. **Transformer架构**: 理解Self-Attention、Cross-Attention
2. **标准注意力**: O(N²)内存和计算复杂度问题
3. **Softmax性质**: 在线softmax的数学原理

### 推荐资源
- Attention Is All You Need (原论文)
- The Illustrated Transformer (Jay Alammar)

## 第二阶段：FlashAttention 核心思想

### 理解要点
1. **Tiling**: 将大矩阵分成小块
2. **Recomputation**: 反向时重算注意力矩阵
3. **IO-Aware**: 考虑内存访问代价

### 论文阅读顺序
1. FlashAttention (NeurIPS 2022)
2. FlashAttention-2 (ICLR 2024)
3. FlashAttention-3 (技术报告)

## 第三阶段：CUDA 编程基础

### 核心概念
- **Thread hierarchy**: Grid, Block, Warp, Thread
- **Memory hierarchy**: registers, shared memory, global memory (HBM)
- **Memory coalescing**: 合并访问优化
- **Warp-level primitives**: __shfl, warp reduction

### 学习资源
- NVIDIA CUDA C++ Programming Guide
- "Programming Massively Parallel Processors" (Hwu & Kirk)

## 第四阶段：CuTeDSL 入门

### CuTe 是什么
- CUDA Template Library 的DSL版本
- 用Python编写，编译为PTX
- 提供高层抽象隐藏硬件细节

### 核心抽象
```python
import cutlass.cute as cute

# Tensor Layout
layout = cute.make_layout(m, n)  # 创建M×N布局

# Tensor View
tensor = cute.make_tensor(ptr, layout)  # 创建tensor视图

# MMA (Matrix Multiply-Accumulate)
tiled_mma = cute.make_tiled_mma(...)  # 创建分块MMA
```

### 学习文件顺序
1. `utils.py` - 基础工具函数
2. `softmax.py` - 在线softmax实现
3. `flash_fwd.py` - 前向基础类
4. `flash_fwd_sm90.py` - Hopper前向实现

## 第五阶段：内核实现分析

### 前向内核 (flash_fwd_sm90.py)

```
1. 加载Q tile
2. 循环遍历K/V块 (流水线):
   - 加载K块
   - 加载V块
   - 计算S = Q @ K^T
   - 应用score_mod/mask_mod
   - 在线softmax更新
   - 累加 O = softmax(S) @ V
3. 存储O和LSE
```

### 反向内核 (flash_bwd_sm90.py)

```
1. 加载dO和O、LSE
2. 计算dS = dO @ V^T
3. 应用softmax反向
4. 计算dQ, dK, dV
```

## 第六阶段：高级特性

### 特性列表
1. **Score Mod**: 自定义注意力分数修改
2. **Mask Mod**: 自定义mask逻辑
3. **Block Sparse**: 块稀疏注意力
4. **Paged KV**: 分页KV缓存
5. **2CTA**: 双CTA协同计算

### 相关文件
- `score_mod_definitions.py` - Score mod示例
- `mask_mod_definitions.py` - Mask mod示例
- `block_sparsity.py` - 块稀疏实现
- `paged_kv.py` - 分页KV缓存

## 第七阶段：性能调优

### 调优参数
- Tile大小 (m_block_size, n_block_size)
- Pipeline阶段数
- 寄存器vs共享内存trade-off
- MMA原子布局

### 调优工具
- SASS反汇编 (`CUTE_CUBIN_PATH`)
- Nsight Compute
- PyTorch profiler

## 实践建议

### 从测试开始
```bash
# 运行基础测试
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x

# 添加打印调试
CUTE_DSL_KEEP_PTX=1 pytest tests/cute/test_flash_attn.py
```

### 修改实验
1. 修改tile大小观察性能变化
2. 添加printf调试kernel
3. 实现自定义score_mod

### 调试技巧
- FakeTensor模式快速编译
- 小规模测试数据
- 使用reference实现对比

## 常见问题

### Q: 什么是TMA?
A: Tensor Memory Accelerator，Hopper/Blackwell的硬件加速拷贝单元

### Q: 在线softmax为什么正确?
A: 利用max和sum的递推性质，流式更新

### Q: 2CTA是什么?
A: 两个CTA协同计算一个更大的tile，用于head_dim=256场景

### Q: PackGQA是什么?
A: 将多个Q头打包到单个KV头上，提高GQA效率
