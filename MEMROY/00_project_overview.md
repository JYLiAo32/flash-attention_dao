# FlashAttention 项目概述

## 项目简介

FlashAttention 是一个快速、内存高效的精确注意力算法实现，用于 Transformer 模型。本项目包含多代实现：

| 版本 | GPU架构 | 实现方式 | 状态 |
|------|---------|----------|------|
| FA2 | SM80 (Ampere) | CUDA C++ | 稳定版 |
| FA3 | SM90 (Hopper H100) | CUDA C++ | Beta |
| FA4 | SM90/SM100 (Hopper/Blackwell) | CuTeDSL (Python) | 活跃开发 |

## 核心思想

1. **IO感知**: 优化内存访问模式，减少HBM读写次数
2. **分块计算**: 将注意力矩阵分块处理，避免O(N²)内存占用
3. **在线Softmax**: 流式计算softmax，无需存储完整注意力矩阵
4. **内核融合**: 将多个操作融合到单个CUDA内核中

## 性能优势

- **内存**: 线性复杂度 vs 标准注意力的二次复杂度
- **速度**: 比PyTorch标准注意力快2-3倍
- **精度**: 精确注意力（非近似）

## 论文引用

1. FlashAttention (NeurIPS 2022): https://arxiv.org/abs/2205.14135
2. FlashAttention-2 (ICLR 2024): https://tridao.me/publications/flash2/flash2.pdf
3. FlashAttention-3: https://tridao.me/publications/flash3/flash3.pdf
