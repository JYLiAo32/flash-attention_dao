# FA4 (CuTeDSL) 架构说明

## 目录结构

```
flash_attn/cute/           # FA4 主目录
├── interface.py           # 公共API入口
├── flash_fwd*.py          # 前向内核
├── flash_bwd*.py          # 反向内核
├── softmax.py             # 在线softmax实现
├── mask.py                # 注意力mask处理
├── block_info.py          # 分块信息
├── seqlen_info.py         # 序列长度信息
├── pipeline.py            # 流水线状态管理
├── tile_scheduler.py      # Tile调度策略
├── copy_utils.py          # 数据拷贝工具
├── utils.py               # 通用工具函数
└── cache_utils.py         # JIT编译缓存
```

## 核心组件

### 1. 公共API (interface.py)

```python
from flash_attn.cute import flash_attn_func

# 基本用法
out = flash_attn_func(q, k, v, causal=True)

# 变长序列
from flash_attn.cute import flash_attn_varlen_func
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)
```

### 2. 前向内核

| 文件 | GPU架构 | 说明 |
|------|---------|------|
| flash_fwd.py | SM80 | Ampere基础版 |
| flash_fwd_sm90.py | SM90 | Hopper (H100) |
| flash_fwd_sm100.py | SM100 | Blackwell (B200) |
| flash_fwd_mla_sm100.py | SM100 | 多头潜在注意力 |

### 3. 反向内核

| 文件 | GPU架构 | 说明 |
|------|---------|------|
| flash_bwd_sm90.py | SM90 | Hopper反向 |
| flash_bwd_sm100.py | SM100 | Blackwell反向 |
| flash_bwd_preprocess.py | 通用 | 反向预处理 |
| flash_bwd_postprocess.py | 通用 | 反向后处理 |

## 关键抽象

### Softmax (softmax.py)

在线softmax实现，核心特性：
- 流式更新row_max和row_sum
- 无需存储完整注意力矩阵
- 支持score_mod和mask_mod

### AttentionMask (mask.py)

处理各种注意力mask：
- Causal mask (因果/下三角)
- Local/Sliding window mask
- Block sparse mask
- 用户自定义mask_mod

### BlockInfo (block_info.py)

管理Tile维度信息：
- m/n block范围计算
- 因果mask的块边界
- Local attention的窗口边界

### SeqlenInfoQK (seqlen_info.py)

序列长度追踪（用于varlen）：
- cu_seqlens累积长度处理
- 序列偏移量计算

## 架构特定辅助

### hopper_helpers.py

SM90专用：
- Warp-group MMA操作
- 共享内存布局创建
- Fence/Commit/Wait同步原语

### blackwell_helpers.py

SM100专用：
- UMBA-based GEMM
- PTX优化路径
- 2CTA (双CTA) 支持

## 编译特性

1. **JIT编译**: 运行时编译为PTX/CUBIN
2. **缓存机制**: 内存LRU + 磁盘持久化
3. **环境变量**:
   - `CUTE_CUBIN_PATH`: 输出CUBIN/SASS
   - `CUTE_DSL_KEEP_PTX=1`: 保留PTX中间文件
   - `FLASH_ATTENTION_FAKE_TENSOR=1`: 假张量模式（无GPU编译）

## Tile配置

### SM90前向配置 (FwdConfig)

```python
@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int      # M方向tile大小
    n_block_size: int      # N方向tile大小
    mma_pv_is_rs: bool     # P@V使用寄存器还是共享内存
    intra_wg_overlap: bool # warp组内重叠
```

### SM90反向配置 (BwdConfig)

```python
@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int      # Q流水线阶段
    num_stages_dO: int     # dO流水线阶段
    num_stages_PdS: int    # PdS流水线阶段
    SdP_swapAB: bool       # 交换A/B布局
    dKV_swapAB: bool
    dQ_swapAB: bool
    # ...
```
