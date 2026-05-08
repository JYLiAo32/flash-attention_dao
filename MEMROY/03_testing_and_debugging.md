# 测试与调试指南

## 快速测试流程

### 环境准备
```bash
# 安装开发版本
pip install -e "flash_attn/cute[dev]"

# 检查GPU
nvidia-smi
```

### 两阶段测试（推荐）

编译耗时占主导，使用两阶段分离：

```bash
# 阶段1: 并行编译所有kernel（FakeTensor模式，无需GPU）
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py

# 阶段2: 运行测试（使用缓存编译好的kernel）
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

### 单个测试
```bash
# 测试特定功能
pytest tests/cute/test_flash_attn.py::test_flash_attn_output -x

# varlen测试
pytest tests/cute/test_flash_attn_varlen.py -x

# mask_mod测试
pytest tests/cute/test_mask_mod.py -x

# score_mod测试
pytest tests/cute/test_score_mod.py -x
```

## 测试文件说明

| 测试文件 | 内容 |
|----------|------|
| test_flash_attn.py | 主要功能测试 |
| test_flash_attn_varlen.py | 变长序列测试 |
| test_flash_attn_combine.py | SplitKV合并测试 |
| test_mask_mod.py | Mask modifier测试 |
| test_score_mod.py | Score modifier测试 |
| test_block_sparsity.py | 块稀疏测试 |
| test_flash_attn_race_condition.py | 竞态条件测试 |

## 调试工具

### 环境变量

| 变量 | 作用 |
|------|------|
| `CUTE_CUBIN_PATH=/tmp/cubin` | 输出编译的CUBIN文件 |
| `CUTE_DSL_KEEP_PTX=1` | 保留PTX中间代码 |
| `CUTE_DSL_LINEINFO=1` | 添加行号信息（用于sanitizer） |
| `FLASH_ATTENTION_FAKE_TENSOR=1` | FakeTensor模式编译 |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` | 启用磁盘缓存 |

### printf 调试

```python
import cutlass.cute as cute

# 在kernel中添加打印
@cute.jit
def my_kernel(...):
    for i in cute.range(...):
        if tidx % 32 == 0:  # 每个warp只打印一次
            cute.printf(f"Value: {value}\n")
```

### Compute Sanitizer

```bash
# 检测竞态条件
compute-sanitizer --tool=racecheck pytest tests/cute/test_flash_attn.py

# 注意：TMA可能产生误报，参见 AI/RACECHECK_TMA_HAZARD.md
```

### SASS 分析

```bash
# 生成SASS
CUTE_CUBIN_PATH=/tmp/cubin pytest tests/cute/test_flash_attn.py

# 反汇编
nvdisasm /tmp/cubin/*.cubin > output.sass
```

## 常见问题排查

### OOM错误
```bash
# 选择空闲GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/cute/test_flash_attn.py
```

### 编译失败
1. 检查CUDA版本 >= 12.3
2. 检查cutlass-dsl版本: `pip show nvidia-cutlass-dsl`
3. 清除缓存: `rm -rf /tmp/${USER}/flash_attention_cute_dsl_cache/`

### 数值不正确
1. 检查dtype (fp16/bf16)
2. 检查head_dim是否对齐
3. 使用reference实现对比

## 调试文档

项目包含多个调试指南文档：

| 文档 | 内容 |
|------|------|
| AI/DEBUG_2CTA.md | 2CTA内核调试 |
| AI/RACECHECK_TMA_HAZARD.md | TMA竞态检测误报 |
| AI/CLC_TRACE_DEBUG.md | CLC调度可视化 |
| AI/SM90_BLOCK_SIZE_TUNING.md | SM90块大小调优 |
