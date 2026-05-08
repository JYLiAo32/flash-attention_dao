# FlashAttention 学习记忆索引

本目录存储FlashAttention项目学习相关的记忆文档。

## 文档列表

### [00_project_overview.md](00_project_overview.md)
项目总体介绍，包含各版本对比、核心思想、性能优势和论文引用。

### [01_fa4_architecture.md](01_fa4_architecture.md)
FA4 (CuTeDSL实现) 的架构说明，包含目录结构、核心组件、关键抽象和编译特性。

### [02_learning_path.md](02_learning_path.md)
推荐的学习路径，从注意力机制基础到高级特性，分7个阶段详细说明。

### [03_testing_and_debugging.md](03_testing_and_debugging.md)
测试与调试指南，包含快速测试流程、环境变量、调试工具和常见问题排查。

## 快速开始

1. 了解项目背景: 阅读 [00_project_overview.md](00_project_overview.md)
2. 查看代码结构: 阅读 [01_fa4_architecture.md](01_fa4_architecture.md)
3. 制定学习计划: 参考 [02_learning_path.md](02_learning_path.md)
4. 运行第一个测试: 参考 [03_testing_and_debugging.md](03_testing_and_debugging.md)

## 项目代码入口

- **主要API**: [flash_attn/cute/interface.py](../flash_attn/cute/interface.py)
- **前向内核**: [flash_attn/cute/flash_fwd_sm90.py](../flash_attn/cute/flash_fwd_sm90.py)
- **反向内核**: [flash_attn/cute/flash_bwd_sm90.py](../flash_attn/cute/flash_bwd_sm90.py)
- **Softmax**: [flash_attn/cute/softmax.py](../flash_attn/cute/softmax.py)
- **Mask**: [flash_attn/cute/mask.py](../flash_attn/cute/mask.py)

## 外部资源

- GitHub: https://github.com/Dao-AILab/flash-attention
- FA3博客: https://tridao.me/blog/2024/flash3/
- CuTe文档: https://nvidia.github.io/cute/

## 记忆目录用途

此目录用于存储学习过程中的笔记、理解总结、代码片段和问题记录。可根据需要添加新的.md文件。
