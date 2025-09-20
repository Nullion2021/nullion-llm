# Nullion LLM

一个从零开始学习大语言模型（LLM）和多模态大模型（LLM-V）实现的项目。本项目参考了 [MiniMind](https://github.com/jingyaogong/minimind/tree/master) 项目，旨在深入理解和实践大模型的各个核心组件。

## 🎯 项目目标

- 深入学习大语言模型的架构和实现原理
- 掌握Transformer、注意力机制、位置编码等核心技术
- 为后续多模态大模型（LLM-V）的学习打下基础
- 从零开始构建完整的大模型系统

## 📁 项目结构

```
nullion-llm/
├── model/                   # 模型实现目录
│   ├── __init__.py         # 模块初始化
│   ├── nullion.py          # 主要模型实现
│   └── model_nullion.py    # 参考模型实现
├── test/                    # 测试文件目录
│   └── modeltest.py        # 模型测试文件
├── Data/                    # 数据集目录
└── README.md               # 项目说明文档
```

## 🚀 核心功能

### 模型架构特性

- **Transformer架构**: 完整实现了标准的Transformer架构
- **注意力机制**: 多头注意力（Multi-Head Attention）
- **位置编码**: RoPE（Rotary Position Embedding）旋转位置编码
- **归一化**: RMSNorm归一化层
- **激活函数**: SwiGLU激活函数
- **权重共享**: 词嵌入与输出层权重绑定

### 配置系统

- **NullionConfig**: 完整的模型配置类，支持：
  - 模型维度配置（hidden_size, intermediate_size）
  - 注意力头配置（num_attention_heads）
  - 位置编码配置（rope_theta, max_position_embeddings）
  - 归一化和Dropout配置

### 核心组件

1. **NullionConfig**: 模型配置管理
2. **RMSNorm**: 均方根归一化层
3. **Attention**: 多头注意力机制实现
4. **FeedForward**: SwiGLU前馈网络
5. **ModelBlock**: Transformer块实现
6. **NullionModel**: 模型主体架构
7. **NullionForCausalLM**: 因果语言建模封装

## 🛠️ 技术栈

- **框架**: PyTorch
- **生态**: Transformers
- **主要库**: torch, transformers, numpy
- **特性**: FlashAttention支持、RoPE实现

## 📋 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.12.0
transformers >= 4.20.0
```

### 运行测试

```bash
# 运行模型测试
python test/modeltest.py

# 测试注意力机制
python test/modeltest.py
```

### 基本使用

```python
from model.nullion import NullionConfig, NullionForCausalLM

# 创建模型配置
config = NullionConfig(
    hidden_size=512,
    num_attention_heads=8,
    num_hidden_layers=8,
    vocab_size=6400
)

# 初始化模型
model = NullionForCausalLM(config)

# 前向传播
input_ids = torch.randint(0, config.vocab_size, (1, 10))
output = model(input_ids)
```

## 🔧 模型配置

### 基础配置
```python
config = NullionConfig(
    hidden_size=512,              # 隐藏层维度
    num_attention_heads=8,       # 注意力头数
    num_hidden_layers=8,          # 隐藏层数
    vocab_size=6400,              # 词汇表大小
    max_position_embeddings=32768,  # 最大序列长度
    rope_theta=1000000.0,         # RoPE参数
    rms_norm_eps=1e-5,           # 归一化epsilon
    dropout=0.1                  # Dropout概率
)
```


## 📊 性能特性

### 注意力机制优化
- **FlashAttention**: 支持优化的注意力计算
- **RoPE**: 高效的旋转位置编码

### 模型架构优化
- **SwiGLU**: 更高效的前馈网络激活函数
- **RMSNorm**: 更稳定的归一化方法
- **权重共享**: 减少参数量，提升训练稳定性


## 🧪 测试覆盖

项目包含完整的测试套件：

- **配置测试**: 验证NullionConfig的各项参数
- **位置编码测试**: 测试RoPE预计算和应用
- **注意力机制测试**: 验证多头注意力的正确性
- **模型创建测试**: 测试基本模型的初始化
- **前向传播测试**: 验证模型的完整推理流程
- **生成测试**: 测试文本生成功能

## 📚 学习路径

### 第一阶段：基础架构
1. 理解Transformer基本架构
2. 实现多头注意力机制
3. 掌握位置编码原理

### 第二阶段：优化技术
1. 学习RoPE实现
2. 掌握FlashAttention

### 第三阶段：高级特性
1. 学习模型并行技术
2. 探索多模态扩展

## 🔄 开发计划

- [x] 基础Transformer架构实现
- [x] RoPE位置编码
- [x] 多头注意力机制
- [x] RMSNorm归一化
- [x] SwiGLU激活函数
- [ ] GQA支持
- [ ] MoE架构实现
- [ ] 模型训练脚本
- [ ] 多模态扩展（LLM-V）
- [ ] 推理优化
- [ ] 模型量化
- [ ] 分布式训练支持

## 📝 代码风格

- 使用中文注释，便于理解
- 遵循PEP 8代码规范
- 完整的函数和类文档
- 模块化设计，便于扩展

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目仅供学习和研究使用。

## 📖 参考资料

- [MiniMind项目](https://github.com/jingyaogong/minimind/tree/master)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

**学习从零开始，深入理解大模型的奥秘！** 🚀