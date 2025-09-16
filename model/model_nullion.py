# Nullion Config 大模型参数

from transformers import PretrainedConfig


class NullionConfig(PretrainedConfig):
    model_type = "nullion"

    def __init__(
            self,
            dropout: float = 0.0,  # Dropout概率
            bos_token_id: int = 0,  # 句首标记的ID(BOS)
            eos_token_id: int = 2,  # 句尾标记的ID(EOS)
            hidden_act: str = "silu",  # 隐藏层使用的激活函数
            intermediate_size: int = None,  # Transformer中间层维度大小, None表示根据模型维度自动计算
            max_position_embeddings: int = 32768,  # 模型支持最大序列长度, 超过这个长度序列将被阶段或特殊处理
            num_attention_heads: int = 8,  # 多头注意力机制中的注意力头数量
            num_hidden_layers: int = 8,  # Transformer中隐藏层数量, 决定模型的深度
            num_key_value_heads: int = 2,  # 键值对注意力头的数量, 用于实现GQA优化
            vocab_size: int = 6400,  # 词汇表大小,模型可识别token的总数
            rms_norm_eps: float = 1e-5,  # RMSNorm归一化的极小值, 用于防止零错误
            rope_theta: int = 1000000.0,  # ROPE(旋转位置编码)位置编码中的theta
            flash_attn: bool = True,  # 是否使用FlashAttention

            # 下面是专家MOE的参数,当use_moe为false,下面参数失效
            use_moe: bool = False,  # 是否启用MOE结构
            num_experts_per_tok: int = 2,  # 每个token选择的专家数量，决定了每个输入token由多少个专家网络处理
            n_routed_experts: int = 4,  # 可被动态路由选择的专家数量(路由专家), 决定每个输入token由多少个专家网络处理
            n_shared_experts: int = 1,  # 共享专家数
            scoring_func: str = 'softmax',  # 专家选择的评分函数
            aux_loss_alpha: float = 0.1,  # MOE辅助的损失的权重系数, 用于平衡各个专家的负载
            seq_aux: bool = True,  # 是否使用序列级别的辅助损失，进一步优化专家选择的稳定性
            norm_topk_prob: bool = True,  # 是否对选中专家的概率进行归一化, 稳定梯度计算
            **kwargs  # 其他未明确列出参数
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        # MOE的参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts  # 总专家数
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# Nullion Model 大模型结构

import math
import torch
from torch import nn

# Hugging Face Transformers 库中的激活函数映射表
from transformers.activations import ACT2FN

# Python 的类型提示模块，用于指定函数参数和返回值的类型, 提高代码的可读性和可维护性
from typing import Optional, Tuple, List, Union

import torch.nn.functional as F

# PretrainedConfig用于定义模型参数的配置参数, PreTrainedModel: 所有预训练模型的基类, GenerationMixin: 提供文本生成相关的方法
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig

# Transformers 库中定义的因果语言模型（Causal LM）输出格式
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    RMSNorm（Root Mean Square Normalization）归一化层
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化RMSNorm层"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """执行RMS归一化核心计算"""
        # .mean(-1, keepdim=True)：在最后一个维度上计算均值，keepdim=True保持维度不变
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向传播函数, 定义RMSNorm层的计算流程"""
        return self.weight * self._norm(x.float()).type_as(x)


# RoPe 旋转位置嵌入
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """预计算RoPE中的余弦和正弦频率矩阵"""
    # 计算频率刻度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    # 生成位置索引
    t = torch.arange(end, device=freqs.device)
    # 计算位置-频率矩阵(outer product外积)
    freqs = torch.outer(t, freqs).float()
    # 扩展频率矩阵到完整维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """RoPe应用到Q和K矩阵, 将位置信息融入注意力计算中"""

    # 辅助函数:将输入向量的后一半维度与前一半维度进行旋转拼接,实现复数域中的选择操作
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 根据位置索引选择对应的cos和sin值
    if position_ids is None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    # 扩展cos和sin的维度,使其与q/k的维度对齐(多头维度)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 对查询向量应用旋转编码
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep:int) -> torch.Tensor:
    """复制键V或值K的注意力头,用于实现分组查询注意力(GQA)或多查询注意力(MQA)"""
    # 解析输入张量的维度
    bs, slen, num_key_value_heads, head_dim = x.shape

    # 若无需复制,直接返回原张量
    if n_rep == 1:
        return x
    # 1. 第3维插入一个新维度
    # 2. 重塑张量 第二维和第三维合并
    # 3. 沿新插入维度扩展n_rep倍
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep)
    )


# Attention


