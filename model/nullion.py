# Model Config(配置类)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN

from model.model_nullion import NullionConfig


class ModelConfig(PretrainedConfig):
    model_type = "Nullion"

    def __init__(self,
                 dropout: float = 0.1,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 hidden_act: str = 'silu',
                 hidden_size: int = 512,
                 intermediate_size: int = None,
                 max_position_embeddings: int = 32768,
                 num_attention_heads: int = 8,
                 num_hidden_layers: int = 8,
                 vocab_size: int = 6400,
                 rope_theta: int = 1000000.0,
                 rms_norm_eps: float = 1e-5,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps


# RMSNorm层实现

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


# 旋转位置编码相关函数

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """预计算旋转位置编码的频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 原来的代码有问题：维度不匹配
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    # 修复版本：
    # freqs_cos = torch.cos(freqs)
    # freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """应用旋转位置编码到查询和键张量"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 原来的代码有问题：维度不匹配
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed


# 注意力机制实现

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 从配置中获取注意力头的数量
        self.n_local_heads = config.num_attention_heads

        # 计算每个注意力头的维度：总隐藏层大小除以头数量
        self.head_dim = config.hidden_size // self.n_local_heads

        # 查询(Query)线性投影层：将输入映射到查询空间
        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        # 键(Key)线性投影层：将输入映射到键空间
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        # 值(Value)线性投影层：将输入映射到值空间
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        # 输出线性投影层：将多头注意力结果映射回原隐藏层维度
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias=False)
        # 注意力权重的dropout层，用于正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        # 残差连接的dropout层，用于正则化
        self.resid_dropout = nn.Dropout(config.dropout)
        # 保存dropout概率
        self.dropout = config.dropout

    def forward(self,
                x: torch.Tensor,                 # 输入张量，形状为 [batch_size, seq_len, hidden_size]
                position_embeddings: Tuple[torch.Tensor, torch.Tensor]  # 旋转位置编码的余弦和正弦部分
                ):
        # 获取输入张量的形状：批次大小、序列长度、隐藏层大小
        bsz, seq_len, _ = x.shape
        # 通过线性投影层得到查询、键、值张量
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 重塑查询张量形状并转置，以便多头注意力计算
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # 重塑键张量形状并转置
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # 重塑值张量形状并转置
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # 从位置编码元组中获取余弦和正弦部分
        cos, sin = position_embeddings
        # 应用旋转位置编码到查询和键张量
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        xq, xk, xv = (
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2)
        )
        # 计算注意力分数：查询与键的点积，除以缩放因子防止梯度消失
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 添加因果掩码：上三角矩阵填充负无穷，防止看到未来信息
        scores = scores + torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # 对注意力分数应用softmax，得到注意力权重
        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        # 对注意力权重应用dropout
        scores = self.attn_dropout(scores)
        # 将注意力权重与值相乘，得到加权的值表示
        output = scores @ xv

        # 将多头注意力结果转置并重塑回原始形状
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 应用输出投影和dropout
        output = self.resid_dropout(self.o_proj(output))
        # 返回注意力层的输出
        return output


# 前馈网络实现

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 如果没有指定中间层大小，则按照SwiGLU的标准公式计算
        if config.intermediate_size is None:
            intermediate_size = (int(config.hidden_size * 8) / 3)
            # 将中间层大小调整为64的倍数，便于硬件优化
            config.intermediate_size = 64 * ((int(intermediate_size) + 64 - 1) // 64)
        # 门控投影层：将输入映射到门控信号
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 下投影层：将中间层结果映射回原隐藏层维度
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 上投影层：将输入映射到中间层维度
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(config.dropout)
        # 激活函数，根据配置选择（默认为SiLU）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU前向传播：
        # 1. 通过上投影层处理输入
        # 2. 通过门控投影层处理输入，然后应用激活函数
        # 3. 将门控信号与上投影结果相乘
        # 4. 通过下投影层映射回原维度
        # 5. 应用dropout并返回结果
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


# Transformer块实现

class ModelBlock(nn.Module):
    def __init__(self, layer_id, config: ModelConfig):
        super().__init__()
        # 保存注意力头数（从配置文件获取）
        self.num_attention_heads = config.num_attention_heads
        # 保存隐藏层维度（模型的特征维度，从配置文件获取）
        self.hidden_size = config.hidden_size
        # 计算每个注意力头的维度
        self.hidden_dim = config.hidden_size // self.num_attention_heads
        # 初始化自注意力层
        self.self_attn = Attention(config)

        # 保存当前层的ID
        self.layer_id = layer_id
        # 输入层归一化：在注意力前对输入进行归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 注意力后的层归一化：在前馈网络前对注意力输出进行归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 初始化前馈网络：使用FeedForward（门控前馈网络）
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings):
        # 保存输入的隐藏状态，用于残差连接
        residual = hidden_states

        # 自注意力计算：
        # 1. 对输入进行层归一化
        # 2. 通过自注意力层处理，传入位置编码
        # 3. 得到注意力输出
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings
        )

        # 注意力残差连接：将注意力输出与原始输入相加
        hidden_states += residual

        # 前馈网络计算：
        # 1. 对注意力输出进行层归一化
        # 2. 通过前馈网络处理
        # 3. 再次进行残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        # 返回处理后的隐藏状态
        return hidden_states


# Nullion模型主体实现

class NullionModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([ModelBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)

        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                **kwargs):
        """模型前向传播"""
        # 获取输入形状：批次大小和序列长度
        batch_size, seq_len = input_ids.shape

        # 词嵌入：将token索引转换为词向量，并应用dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前位置编码（RoPE）
        position_embeddings = (
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len]
        )

        # 通过多层Transformer块处理
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states


# 因果语言建模模型实现

class NullionForCausalLM(PreTrainedModel, GenerationMixin):
    """
        Nullion模型的因果语言建模（CausalLM）封装类，用于文本生成任务（如续写、翻译等）。
        继承自HuggingFace的PreTrainedModel（提供模型加载/保存等基础功能）和GenerationMixin（提供文本生成方法）。
    """

    # 指定配置类, 用于模型参数初始化和配置管理
    config_class = NullionConfig

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        super().__init__(self.config)
        # 初始化模型主体（NullionModel，包含嵌入层和多个ModelBlock）
        self.model = NullionModel(self.config)
        # 语言模型头（LM Head）：将隐藏层特征映射到词表维度，用于预测下一个token
        # 输入维度：hidden_size（模型隐藏层维度），输出维度：vocab_size（词表大小）
        # 无偏置（bias=False），符合大模型轻量化设计
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重共享：将词嵌入层（embed_tokens）的权重与LM头的权重绑定
        # 作用：减少参数量，同时在预训练中使输入嵌入和输出预测共享语义空间
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        前向传播：输入token序列 → 模型主体处理 → 输出预测logits
        Args:
           input_ids: 输入token的索引序列，形状为(bsz, seq_len)
           attention_mask: 注意力掩码，形状为(bsz, seq_len)，标记有效token位置
           past_key_values: 历史KV缓存列表，每个元素为一个Transformer层的(K,V)缓存
           use_cache: 是否缓存当前层的KV用于后续生成（推理时启用）
           logits_to_keep: 控制输出logits的范围（仅保留最后N个token的预测结果）
               - 整数N：保留最后N个token的logits
               - 张量：按索引保留指定位置的logits
           **args: 其他可选参数（如position_ids等）
        Returns:
           CausalLMOutputWithPast: 包含logits、past_key_values等的输出对象
       """

        # 通过模型主体获取隐藏状态
        h = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # 确定需要保留的logits范围（优化效率，避免计算所有位置的logits）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 计算预测logits：通过LM头将隐藏层特征映射到词表维度
        logits = self.lm_head(h[:, slice_indices, :])

        # 填充输出对象
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', None)  # 当前模型没有MoE，aux_loss为None
        self.OUT.__setitem__('past_key_values', None)  # 当前模型没有KV缓存
        return self.OUT