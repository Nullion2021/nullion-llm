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
            hidden_size: int = 512,  # 隐藏层层数
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
        self.hidden_size = hidden_size
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
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
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
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# Attention

class Attention(nn.Module):
    def __init__(self, args: NullionConfig):
        super().__init__()
        # 确定键值头的数量, 如果配置中指定了num_key_value_heads则使用, 否则使用注意力头总数
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        # 验证注意力头总数必须是键值头数量的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0

        # 记录本地注意力头总数(通常与全局配置的注意力头数量一致)
        self.n_local_heads = args.num_attention_heads

        # 记录本地键值头的数量(通常与配置一致)
        self.n_local_kv_heads = self.num_key_value_heads

        # 计算每个键值头需要重复的次数,实现多头注意力中的分组映射
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 计算每个注意力头的维度: 隐藏层维度除以注意力头维度
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 初始化查询投影层：将输入的隐藏层维度映射到所有注意力头的总维度（不带偏置）
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # 初始化键投影层：将输入的隐藏层维度映射到键值头的总维度（不带偏置）
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 初始化值投影层：将输入的隐藏层维度映射到键值头的总维度（不带偏置）
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 初始化输出投影层：将所有注意力头的输出维度映射回原始隐藏层维度（不带偏置）
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        # 初始化注意力dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        # 初始化残差连接dropout层
        self.resid_dropout = nn.Dropout(args.dropout)
        # 记录dropout概率值
        self.dropout = args.dropout

        # 确定是否使用Flash注意力加速: 检查Pytorch是否支持
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):

        # 获取输入张量的形状：batch_size(批量大小)、sequence_length(序列长度)
        bsz, seq_len, _ = x.shape

        # 通过投影层生成Q、K、V矩阵
        # xq: 查询矩阵 (bsz, seq_len, num_attention_heads * head_dim)
        # xk: 键矩阵 (bsz, seq_len, num_key_value_heads * head_dim)
        # xv: 值矩阵 (bsz, seq_len, num_key_value_heads * head_dim)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 重塑Q、K、V为多头格式（拆分头部维度）
        # xq形状变为: (bsz, seq_len, n_local_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # xk/xv形状变为: (bsz, seq_len, n_local_kv_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用RoPE
        # 从position_embeddings获取cos和sin分量
        cos, sin = position_embeddings
        # 对查询和键矩阵施加旋转位置编码，增强位置敏感性
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache
        if past_key_value is not None:
            # 如果存在历史KV缓存,将当前KV与历史KV拼接(延长序列)
            xk = torch.cat([past_key_value[0], xk], dim=1) # 拼接键矩阵
            xv = torch.cat([past_key_value[1], xv], dim=1) # 拼接值矩阵

        # 若需要缓存,保存当前kv供下次推理使用
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序并扩展KV头
        xq, xk, xv = (
            xq.transpose(1, 2), #  Q维度变为: (bsz, n_local_heads, seq_len, head_dim)
            # 对KV进行重复扩展（n_rep倍）以匹配Q的头数，再调整维度
            repeat_kv(xk, self.n_rep).transpose(1, 2), # K维度变为: (bsz, n_local_heads, seq_len, head_dim)
            repeat_kv(xv, self.n_rep).transpose(1, 2)  # V维度变为: (bsz, n_local_heads, seq_len, head_dim)
        )

        # 注意力技算(分支1:使用flash Attention加速)
        if self.flash and seq_len != 1:
            # 训练时使用dropout, 推理时关闭
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None

            # 处理注意力掩码(padding掩码)
            if attention_mask is not None:
                # 扩展掩码维度以匹配注意力分数形状
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None # 转为布尔掩码

            # 调用PyTorch内置的Flash注意力实现（高效GPU加速）
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        # 注意力计算(常规计算)
        else:
            # 注意力分数计算: Q与K的点积，除以头维度的平方根（缩放）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 施加因果掩码（上三角矩阵置为负无穷，防止关注未来token
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1 # 对角线以上的元素被masking
            ).unsqueeze(0).unsqueeze(0) # 扩展维度以匹配scores

            # 施加注意力掩码
            if attention_mask is not None:
                # 掩码转为负值无穷形式
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # 计算注意力权重并应用dropout
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 注意力权重与V矩阵相乘,得到输出
            output = scores @ xv

        # 调整维度顺序并重塑为 (bsz, seq_len, 总维度)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 通过输出投影层映射回隐藏层维度，并应用残差dropout
        output = self.resid_dropout(self.o_proj(output))
        # 返回注意力输出和KV缓存（若启用）
        return output, past_kv

class FeedForward(nn.Module):
    """前馈神经网络模块: 门控前馈"""
    def __init__(self, config: NullionConfig):
        super().__init__()
        # 计算中间层维度
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3) # 若为指定中间层维度,则按隐藏层维度的8/3倍计算
            config.intermediate_size = 64 * ((intermediate_size + 64 -1)//64) # 将中间层维度调整为64的整数倍

        # 定义三个线性投影层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        # 定义dropout层
        self.dropout = nn.Dropout(config.dropout)

        # 获取激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 门控机制计算流程：
        # 1. 门控路径：输入 → 门控投影 → 激活函数
        # 2. 上投影路径：输入 → 上投影
        # 3. 两路结果逐元素相乘 → 下投影 → dropout → 输出
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MoEGate(nn.Module):
    """
    混合专家模型的门控模块,负责将输入序列分配给不同的专家网络,并计算辅助损失
    核心功能:1. 生成输入与各专家的匹配分数; 2. 选择Top-K个专家
    """
    def __init__(self, config: NullionConfig):
        super().__init__()

        # 保存配置对象
        self.config = config
        # 每个token需要的分配的专家数量(Top-K)
        self.top_k = config.num_experts_per_tok
        # 可路由的专家总数(及模型中专家网络的数量)
        self.n_routed_experts = config.n_routed_experts

        # 门控分数的计算函数(softmax)
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数(控制负载均衡损失的影响程度)
        self.alpha = config.aux_loss_alpha
        # 是否按照序列维度计算辅助损失
        self.seq_aux = config.seq_aux

        # 是否对Top-K专家的权重进行归一化(确保权重和为1)
        self.norm_topk_prob = config.norm_topk_prob
        # 门控网络的输入维度(与Transform隐藏层维度一致)
        self.gating_dim = config.hidden_size

        # 门控网络的核心参数: 权重矩阵,形状为(专家数, 隐藏层维度)
        # 作用: 将输入隐藏态映射为各专家的匹配分数
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 初始化权重参数(使用Kaiming均匀分布初始化)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 导入PyTorch的参数初始化模块
        import torch.nn.init as init

        # 用Kaiming均匀分布初始化权重矩阵，适用于ReLU类激活函数的场景
        # a=math.sqrt(5)是该初始化方法的默认参数，对应增益系数的计算
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        门控模块前向传播：输入隐藏态 → 计算专家匹配分数 → 选择Top-K专家 → 计算辅助损失
        Args:
            hidden_states: Transformer输出的隐藏态，形状为(bsz, seq_len, hidden_size)
        Returns:
            topk_idx: Top-K专家的索引，形状为(bsz*seq_len, top_k)
            topk_weight: Top-K专家的归一化权重，形状为(bsz*seq_len, top_k)
            aux_loss: 辅助损失（训练时用于负载均衡，推理时为0）
        """
        # 解析输入形状
        bsz, seq_len, h = hidden_states.shape
        # 重塑输入 重塑输入：将(bsz, seq_len, h)展平为(bsz*seq_len, h)
        # 将每个token视为独立样本, 便于批量计算与所有专家的匹配分数
        hidden_states = hidden_states.view(-1, h)

        # 计算门控分数: 输入隐藏态 x 权重矩阵
        logits = F.linear(hidden_states, self.weight, None)

        # 将分数转化为概率分布
        if self.scoring_func == 'softmax':
            # softmax沿专家维度(-1)计算,得到每个token对各专家的匹配概率
            scores = logits.softmax(dim=-1)
        else:
            # 抛出异常
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择Top-K个匹配概率最高的专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 若Top-k>1且启用归一化, 将Top-K权重归一化(确保权重和为1)
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算Top-K权重的和(加1e-20避免分母为0)
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # 归一化
            topk_weight = topk_weight / denominator

        # 训练阶段: 计算辅助损失(平衡各个专家的负载, 避免部分专家闲置)
        if self.training and self.alpha > 0.0:
            # 辅助损失计算基于原始分数
            scores_for_aux = scores
            # 辅助损失使用的Top-K数量
            aux_topk = self.top_k
            # 将Top-K专家索引重塑为(bsz, seq_len*top_k)，便于按样本处理
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            # 分支1: 按序列维度计算辅助损失
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            # 分支2: 全局计算辅助损失
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        # 推理阶段或辅助损失权重为0
        else:
            aux_loss = 0
        # 返回: Top-K专家索引 Top-k归一化权重 辅助损失
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: NullionConfig):
        super().__init__()
        self.config = config

        # 初始化路由专家列表,每个专家都是一个FeedForward网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 初始化门控模块:负责决定输入分配给哪些路由专家
        self.gate = MoEGate(config)
        # 若配置中制定了共享专家数量, 则初始化"共享专家"列表
        # 共享专家会处理所有输入
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
            MoE前馈网络的前向传播：
            1. 门控模块决定输入分配给哪些路由专家
            2. 路由专家处理被分配的输入，共享专家处理所有输入
            3. 融合所有专家的输出，返回最终结果
            Args:
                x: 输入张量，形状为(bsz, seq_len, hidden_size)
            Returns:
                y: 融合后的输出张量，形状与输入一致(bsz, seq_len, hidden_size)
        """
        # 保存输入的原始值
        identity = x
        # 保存输入的原始形状
        orig_shape = x.shape
        # 解析输入形状
        bsz, seq_len, _ = x.shape

        # 1. 门控路由：通过门控模块获取每个token的Top-K专家索引和权重，以及辅助损失
        # topk_idx: 形状(bsz*seq_len, top_k)，每个token被分配的专家索引
        # topk_weight: 形状(bsz*seq_len, top_k)，每个专家的权重（用于融合输出）
        # aux_loss: 门控模块计算的辅助损失（用于平衡专家负载）
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 2. 输入重塑：将(bsz, seq_len, hidden_size)展平为(bsz*seq_len, hidden_size)
        # 目的：将每个token视为独立样本，便于按专家分配处理
        x = x.view(-1, x.shape[-1])

        # 将Top-K专家索引展平为1维：(bsz*seq_len*top_k,)
        # 每个元素对应一个"token-专家"对的索引
        flat_topk_idx = topk_idx.view(-1)
        # 训练阶段：路由专家处理被分配的输入（并行度优先）
        if self.training:
            # 复制输入：每个token复制top_k次（与Top-K专家数量匹配）
            # 形状从(bsz*seq_len, hidden_size)变为(bsz*seq_len*top_k, hidden_size)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            # 初始化专家输出容器，形状与复制后的x一致（用float16节省内存）
            y = torch.empty_like(x, dtype=torch.float16)

            # 遍历每个专家，处理被分配给该专家的token
            for i, expert in enumerate(self.experts):
                # 筛选出分配给当前专家i的token索引（flat_topk_idx中等于i的位置）
                # 专家处理这些token，并将结果存入输出容器的对应位置（确保数据类型一致）
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致

            # 融合专家输出：
            # a. 将y重塑为(bsz*seq_len, top_k, hidden_size)
            # b. 与topk_weight（形状(bsz*seq_len, top_k)）按元素相乘（权重加权）
            # c. 沿top_k维度求和，得到每个token的最终融合结果（形状(bsz*seq_len, hidden_size)）
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 将输出重塑为原始输入形状(bsz, seq_len, hidden_size)
            y = y.view(*orig_shape)

        # 推理阶段：路由专家处理被分配的输入（效率优先，使用缓存机制）
        else:
            # 调用moe_infer方法进行高效推理，返回形状(bsz*seq_len, hidden_size)
            # 再重塑为原始形状(bsz, seq_len, hidden_size)
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 若存在共享专家，将其输出添加到路由专家的融合结果中（残差方式）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                # 共享专家处理原始输入（identity），结果与y相加
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad() # 推理阶段禁用梯度计算，提高效率
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
           MoE推理阶段的高效实现：按专家分组处理输入，减少冗余计算
           Args:
               x: 展平后的输入张量，形状(bsz*seq_len, hidden_size)
               flat_expert_indices: 展平的专家索引，形状(bsz*seq_len*top_k,)
               flat_expert_weights: 展平的专家权重，形状(bsz*seq_len*top_k, 1)
           Returns:
               expert_cache: 所有专家输出融合后的结果，形状(bsz*seq_len, hidden_size)
        """
        # 初始化专家输出缓存（用于累加所有专家的加权输出）
        expert_cache = torch.zeros_like(x)

        # 对专家索引排序，使同一专家的token连续排列（便于批量处理）
        # idxs为排序后的索引位置，形状与flat_expert_indices一致
        idxs = flat_expert_indices.argsort()

        # 计算每个专家被分配的token数量（累计和）
        # 例如：[6, 15, 20]表示专家0有6个token，专家1有9个（15-6），专家2有5个（20-15)
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 计算每个"token-专家"对对应的原始token索引（用于将结果映射回原始位置）
        # 由于每个token被复制了top_k次，通过整除top_k得到原始索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            # 获取当前专家的网络实例
            expert = self.experts[i]
            # 筛选出当前专家处理的原始token索引（从排序后的token_idxs中截取）
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 提取这些token的输入数据
            expert_tokens = x[exp_token_idx]
            # 专家处理输入，并转换为缓存的数据类型
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 应用专家权重（按元素相乘）
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将加权后的专家输出累加到缓存的对应位置（scatter_add_实现按索引累加）
            # 1. exp_token_idx.view(-1, 1).repeat(1, x.shape[-1])：将token索引扩展到隐藏层维度
            # 2. 按扩展后的索引，将expert_out累加到expert_cache中
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class NullionBlock(nn.Module):
    """
        MiniMind模型的基础构建块（Transformer块），包含一个自注意力层和一个前馈网络层，
        采用残差连接（Residual Connection）和层归一化（Layer Normalization）增强训练稳定性。
        支持可选的混合专家（MoE）前馈网络，通过配置动态切换。
        """
    def __init__(self, layer_id: int, config: NullionConfig):
        super().__init__()
        # 保存注意力头数（从配置文件获取）
        self.num_attention_heads = config.num_attention_heads
        # 保存隐藏层维度（模型的特征维度，从配置文件获取）
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 初始化前馈网络：
        # - 若不使用MoE（use_moe=False），则使用普通的FeedForward（门控前馈网络）
        # - 若使用MoE（use_moe=True），则使用MOEFeedForward（混合专家前馈网络）
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class NullionModel(nn.Module):
    """
        Nullion模型的主体结构，包含词嵌入层、多个Transformer块（NullionBlock）、
        位置编码（RoPE）和最终的层归一化，是模型特征提取的核心模块。
        支持混合专家（MoE）结构，可通过配置动态启用。
    """
    def __init__(self, config: NullionConfig):
        super().__init__()
        self.config = config
        # 获取词表大小和隐藏库层数
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # 词嵌入层: 将token索引映射为向量
        # 输入维度: vocab_size词表大小, 输出维度: hidden_size 隐藏层维度
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Dropout层: 用于训练时随机丢弃部分词嵌入, 防止过拟合
        self.dropout = nn.Dropout(config.dropout)

        # Transformer块列表, 由多个NullionBlck组成
        self.layers = nn.ModuleList([NullionBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)

        # 将预计算的位置编码注册为非持久化缓冲区
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):

        # 解析输入形状
        batch_size, seq_length = input_ids.shape
        # 初始化历史KV缓存
        past_key_values = past_key_values or [None] * len(self.layers)
        # 计算当前序列的起始位置
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入与Dropout, 将token转化为词向量, 再应用Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前序列的位置编码RoPE
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 多层Transformer块处理
        presents = [] # 存储当前层的KV缓存
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 每个Transformer块处理输入特征, 返回更新后的特征和当前层的KV缓存
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            # 保存当前KV缓存
            presents.append(present)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        # 计算混合专家MoE的辅助损失总和
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss

class NullionForCausalLM(PreTrainedModel, GenerationMixin):
    """
        MiniMind模型的因果语言建模（CausalLM）封装类，用于文本生成任务（如续写、翻译等）。
        继承自HuggingFace的PreTrainedModel（提供模型加载/保存等基础功能）和GenerationMixin（提供文本生成方法）。
    """

    # 指定配置类, 用于模型参数初始化和配置管理
    config_class = NullionConfig

    def __init__(self, config: NullionConfig = None):
        self.config = config or NullionConfig()
        super().__init__(self.config)
        # 初始化模型主体（NullionModel，包含嵌入层和多个NullionBlock）
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

        # h: 最后一层的隐藏层特征，形状为(bsz, seq_len, hidden_size)
        # past_kvs: 当前层的KV缓存列表（若use_cache=True）
        # aux_loss: 混合专家模型的辅助损失（用于负载均衡）
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 确定需要保留的logits范围（优化效率，避免计算所有位置的logits）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # s计算预测logits：通过LM头将隐藏层特征映射到词表维度
        logits = self.lm_head(h[:, slice_indices, :])

        # 4. 填充输出对象
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT