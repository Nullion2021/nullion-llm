import torch
import math
from torch import nn
from torchvision.models.video.mvit import PositionalEncoding

from model.nullion import ModelConfig, Attention, precompute_freqs_cis

def test_attention():
    """测试注意力机制"""
    config = ModelConfig()
    attention = Attention(config)
    batch_size = 2
    seq_len = 512
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=head_dim,
        end=config.max_position_embeddings
    )

    # 修复：使用正确的切片，获取前seq_len个位置的编码
    position_embeddings = (
        freqs_cos[:seq_len],  # 形状: [seq_len, head_dim]
        freqs_sin[:seq_len]   # 形状: [seq_len, head_dim]
    )

    print(f"查询/键形状: {x.shape}")
    print(f"位置编码cos形状: {position_embeddings[0].shape}")
    print(f"位置编码sin形状: {position_embeddings[1].shape}")

    output = attention(x, position_embeddings)

    print(f"输出形状: {output.shape}")
    assert output.shape == x.shape, f"输入输出维度不一致! 输入: {x.shape}, 输出: {output.shape}"
    print("测试成功: 输入输出维度一致")

def test_shorter_sequence():
    """测试较短序列"""
    config = ModelConfig()
    attention = Attention(config)
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=head_dim,
        end=config.max_position_embeddings
    )

    position_embeddings = (
        freqs_cos[:seq_len],
        freqs_sin[:seq_len]
    )

    output = attention(x, position_embeddings)
    assert output.shape == x.shape, f"输入输出维度不一致! 输入: {x.shape}, 输出: {output.shape}"
    print("短序列测试成功")

def test_different_config():
    """测试不同配置"""
    config = ModelConfig(hidden_size=256, num_attention_heads=4)
    attention = Attention(config)
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=head_dim,
        end=config.max_position_embeddings
    )

    position_embeddings = (
        freqs_cos[:seq_len],
        freqs_sin[:seq_len]
    )

    output = attention(x, position_embeddings)
    assert output.shape == x.shape, f"输入输出维度不一致! 输入: {x.shape}, 输出: {output.shape}"
    print("不同配置测试成功")

if __name__ == "__main__":
    print("开始测试注意力机制...\n")

    test_attention()
    test_shorter_sequence()
    test_different_config()

    print("\n🎉 所有注意力测试通过!")