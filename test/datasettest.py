from torch.utils.data import DataLoader, DistributedSampler  # 导入数据加载器和分布式采样器
from transformers import AutoTokenizer

from Data.lm_dataset import PretrainDataset

# 从本地model文件夹加载tokenizer配置
tokenizer = AutoTokenizer.from_pretrained("model/")

# 创建预训练数据集实例
# 参数说明：
# - data_path: 预训练数据文件路径 (JSONL格式，每行包含text字段)
# - tokenizer: 分词器，用于文本编码
# - max_length: 最大序列长度，超过此长度的文本将被截断
train_ds = PretrainDataset("Data/pretrain_hq.jsonl",
                           tokenizer,
                           max_length=512)

# 创建分布式采样器，用于多GPU训练时的数据采样
# 确保每个GPU进程处理不同的数据子集，避免重复
train_sampler = DistributedSampler(train_ds)

# 创建数据加载器，用于批量加载数据
# 参数说明：
# - batch_size: 每个批次的样本数量
# - pin_memory: 将数据固定在内存中，加速GPU传输
# - drop_last: 是否丢弃最后一个不完整的批次
# - shuffle: 是否打乱数据顺序（由sampler控制，这里设为False）
# - num_workers: 数据加载的并行进程数
# - sampler: 使用的采样器，这里使用分布式采样器
train_loader = DataLoader(
    train_ds,
    batch_size = 32,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
    num_workers=1,
    sampler = train_sampler
)
