import os  # 导入操作系统接口模块
import sys  # 导入系统相关功能模块
__package__ = "trainer"  # 设置包名为trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse  # 导入命令行参数解析模块
import time  # 导入时间处理模块
import math  # 导入数学计算模块
import warnings  # 导入警告处理模块
import torch  # 导入PyTorch深度学习框架
import torch.distributed as dist  # 导入PyTorch分布式训练模块
from torch import optim, nn  # 从PyTorch导入优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 导入分布式数据并行模块
from torch.utils.data import DataLoader, DistributedSampler  # 导入数据加载器和分布式采样器
from contextlib import nullcontext  # 导入空上下文管理器
from transformers import AutoTokenizer  # 从HuggingFace导入预训练分词器
from model.nullion import ModelConfig, NullionForCausalLM  # 导入MiniMind模型配置和模型类
from Data.lm_dataset import PretrainDataset  # 导入预训练数据集类

warnings.filterwarnings('ignore')  # 忽略所有警告信息

def Logger(content):  # 定义日志记录函数
    if not ddp or dist.get_rank() == 0:  # 如果不是分布式训练或是主进程(排名为0)
        print(content)  # 打印日志内容

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def init_model(lm_config):
    # 使用本地tokenizer文件，避免网络依赖
    tokenizer = AutoTokenizer.from_pretrained("../model")
    # tokenizer = PreTrainedTokenizerFast(
    #     tokenizer_file="../model/tokenizer.json",
    #     config_file="../model/tokenizer_config.json",
    #     pad_token="<|pad|>",
    #     eos_token="<|im_end|>",
    #     bos_token="<|im_start|>",
    #     unk_token="<|unk|>"
    # )
    model = NullionForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')  # 打印模型可训练参数数量
    return model, tokenizer

def get_lr(current_step, total_steps, lr):  # 定义学习率调度函数
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))  # 使用余弦退火调度学习率


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 定义交叉熵损失函数，不进行平均值计算
    start_time = time.time()  # 记录开始时间
    for step, (X, Y, loss_mask) in enumerate(train_loader):  # 遍历训练数据加载器
        X = X.to(args.device)  # 将输入数据移动到指定设备
        Y = Y.to(args.device)  # 将标签数据移动到指定设备
        loss_mask = loss_mask.to(args.device)  # 将损失掩码移动到指定设备

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)  # 计算当前学习率
        for param_group in optimizer.param_groups:  # 遍历优化器参数组
            param_group['lr'] = lr  # 更新学习率

        with ctx:  # 在指定上下文中执行(可能是混合精度训练)
            res = model(X)  # 前向传播，获得模型输出
            loss = loss_fct(  # 计算损失
                res.logits.view(-1, res.logits.size(-1)),  # 将logits展平为二维张量
                Y.view(-1)  # 将标签展平为一维张量
            ).view(Y.size())  # 恢复为原始形状
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 应用损失掩码，只计算有效位置的损失
            loss = loss / args.accumulation_steps  # 除以梯度累积步数

        scaler.scale(loss).backward()  # 使用混合精度缩放器反向传播

        if (step + 1) % args.accumulation_steps == 0:  # 达到梯度累积步数时更新参数
            scaler.unscale_(optimizer)  # 反向缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防止梯度爆炸

            scaler.step(optimizer)  # 使用缩放器更新优化器参数
            scaler.update()  # 更新缩放器

            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        if step % args.log_interval == 0:  # 每隔指定步数记录日志
            spend_time = time.time() - start_time  # 计算已用时间
            Logger(  # 打印训练状态信息
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,  # 当前epoch数
                    args.epochs,  # 总epoch数
                    step,  # 当前步数
                    iter_per_epoch,  # 每个epoch的迭代次数
                    loss.item() * args.accumulation_steps,  # 实际损失值
                    optimizer.param_groups[-1]['lr'],  # 当前学习率
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60  # 预计剩余时间
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):  # 如果使用wandb且是主进程
                wandb.log({  # 记录到wandb
                    "loss": loss.item() * args.accumulation_steps,  # 损失值
                    "lr": optimizer.param_groups[-1]['lr'],  # 学习率
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60  # 预计剩余时间
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):  # 每隔指定步数保存模型
            model.eval()  # 设置模型为评估模式
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}.pth'  # 构建检查点保存路径

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # 如果是分布式模型
                state_dict = model.module.state_dict()  # 获取原始模型的状态字典
            else:  # 如果不是分布式模型
                state_dict = model.state_dict()  # 直接获取状态字典

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 将模型参数转换为半精度以节省空间
            torch.save(state_dict, ckp)  # 保存模型检查点
            model.train()  # 设置模型为训练模式


if __name__ == "__main__":
    print(f"当前执行目录: {os.getcwd()}")
    print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"当前脚本文件: {os.path.abspath(__file__)}")

    parser = argparse.ArgumentParser(description='Nullion Pretraining')
    parser.add_argument("--out_dir", type=str, default= "../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数参数
    parser.add_argument("--batch_size", type=int, default=32)  # 批次大小参数
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # 学习率参数
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择参数
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型参数
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用wandb记录
    parser.add_argument("--wandb_project", type=str, default="Nullion-Pretrain")  # wandb项目名称
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载器工作进程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=8)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热迭代次数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志记录间隔
    parser.add_argument("--save_interval", type=int, default=100)  # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 本地排名参数
    parser.add_argument('--hidden_size', default=512, type=int)  # 模型隐藏层大小
    parser.add_argument('--num_hidden_layers', default=8, type=int)  # 隐藏层数量
    parser.add_argument('--max_seq_len', default=512, type=int)  # 最大序列长度
    parser.add_argument("--data_path", type=str, default="../../Data/gongjy/minimind_dataset/pretrain_hq.jsonl")  # 训练数据路径
    args = parser.parse_args()  # 解析命令行参数

    lm_config = ModelConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    args.save_dir = os.path.join(args.out_dir)  # 设置模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    os.makedirs(args.out_dir, exist_ok=True)  # 创建输出目录

    tokens_per_iter = args.batch_size * args.max_seq_len  # 计算每次迭代处理的token数量
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 确定设备类型

    args.wandb_run_name = f"Nullion-Pretrain-Epoch-{args.epochs}--BatchSize--{args.batch_size}-LearnignRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()  # 根据设备类型设置上下文管理器

    ddp = int(os.environ.get("RANK", -1)) != -1  # 判断是否为分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 初始化分布式训练相关变量

    base_seed = 2025  # 设置基础随机种子
    torch.manual_seed(base_seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed(base_seed)  # 设置CUDA随机种子

    if ddp:  # 如果是分布式训练
        init_distributed_mode()  # 初始化分布式训练
        args.device = torch.device(DEVICE)  # 设置设备
        rank = dist.get_rank()  # 获取进程排名
        torch.manual_seed(base_seed + rank)  # 为每个进程设置不同的随机种子
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)  # 为每个进程设置不同的CUDA随机种子

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:  # 如果是分布式训练
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 设置DDP忽略的参数
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])  # 包装为分布式模型

    iter_per_epoch = len(train_loader)  # 计算每个epoch的迭代次数
    for epoch in range(args.epochs):  # 遍历所有epoch
        train_epoch(epoch, wandb)  # 训练一个epoch
