import os
import json

import torch
from prompt_toolkit.shortcuts import input_dialog
from torch.utils.data import Dataset

# 设置tokenizers并行化环境变量，避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    """
    预训练数据集类，用于语言模型的预训练任务

    该数据集继承自PyTorch的Dataset类，专门用于处理大规模文本数据的预训练。
    支持动态批处理、序列长度控制和损失掩码计算。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集

        Args:
            data_path (str): 数据文件路径，每行一个JSON格式的文本样本
            tokenizer: 分词器对象，用于文本编码
            max_length (int): 最大序列长度，超过此长度的文本将被截断
        """
        super().__init__()
        self.tokenizer = tokenizer  # 保存分词器
        self.max_length = max_length  # 保存最大序列长度
        self.samples = self.load_data(data_path)  # 加载数据

    def load_data(self, path):
        """
        从JSON文件中加载数据样本

        Args:
            path (str): 数据文件路径

        Returns:
            list: 包含所有数据样本的列表，每个样本是一个字典
        """
        samples = []  # 初始化样本列表
        with open(path, 'r', encoding='utf-8') as f:
            # 逐行读取文件，记录行号便于调试
            for line_num, line in enumerate(f, 1):
                # 解析JSON格式的文本数据
                data = json.loads(line.strip())
                samples.append(data)
            return samples

    def __len__(self):
        """
        返回数据集的大小

        Returns:
            int: 数据集中样本的总数
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本，并进行预处理

        Args:
            idx (int): 样本索引

        Returns:
            tuple: (X, Y, loss_mask)
                X (torch.Tensor): 输入序列（去掉最后一个token）
                Y (torch.Tensor): 目标序列（去掉第一个token）
                loss_mask (torch.Tensor): 损失掩码，标记哪些位置需要计算损失
        """
        # 获取指定索引的样本
        sample = self.samples[idx]

        # 使用分词器对文本进行编码
        encoding = self.tokenizer(
            str(sample['text']),  # 确保文本为字符串格式
            max_length=self.max_length,  # 限制最大长度
            padding="max_length",  # 填充到固定长度
            truncation=True,  # 截断过长的文本
            return_tensors="pt",  # 返回PyTorch张量
        )

        # 去除batch维度，得到形状为 (seq_len,) 的张量
        input_ids = encoding.input_ids.squeeze()

        # 创建损失掩码：非填充token的位置为True，填充token的位置为False
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 构造自回归训练的输入和目标：
        # X: 输入序列（去掉最后一个token）
        # Y: 目标序列（去掉第一个token，即预测下一个token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)

        # 对应的损失掩码也去掉第一个位置（因为目标序列去掉了第一个token）
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample = self.load_data(json_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.sample)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.sample[index]
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask

class DPODataset(Dataset):
    """
    DPO (Direct Preference Optimization) 数据集类

    用于直接偏好优化的数据集，每条数据包含一个被选择的回复和一个被拒绝的回复。
    模型需要学习区分哪个回复更好，从而提升回复质量。
    """

    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化DPO数据集

        Args:
            file_path (str): JSON文件路径，每行包含chosen和rejected两个字段
            tokenizer: 分词器对象
            max_length (int): 最大序列长度，默认4096
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 获取助手回复的开始和结束标记
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False)
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False)
        # 加载DPO训练数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的DPO数据样本

        Args:
            index (int): 样本索引

        Returns:
            dict: 包含chosen和rejected数据的字典
        """
        item = self.data[index]
        chosen = item['chosen']
        rejected = item['rejected']

        # 为被选择的回复生成对话模板
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        # 为被拒绝的回复生成对话模板
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # 编码被选择的回复
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        # 编码被拒绝的回复
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 获取编码后的输入ID序列
        chosen_input_ids = chosen_encoding['input_ids']
        # 生成被选择回复的损失掩码
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        # 生成被拒绝回复的损失掩码
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        # 构造自回归训练数据：输入序列（去掉最后一个token）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        # 目标序列（去掉第一个token，预测下一个token）
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        # 对应的损失掩码
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        # 同样处理被拒绝的回复
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        """
        生成损失掩码，只对助手回复内容计算损失

        Args:
            input_ids (list): 输入token ID序列

        Returns:
            list: 损失掩码，1表示计算损失，0表示不计算
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 查找助手回复开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 查找助手回复结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 对助手回复内容（除第一个token外）设置损失掩码为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask