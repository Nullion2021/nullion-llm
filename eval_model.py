import argparse
import random
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.nullion import ModelConfig, NullionForCausalLM

import torch
warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    if args.load == 0:
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}.pth'

        model = NullionForCausalLM(ModelConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers
        ))

        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

    else:
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    print(f'Nullion模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        # 通用对话问题
        prompt_datas = [
            '请介绍一下自己。',
            '你更擅长哪一个学科？',
            '鲁迅的《狂人日记》是如何批判封建礼教的？',
            '我咳嗽已经持续了两周，需要去医院检查吗？',
            '详细的介绍光速的物理概念。',
            '推荐一些杭州的特色美食吧。',
            '请为我讲解"大语言模型"这个概念。',
            '如何理解ChatGPT？',
            'Introduce the history of the United States, please.'
        ]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with Nullion")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # 此处max_seq_len（最大输出长度）并不意味模型具有对应的长文本的性能，仅防止QA出现被截断的问题
    # MiniMind2-Small (26M)：(hidden_size=512, num_hidden_layers=8)
    # MiniMind2 (104M)：(hidden_size=768, num_hidden_layers=16)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    # 携带历史对话上下文条数
    # history_cnt需要设为偶数，即【用户问题, 模型回答】为1组；设置为0时，即当前query不携带历史上文
    # 模型未经过外推微调时，在更长的上下文的chat_template时难免出现性能的明显退化，因此需要注意此处设置
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型，4: RLAIF-Chat模型")
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        # setup_seed(random.randint(0, 2048))
        setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0: print(f'👶: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_seq_len,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature
        )

        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()
