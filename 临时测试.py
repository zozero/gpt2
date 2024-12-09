import matplotlib.pyplot as plt
import numpy as np
import torch

from 训练 import 获得学习率


def 学习率变化曲线():
    x = np.linspace(1, 60, 60)
    y=list(map(获得学习率,x))
    plt.figure(1)
    plt.plot(x, y)
    plt.show()

def 测试编码():
    # 学习率变化曲线()
    import tiktoken

    # 获取 GPT-2 编码器
    gpt2_encoder = tiktoken.get_encoding("gpt2")

    # 打印编码器的名称
    print(f"Encoder name: {gpt2_encoder.name}")

    # 打印前几个 BPE 合并规则
    print(gpt2_encoder.encode("你好世界"))

    # 打印特殊 tokens
    print(f"Special tokens: ")

if __name__ == '__main__':
    a1=torch.load("./日志/"+"模型_00099.pt",weights_only=False)
    print(a1["模型"])

