"""
论文下载网站：https://arxiv.org/
英文源码地址：https://github.com/openai/gpt-2
需要能够访问外网
模型下载地址：https://huggingface.co/openai-community/gpt2 没用过，可能无法在火炬（pytorch）中使用，源码是用张量流写（tensorflow）
视频讲解地址：https://www.youtube.com/watch?v=l8pRSuU81PU&t=920s
"""

from dataclasses import dataclass

from torch import nn


@dataclass
class 预生转换器配置:
    #
    块量: int = 256
    # 输入的字词的数量
    字量: int = 65
    # 隐藏层的层数
    层数: int = 6
    头数: int = 6
    # 嵌入层的输出数量
    嵌数: int = 384


class 块(nn.Module):
    def __init__(self, 配置: 预生转换器配置):
        """
        转换器的基础块
        :param 配置:
        """
        super().__init__()
        self.第一个归一层=nn.LayerNorm(配置.嵌数)
        self.注意力层 = 自注意力因果(配置)


class 预生转换器(nn.Module):
    def __init__(self, 配置: 预生转换器配置):
        """
        预训练生成式转换器
        :param 配置: 预训练生成式转换器的配置
        """
        super().__init__()
        self.配置 = 配置

        self.转换器 = nn.ModuleDict(dict(
            # 词嵌入层的权重，wte：weight token embedding
            # 在转换器图中代表输出的嵌入层（Output Embedding）
            词嵌权=nn.Embedding(配置.字量, 配置.嵌数),
            # 位置嵌入层的权重，wpe：weight positional embedding
            # 在转换器图中代表位置编码器（Positional Encoding）
            位嵌权=nn.Embedding(配置.块量, 配置.嵌数),
            # 隐藏层，hidden
            # 转换器的骨干部分
            隐层=nn.ModuleList([块(配置) for _ in range(配置.层数)]),
            # 最后，final
            末端归一层=nn.LayerNorm(配置.嵌数)
        ))
        # 大语言模型的头，language model head
        self.大言模头 = nn.Linear(配置.嵌数, 配置.字量, bias=False)
