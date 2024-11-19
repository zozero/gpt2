import torch

from 模型 import 预生转换器
from torch.nn import functional as 函

if __name__ == '__main__':
    设备 = "cpu"
    if torch.cuda.is_available():
        设备 = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Metal Performance Shaders，“金属性能着色器”
        # 苹果芯片内置的图像处理器
        设备 = "mps"
    返回的序列数量 = 5
    最大长度 = 30

    模型 = 预生转换器.来自预训练("预生转换器2")
    模型.eval()
    模型.to(设备)

    # pip install tiktoken 它提供了一种简单的方式来计数和分割文本为tokens。
    import tiktoken
    # 如果你没有下载过gpt2的模型，那么它会先去网络上下载。
    编码器 = tiktoken.get_encoding("gpt2")

    # 用于查看输入的编码：https://tiktokenizer.vercel.app/，注意需要选择gpt2的模式
    字词 = 编码器.encode("Hello,I'm a language model")
    字词 = torch.tensor(字词, dtype=torch.long)
    # tensor.repeat() 是一个用于重复张量（tensor）中元素的函数，
    # 它会返回一个新的张量，其中包含了原始张量的多次复制。
    字词 = 字词.unsqueeze(0).repeat(返回的序列数量, 1)
    x = 字词.to('cuda')

    # 开始生成
    # 设置随机数种子
    while x.size(1) < 最大长度:
        with torch.no_grad():
            # （批，序，字），字：字的数量
            逻辑果 = 模型(x)
            # 只留下了最后一个输入的字，对应的下一个可能的所有文字 ，形状（批，字）
            逻辑果 = 逻辑果[:, -1, :]
            # 经过软最大，变成概率
            概率 = 函.softmax(逻辑果, dim=-1)
            # 进行50个的数顶（top-k）采样（抱抱脸管道的默认设置）
            # 数顶：数个顶部的数
            # 数顶概率形状在这里变为(5, 50)
            数顶概率, 数顶索引 = torch.topk(概率, 50, dim=-1)
            # 从概率最高的k个字词中选择一个字词。
            # multinomial：多项式抽样，
            # 这个函数会对数顶概率中的每一行独立执行多项分布抽样，返回每一行抽样结果的索引。
            # 1：这个参数指定了每个多项分布中抽取样本的数量。在这里，我们只抽取一个样本。
            # 索引的形状为(批,1)
            索引 = torch.multinomial(数顶概率, 1)
            # 收集相应的索引
            # torch.gather() 收集，函数用于从输入张量中根据指定的索引张量提取子集。
            # -1：表示数顶索引中的最后一维操作。
            x列 = torch.gather(数顶索引, -1, 索引)
            # 在维度1上连接，实际上是在句子后面再加了一个字词
            x = torch.cat((x, x列), dim=1)
    # 输出生成的结果
    for 引 in range(返回的序列数量):
        字词 = x[引, :最大长度].tolist()
        解码的字词 = 编码器.decode(字词)
        print("句子：", 解码的字词)
