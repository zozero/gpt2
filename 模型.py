"""
论文下载网站：https://arxiv.org/
英文源码地址：https://github.com/openai/gpt-2
gpt-2源码：https://github.com/openai/gpt-2/tree/master
需要能够访问外网
模型下载地址：https://huggingface.co/openai-community/gpt2
视频讲解地址：https://www.youtube.com/watch?v=l8pRSuU81PU&t=920s
视频源码地址：https://github.com/karpathy/build-nanogpt/tree/master
"""
import inspect
import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as 函

from 中文状态字典 import 中对英状态字典对应字典
from 数据加载器 import 轻量数据加载器


@dataclass
class 预生转换器配置2:
    # 序列长度，就是输入句子的长度。
    序长: int = 256
    # 输入的字词的数量
    字量: int = 65
    # 隐藏层的层数
    层数: int = 6
    头数: int = 6
    # 嵌入层的长度
    # 数量无法形容，可能导致意思变成多个嵌入层，这里需要衡量嵌入层的长度
    嵌长: int = 384


@dataclass
class 预生转换器配置:
    # 序列长度，就是输入句子的长度。
    序长: int = 1024
    # 有多少不同的文字
    字量: int = 50257
    # 隐藏层的层数
    层数: int = 12
    头数: int = 12
    # 嵌入层的长度
    # 数量无法形容，可能导致意思变成多个嵌入层，这里需要衡量嵌入层的长度
    嵌长: int = 768


class 正切高斯误差线性单元(nn.Module):
    """
    这代码不会使用，但可以用作某个参考
    """

    def forward(self, 输入):
        """
        TanhGELU
        火炬文档相关链接：https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#gelu
        :param 输入:
        :return:
        """
        return 0.5 * 输入 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (输入 + 0.044715 * torch.pow(输入, 3.0))))


class 多层感知机模块(nn.Module):
    def __init__(self, 配置: 预生转换器配置):
        """

        :param 配置:
        """
        super().__init__()
        # c_fc：Conv1d Full Connected，这里命名是由于openai使用的是Conv1d（一位卷积层），
        # 视频作者为了保持变量名一致，所以没改
        self.全连接 = nn.Linear(配置.嵌长, 4 * 配置.嵌长)
        # 高斯误差线性单元火炬地址：https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
        # 论文地址：https://arxiv.org/abs/1606.08415
        self.高斯误差线性单元 = nn.GELU(approximate='tanh')
        # c_proj：Conv1d Projection
        self.投影 = nn.Linear(4 * 配置.嵌长, 配置.嵌长)

        self.投影.小小预生转换器比例差值 = 1

    def forward(self, x):
        """

        :param x: 输入
        :return:
        """
        x = self.全连接(x)
        x = self.高斯误差线性单元(x)
        x = self.投影(x)

        return x


class 因果自注意力模块(nn.Module):
    def __init__(self, 配置: 预生转换器配置):
        """

        :param 配置:
        """
        super().__init__()
        # 断言嵌长是头数的倍数
        assert 配置.嵌长 % 配置.头数 == 0
        # 这里命名是由于openai使用的是Conv1d（一位卷积层），
        # 视频作者为了保持变量名一致，所以没改。
        # 所有头部的键、查询、值投影，但在一个批次中
        self.注意力 = nn.Linear(配置.嵌长, 3 * 配置.嵌长)
        # 输出投影
        self.投影 = nn.Linear(配置.嵌长, 配置.嵌长)
        # 正则化
        self.头数 = 配置.头数
        self.嵌长 = 配置.嵌长
        # 并不是真正的“偏置项”，更像是一种掩饰，但遵循了 OpenAI/HF 的命名，我这里改名了。
        # 注册缓存区，用于在模型中注册一个不可训练的（non-trainable）参数的方法。
        # torch.tril，它返回一个输入张量的下三角部分，其余部分被设置为零。这个函数通常用于创建下三角矩阵，但也可以用于更高维度的张量。
        # lower triangular：下三角
        self.register_buffer("偏置项", torch.tril(torch.ones(配置.序长, 配置.序长)).view(1, 1, 配置.序长, 配置.序长))

    def forward(self, x):
        """
        # 来源torch.nn.functional
        def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype)
            if is_causal:
                assert attn_mask is None
                # .tril() 是一个张量方法，它返回一个新的张量，该张量包含了原张量的下三角部分。参数 对角线（diagonal） 控制了返回下三角部分的边界。
                # diagonal=0：这意味着返回的下三角部分将包括主对角线（从左上角到右下角的对角线），并且主对角线上的元素保持不变。所有主对角线以上的元素（即上三角部分）将被设置为0。
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                # 原张量的所有位置，其中 temp_mask 原本是 False（在逻辑非操作后变为 True）的位置，都会被填充为 float("-inf")。
                # 首先，对 temp_mask 这个布尔型张量执行逻辑非操作。temp_mask 是一个布尔张量，其中 True 表示需要被掩码的位置，False 表示不需要被掩码的位置。执行逻辑非操作后，True 和 False 的位置会互换。
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(query.dtype)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias += attn_mask
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            # 负无穷加减乘除上任何数都是负无穷
            # 加法和减法：
            # 负无穷加上任何正数仍然是负无穷。
            # 负无穷加上任何负数也是负无穷。
            # 负无穷减去任何正数仍然是负无穷。
            # 负无穷减去任何负数也是负无穷。
            # 乘法：
            # 负无穷乘以任何正数是负无穷。
            # 负无穷乘以任何负数是正无穷。
            # 除法：
            # 负无穷除以任何正数是负无穷。
            # 负无穷除以任何负数是正无穷。
            attn_weight += attn_bias
            # 这里使用软最大后取值在[0,1]之间，无穷小得出值无限接近于零
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value

        :param x: 输入
        :return:
        """
        # 批的大小、序列的长度、嵌入层的维度（通道）
        批, 序, 道 = x.size()
        # 计算批次中所有头的询、键和值，并将头向前移动到批后面
        # 通道数量= 头的数量 * 头的尺寸
        # 例如在预生转换器-2（124兆）中，头数=12，头寸=64，因此转换器中的 头数x头寸=通道=768 个通道
        询键值 = self.注意力(x)
        # 根据嵌入层的长度在维度2上分割
        询, 键, 值 = 询键值.split(self.嵌长, dim=2)
        # transpose(1,2)，维度1和维度2交换
        键 = 键.view(批, 序, self.头数, 道 // self.头数).transpose(1, 2)
        询 = 询.view(批, 序, self.头数, 道 // self.头数).transpose(1, 2)
        值 = 值.view(批, 序, self.头数, 道 // self.头数).transpose(1, 2)

        注意力 = (询 @ 键.transpose(-2, -1)) * (1.0 / math.sqrt(键.size(-1)))
        注意力 = 注意力.masked_fill(self.偏置项[:, :, :序, :序] == 0, float("-inf"))
        注意力 = 函.softmax(注意力, dim=-1)
        y = 注意力 @ 值

        # 这里是为了优化计算性能，所以直接使用以下函数代替了前面4行计算的代码
        # 它需要先开启编译模式，才能够使用，如果没编译就使用那么会弹出警告，不过使用了该函数，依旧能提升些许性能。
        # 详细计算过程请看初始化的注释
        # y=函.scaled_dot_product_attention(询,键,值,is_causal=True)

        y = y.transpose(1, 2).contiguous().view(批, 序, 道)
        y = self.投影(y)
        return y


class 块(nn.Module):
    def __init__(self, 配置: 预生转换器配置):
        """
        转换器的基础块
        :param 配置:
        """
        super().__init__()
        self.第一个归一层 = nn.LayerNorm(配置.嵌长)
        self.注意力层 = 因果自注意力模块(配置)
        self.第二个归一层 = nn.LayerNorm(配置.嵌长)
        # mlp：Multi Layer Perceptron
        self.多层感知机 = 多层感知机模块(配置)

    def forward(self, x):
        """

        :param x: 输入
        :return:
        """
        x = x + self.注意力层(self.第一个归一层(x))
        x = x + self.多层感知机(self.第二个归一层(x))
        return x


class 预生转换器(nn.Module):
    def __init__(self, 配置: 预生转换器配置):
        """
        预训练生成式转换器
        :param 配置: 预训练生成式转换器的配置
        """
        super().__init__()
        self.配置 = 配置

        self.转换器 = nn.ModuleDict(dict(
            # 字嵌入层的权重，wte：weight token embedding
            # 在转换器图中代表输出的嵌入层（Output Embedding）
            字嵌权=nn.Embedding(配置.字量, 配置.嵌长),
            # 位置嵌入层的权重，wpe：weight positional embedding
            # 在转换器图中代表位置编码器（Positional Encoding）
            位嵌权=nn.Embedding(配置.序长, 配置.嵌长),
            # 隐藏层，hidden
            # 转换器的骨干部分
            隐藏层=nn.ModuleList([块(配置) for _ in range(配置.层数)]),
            # 最后，final，最后的归一化层
            末归层=nn.LayerNorm(配置.嵌长)
        ))
        # 大语言模型的头，language model head
        # 如果你是从预训练模型载入权重数据，那么这层的权重数据等于字嵌权层的数据
        # 可能是由于它们的功能是一样的所以值样的话也行。
        # https://github.com/openai/gpt-2/blob/master/src/model.py，第157、171行
        self.大言模头 = nn.Linear(配置.嵌长, 配置.字量, bias=False)

        # 视频讲这样做会稍微好一些，它们会指向同一存放地址
        # self.转换器.字嵌权.weight.data_ptr()等于self.大言模头.weight.data_ptr()
        self.转换器.字嵌权.weight = self.大言模头.weight

        # self.apply() 的主要作用是方便地对神经网络中的所有层（模块）执行某个操作。
        self.apply(self._初始化_权重)

    def _初始化_权重(self, 模块):
        """
        手动初始化模块
        在最后的自注意力块之后增加了一个额外的层归一化的初始化方法考虑了随着模型深度增加在残差路径上的累积。
        在初始化时，我们将残差层的权重按1/N的比例进行缩放，其中N是残差层的数量。
        :param 模块:
        :return:
        """
        if isinstance(模块, nn.Linear):
            标准差 = 0.02
            if hasattr(模块, "小小预生转换器比例差值"):
                标准差 *= (2 * self.配置.层数) ** -0.5
            torch.nn.init.normal_(模块.weight, mean=0.0, std=标准差)
            if 模块.bias is not None:
                torch.nn.init.zeros_(模块.bias)
        elif isinstance(模块, nn.Embedding):
            torch.nn.init.normal_(模块.weight, mean=0.0, std=0.02)

    def forward(self, 字词, 目标=None):
        """

        :param 索引: 字词的索引，形状是（批长、序长），序列长度
        :param 目标:
        :return:
        """
        # （批长、序长）
        批, 序 = 字词.size()
        assert 序 <= self.配置.序长, f"不能前向传播长度为{序}的序列，序列长度只有{self.配置.序长}"
        # 前向传播位置编码层
        # 这里需要产生一个0到序列长度的张量，就表示他们的位置，然后就像权重一样加上去，这点非常好。
        位置 = torch.arange(0, 序, dtype=torch.long, device=字词.device)
        位置编码 = self.转换器.位嵌权(位置)
        # 前向传播字词编码层
        字词编码 = self.转换器.字嵌权(字词)
        x = 字词编码 + 位置编码
        # 前向传播转换器的基础块
        for 块 in self.转换器.隐藏层:
            x = 块(x)
        # 前向传播最后的归一化层
        x = self.转换器.末归层(x)
        # 前向传播大语言模型的头
        逻辑果 = self.大言模头(x)
        损失值 = None
        if 目标 is not None:
            损失值 = 函.cross_entropy(逻辑果.view(-1, 逻辑果.size(-1)), 目标.view(-1))
        return 逻辑果, 损失值

    # @classmethod说明
    # 第一个参数是类本身：类方法的第一个参数通常是cls，代表类本身，而不是类的实例。
    # 不需要实例化：类方法可以直接通过类来调用，而不需要创建类的实例。
    # 可以修改类状态：类方法可以修改类变量，影响所有实例。
    @classmethod
    def 来自预训练(cls, 模型类型):
        """
        载入来自抱抱脸的预训练模型
        :param 模型类型:
        :return:
        """
        assert 模型类型 in {"预生转换器2", "中等预生转换器2", "大型预生转换器2", "超大型预生转换器2"}
        from transformers import GPT2LMHeadModel
        print("载入预训练预生转换器权重类型：%s" % 模型类型)

        # 124M其中M是指百万（Million）
        配置参数字典 = {
            # 1.24亿个参数
            "预生转换器2": dict(层数=12, 头数=12, 嵌长=768),
            # 3.5亿个参数
            "中等预生转换器2": dict(层数=24, 头数=16, 嵌长=1024),
            # 7.74亿个参数
            "大型预生转换器2": dict(层数=36, 头数=20, 嵌长=1280),
            # 15.58亿个参数
            "超大型预生转换器2": dict(层数=48, 头数=25, 嵌长=1600)
        }[模型类型]

        配置参数字典['字量'] = 50257
        配置参数字典['序长'] = 1024
        配置 = 预生转换器配置(**配置参数字典)
        模型 = 预生转换器(配置)
        # 状态字典
        状典 = 模型.state_dict()
        状典键 = 状典.keys()
        # 忽略掉.注意力层.偏置项这个掩码
        状典键 = [键 for 键 in 状典键 if not 键.endswith(".注意力层.偏置项")]
        # print(状典键)

        抱抱脸模型类型字典 = {
            "预生转换器2": 'gpt2',
            "中等预生转换器2": 'gpt2-medium',
            "大型预生转换器2": 'gpt2-large',
            "超大型预生转换器2": 'gpt2-xl'
        }
        # 从 huggingface/transformers 中初始化模型
        抱抱脸模型 = GPT2LMHeadModel.from_pretrained(抱抱脸模型类型字典[模型类型])
        # 抱抱脸模型的状态字典
        抱抱脸状典 = 抱抱脸模型.state_dict()

        # 复制，同时确保所有参数在名称和形状中对齐并匹配
        抱抱脸状典键 = 抱抱脸状典.keys()
        # 忽略掉.attn.masked_bias这个，他只是用来缓冲的
        抱抱脸状典键 = [键 for 键 in 抱抱脸状典键 if not 键.endswith('.attn.masked_bias')]
        # 忽略掉.attn.bias这个掩码
        抱抱脸状典键 = [键 for 键 in 抱抱脸状典键 if not 键.endswith('.attn.bias')]
        # print(抱抱脸状典键)
        # 基本上 openai 检查点使用一个 “Conv1D” 模块，但我们只想使用一个原版的 Linear
        # 这意味着我们在导入这些权重时必须转置它们
        需转置 = ["attn.c_attn.weight", 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(抱抱脸状典键) == len(状典键), f"状态字典键匹配错误：{len(抱抱脸状典键)}不等于{len(状典键)}"

        for 键 in 抱抱脸状典键:
            # any() 是一个内置函数，它用于判断给定的可迭代对象（如列表、元组、字符串、字典等）是否至少包含一个 True 值。
            if any(键.endswith(字符串) for 字符串 in 需转置):
                # 对需要转置的 Conv1D 权重进行特殊处理
                assert 抱抱脸状典[键].shape[::-1] == 状典[中对英状态字典对应字典[键]].shape
                with torch.no_grad():
                    # 转置后拷贝给我们的模型状态字典对应的键中
                    状典[中对英状态字典对应字典[键]].copy_(抱抱脸状典[键].t())
            else:
                # 否则就直接原原本本的复制参数
                assert 抱抱脸状典[键].shape == 状典[中对英状态字典对应字典[键]].shape
                with torch.no_grad():
                    状典[中对英状态字典对应字典[键]].copy_(抱抱脸状典[键])
        return 模型

    def 配置优化器(self, 权重衰减系数, 学习率, 设备):
        """
        自定义优化器
        :param 权重衰减系数:
        :param 学习率:
        :param 设备:
        :return: 优化器
        """
        # self.named_parameters() 是一个生成器，它遍历模型中所有的参数，并返回它们的名称和值。它是nn.Module继承而来的。
        参数字典 = {参名: 参 for 参名, 参 in self.named_parameters()}
        # 再次过滤，将需要梯度的参数筛选出来，放入字典中
        参数字典 = {参名: 参 for 参名, 参 in 参数字典.items() if 参.requires_grad}
        # 创建优化器组。任何二维的参数都将应用权重衰减，否则不应用。
        # 也就是说，所有矩阵乘法（matmuls）和嵌入层（embeddings）中的权重张量都将衰减，而所有的偏置（biases）和层归一化（layernorms）则不会。
        衰减参数列表 = [参 for 名, 参 in 参数字典.items() if 参.dim() >= 2]
        不衰减参数列表 = [参 for 名, 参 in 参数字典.items() if 参.dim() < 2]

        优化组 = [
            {'params': 衰减参数列表, 'weight_decay': 权重衰减系数},
            {'params': 不衰减参数列表, 'weight_decay': 0.0}
        ]
        # .numel() 元素数量（number of elements），计算每个张量中元素的数量。
        衰减参数数量 = sum(参.numel() for 参 in 衰减参数列表)
        不衰减参数数量 = sum(参.numel() for 参 in 不衰减参数列表)
        print(f"应用衰减的张量有：{len(衰减参数列表)}，共包含{衰减参数数量}个参数")
        print(f"应用不衰减的张量有：{len(不衰减参数列表)}，共包含{不衰减参数数量}个参数")
        # 创建权重衰减版本的自适应矩估计（AdamW）优化器，如果融合（fused）版本可用的话就使用。
        # inspect.signature：审查.签名，用于获取一个函数、方法或类的构造函数的签名信息。签名信息包括函数的参数名、参数的默认值、参数的类型注解（如果有的话），以及函数是否接受关键字参数等。
        融合可用 = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # 使用融合可以优化训练速度
        使用融合 = 融合可用 and 'cuda' in 设备
        print(f"使用融合与权重衰减版本的自适应矩估计：{使用融合}")
        优化器 = torch.optim.AdamW(优化组, lr=学习率, betas=(0.9, 0.95), eps=1e-8, fused=使用融合)
        return 优化器


if __name__ == '__main__':
    返回的序列数量 = 5
    最大长度 = 30

    设备 = "cpu"
    if torch.cuda.is_available():
        设备 = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Metal Performance Shaders，“金属性能着色器”
        # 苹果芯片内置的图像处理器
        设备 = "mps"

    # 测试时为了再现某些情况
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1377)

    # 当批=4、序=1024时需要12G显存或内存，批最好是2的倍数或者幂次，奇数会导致性能下降
    训练时加载器 = 轻量数据加载器(批=4, 序=1024)

    # 这里是将数据类型转换具体看文档链接：https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    # 这样有更高的效率。毕竟精度减少了。
    # 老旧的显卡可能不适配，如果不适配的可以直接注释
    # torch.set_float32_matmul_precision('high')

    模型 = 预生转换器(预生转换器配置())
    模型.eval()
    模型.to(设备)
    # 起初需要编译会花费不少时间，后续能够大幅加快训练速度
    # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#introduction-to-torch-compile
    # 但老旧型号显卡可能不适用。
    # 这是我的报错原因，Triton 编译器只支持 CUDA Capability 7.0 或更高的设备，而你的 GTX 1080 Ti 显卡的 CUDA Capability 是 6.1。
    # 模型 = torch.compile(模型)

    优化器 = torch.optim.AdamW(模型.parameters(), lr=3e-4)

    for 索引 in range(50):
        时间1 = time.time()
        x, y = 训练时加载器.下一批()
        x, y = x.to(设备), y.to(设备)
        # 梯度清零
        优化器.zero_grad()

        # 自动混合精度，https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        # 减少神经网络的运行消耗时间和内存占用。
        # 并不是所有的部分数据都转换成bfloat16类型，可以查看下面的文档链接
        # https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
        # 我尝试了一下反而更慢了，是因为1080ti显卡不支持bfloat16，所以我更换成了dtype=float16。
        with torch.autocast(device_type=设备, dtype=torch.float16):
            # 初始时每个字出现的概率是大约是1/50527，当前设置是50527的单词
            # 所以损失值是-ln(1/50527)大概值是10.8302，
            # 代码计算出来的损失值是11.0794
            逻辑果, 损失值 = 模型(x, y)

        损失值.backward()
        优化器.step()
        # 用于同步 CUDA 事件
        # 需要注意的是，torch.cuda.synchronize() 会阻塞调用它的线程，直到所有 CUDA 操作都完成。
        # 因此，过度使用它可能会降低程序的性能，因为它会引入不必要的等待时间。
        # 通常，只有在确实需要同步操作时才应该使用它。在默认流（default stream）中，大多数 PyTorch 操作在返回之前都会自动同步，所以通常不需要显式调用 synchronize()。
        torch.cuda.synchronize()
        时间2 = time.time()
        间隔 = (时间2 - 时间1) * 1000
        每秒字词 = (训练时加载器.批 * 训练时加载器.序) / (时间2 - 时间1)
        print(f"第 {索引} 步，损失值：{损失值.item()}，时间间隔：{间隔:.2f}毫秒，字词/秒：{每秒字词:.2f}")

    exit()
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
