import math
import time

import torch

from 数据加载器 import 轻量数据加载器
from 模型 import 预生转换器配置, 预生转换器


class 训练配置:
    最大步数 = 50


def 获得学习率(步数):
    """

    :param 步数: 这里也有可能是轮回的次数
    :return:
    """
    最大学习率 = 6e-4
    最小学习率 = 最大学习率 * 0.1
    预热步数 = 10
    最大步数 = 训练配置.最大步数

    # 对于预热步骤使用线性预热
    if 步数 < 预热步数:
        return 最大学习率 * (步数 + 1) / 预热步数
    # 如果大于最大步数的直接使用最小学习率
    if 步数 > 最大步数:
        return 最小学习率
    # 对于两者之间的使用余弦衰减到最小学习率
    衰减比例 = (步数 - 预热步数) / (最大步数 - 预热步数)
    assert 0 <= 衰减比例 <= 1
    # 从1开始一直衰减到0
    系数 = 0.5 * (1.0 + math.cos(math.pi * 衰减比例))
    return 最小学习率 + 系数 * (最大学习率 - 最小学习率)


if __name__ == '__main__':
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

    # 之所有设置字量=50304是因为它可以被128整除，而128是2的7次幂。这样可以优化些许性能，但要看设备和计算量
    模型 = 预生转换器(预生转换器配置(字量=50304))
    模型.eval()
    模型.to(设备)
    # 起初需要编译会花费不少时间，后续能够大幅加快训练速度
    # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#introduction-to-torch-compile
    # 但老旧型号显卡可能不适用。
    # 这是我的报错原因，Triton 编译器只支持 CUDA Capability 7.0 或更高的设备，而你的 GTX 1080 Ti 显卡的 CUDA Capability 是 6.1。
    # 模型 = torch.compile(模型)

    # 优化器 = torch.optim.AdamW(模型.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    优化器 = 模型.配置优化器(权重衰减系数=0.1, 学习率=6e-4, 设备=设备)

    for 索引 in range(训练配置.最大步数):
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
        # 这行代码的作用是限制模型参数的梯度范数，以防止在训练过程中出现梯度爆炸（gradient explosion）的问题。
        # 它会计算所有模型参数的梯度的范数（默认是L2范数，即欧几里得范数）。
        # 比较并裁剪：如果计算出的梯度范数大于给定的阈值（在这个例子中是1.0），那么它会按照比例缩小所有参数的梯度，使得它们的范数等于这个阈值。
        # 防止梯度太大对冲击权重参数，提高训练稳定性
        范数 = torch.nn.utils.clip_grad_norm_(模型.parameters(), 1.0)
        学习率 = 获得学习率(索引)
        for 参数组 in 优化器.param_groups:
            参数组['lr'] = 学习率
        优化器.step()
        # 用于同步 计算统一设备架构（CUDA） 事件
        # 需要注意的是，torch.cuda.synchronize() 会阻塞调用它的线程，直到所有 计算统一设备架构 操作都完成。
        # 因此，过度使用它可能会降低程序的性能，因为它会引入不必要的等待时间。
        # 通常，只有在确实需要同步操作时才应该使用它。在默认流（default stream）中，大多数 PyTorch 操作在返回之前都会自动同步，所以通常不需要显式调用 synchronize()。
        torch.cuda.synchronize()
        时间2 = time.time()
        间隔 = (时间2 - 时间1) * 1000
        每秒字词 = (训练时加载器.批 * 训练时加载器.序) / (时间2 - 时间1)
        print(
            f"第 {索引} 步，损失值：{损失值.item():.6f}，学习率：{学习率:.4e}，范数：{范数:.4f}，时间间隔：{间隔:.2f}毫秒，字词/秒：{每秒字词:.2f}")
