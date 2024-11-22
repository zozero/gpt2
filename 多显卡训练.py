import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel as 分布式数据并行
import torch.distributed as 分布式

from 多显卡数据加载器 import 多显卡轻量数据加载器
from 模型 import 预生转换器配置, 预生转换器


class 训练配置:
    最大步数 = 50
    # 2**19，~50万，字词的数量，为了进一步对梯度操作计算
    所有字词的总数 = 524288
    #  2**19/(16*1024)=32，2**19/(4*1024)=128
    批长 = 4
    序长 = 1024


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

    from torch.distributed import init_process_group, destroy_process_group

    # 设置分布式数据并行（DDP，distributed data parallel）
    # 使用torchrun命令设置环境变量 RANK、LOCAL_RANK、WORLD_SIZE
    # RANK: 当前进程的全局排名。在一个分布式训练作业中，每个进程都会被分配一个唯一的全局排名，用于确定其在整个训练集群中的位置。
    # LOCAL_RANK: 当前进程在其所在节点（机器）上的局部排名。在一个具有多个GPU的节点上，每个GPU会被分配一个局部排名，用于确定进程应该使用哪个GPU。
    # WORLD_SIZE: 整个训练集群中的进程总数。这个变量告诉每个进程整个集群中有多少个进程参与训练。
    # 这是否是一个分布式数据并行（DDP）运行？
    分数并行 = int(os.environ.get('RANK', -1)) != -1

    if 分数并行:
        assert torch.cuda.is_available(), "为了使用分布式数据并行，我们需要计算统一设备架构（CUDA）。"
        # 它是火炬（PyTorch）分布式包中的一个函数，用于初始化一个进程组，这个进程组包含了参与分布式训练的所有进程。
        # 调用 初始化进程组（init_process_group）的目的是设置必要的通信基础设施，使得这些进程可以相互协调和交换信息。
        # backend: 这个参数指定了用于进程间通信的后端。
        # nccl：是英伟达集体通信库（NVIDIA Collective Communications Library）的缩写，它是专门为英伟达图形处理单元设计的，用于实现高效的集体通信操作，如全归约（all-reduce）、全聚集（all-gather）等。
        # windows不支持nccl，建议使用Windows的wsl2，然后安装Ubuntu系统，当然这一切确保电脑有两张显卡，请不要用sli连接你的显卡
        init_process_group(backend='nccl', )
        分数并行班号 = int(os.environ['RANK'])
        当前分数并行班号 = int(os.environ['LOCAL_RANK'])
        分布式世界数量 = int(os.environ['WORLD_SIZE'])
        设备 = f'cuda:{分数并行班号}'
        torch.cuda.set_device(设备)
        print("设备", 设备)
        # 此进程将负责日志记录、保存检查点等操作。
        主进程 = 分数并行班号 == 0
    else:
        print("原始，非分数并行运行")
        # 原始，非分数并行运行
        分数并行班号 = 0
        当前分数并行班号 = 0
        分布式世界数量 = 1
        主进程 = True
        # 尝试自动检测设备
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

    # 524288/(4*1024*2)=64
    assert 训练配置.所有字词的总数 % (
            训练配置.批长 * 训练配置.序长 * 分布式世界数量) == 0, "确保 所有字词的总数 能够被 训练配置.批长 x 训练配置.序长 x 当前世界数量 整除"
    梯度累积的步数 = 训练配置.所有字词的总数 // (训练配置.批长 * 训练配置.序长 * 分布式世界数量)
    if 主进程:
        print(f"所期望的字词的总数：{训练配置.所有字词的总数}")
        print(f"计算出的梯度累积的步数：{梯度累积的步数}")

    # 当批=4、序=1024时需要12G显存或内存，批最好是2的倍数或者幂次，奇数会导致性能下降
    训练时加载器 = 多显卡轻量数据加载器(批=训练配置.批长, 序=训练配置.序长, 进程班号=分数并行班号,
                                        进程数量=分布式世界数量)

    # 这里是将数据类型转换具体看文档链接：https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    # 这样有更高的效率。毕竟精度减少了。
    # 老旧的显卡可能不适配，如果不适配的可以直接注释
    # torch.set_float32_matmul_precision('high')

    # 之所有设置字量=50304是因为它可以被128整除，而128是2的7次幂。这样可以优化些许性能，但要看设备和计算量
    模型 = 预生转换器(预生转换器配置(字量=50304))
    模型.to(设备)
    if 分数并行:
        模型 = 分布式数据并行(模型, device_ids=[当前分数并行班号])

    # 始终包含“原始”未包装的模型
    原模型 = 模型.module if 分数并行 else 模型
    # 起初需要编译会花费不少时间，后续能够大幅加快训练速度
    # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#introduction-to-torch-compile
    # 但老旧型号显卡可能不适用。
    # 这是我的报错原因，Triton 编译器只支持 CUDA Capability 7.0 或更高的设备，而你的 GTX 1080 Ti 显卡的 CUDA Capability 是 6.1。
    # 模型 = torch.compile(模型)

    # 优化器 = torch.optim.AdamW(模型.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    优化器 = 原模型.配置优化器(权重衰减系数=0.1, 学习率=6e-4, 设备=设备)

    for 步数 in range(训练配置.最大步数):
        时间0 = time.time()
        # 梯度清零
        优化器.zero_grad()
        累计损失 = 0.0

        for 小步数 in range(梯度累积的步数):
            x, y = 训练时加载器.下一批()
            x, y = x.to(设备), y.to(设备)
            # 自动混合精度，https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            # 减少神经网络的运行消耗时间和内存占用。
            # 并不是所有的部分数据都转换成bfloat16类型，可以查看下面的文档链接
            # https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
            # 我尝试了一下反而更慢了，是因为1080ti显卡不支持bfloat16，所以我更换成了dtype=float16。
            with torch.autocast(device_type=设备, dtype=torch.float16):
                # 初始时每个字出现的概率是大约是1/50527，如果当前设置是50527的单词
                # 所以损失值是-ln(1/50527)大概值是10.8302，
                # 代码计算出来的损失值是11.0794
                逻辑果, 损失值 = 模型(x, y)

            # 防止梯度不断相加，所以需要将当前损失除以梯度累积的步数。
            # 这样损失被拆分成一小份一小份，后向传播成梯度后也是一小份一小份的相加
            损失值 = 损失值 / 梯度累积的步数
            # detach()，从计算图中分离出来，这里只需要值
            累计损失 += 损失值.detach()
            if 分数并行:
                # .require_backward_grad_sync：这是一个布尔型属性，用于指示是否需要在反向传播（backward pass）后同步梯度。
                # 在分布式训练中，尤其是在使用数据并行（data parallelism）时，不同设备（如GPU）上的模型副本会在每个批次（batch）后同步它们的梯度，以便进行参数更新。
                模型.require_backward_grad_sync = (小步数 == 梯度累积的步数 - 1)
            损失值.backward()

        if 分数并行:
            # all_reduce：它将所有进程中指定的张量（tensor）进行规约操作，并将结果广播到所有进程中。这意味着每个进程都会得到相同的输出张量。
            # op，操作，这是规约操作的类型。ReduceOp.AVG 表示执行的是平均操作。具体来说，这个操作会将所有进程中 累计损失 张量的值相加，然后除以进程的总数，从而得到全局平均损失。
            分布式.all_reduce(累计损失, op=分布式.ReduceOp.AVG)
        # 这行代码的作用是限制模型参数的梯度范数，以防止在训练过程中出现梯度爆炸（gradient explosion）的问题。
        # 它会计算所有模型参数的梯度的范数（默认是L2范数，即欧几里得范数）。
        # 比较并裁剪：如果计算出的梯度范数大于给定的阈值（在这个例子中是1.0），那么它会按照比例缩小所有参数的梯度，使得它们的范数等于这个阈值。
        # 防止梯度太大对冲击权重参数，提高训练稳定性
        范数 = torch.nn.utils.clip_grad_norm_(模型.parameters(), 1.0)
        学习率 = 获得学习率(步数)
        for 参数组 in 优化器.param_groups:
            参数组['lr'] = 学习率
        优化器.step()
        # 用于同步 计算统一设备架构（CUDA） 事件
        # 需要注意的是，torch.cuda.synchronize() 会阻塞调用它的线程，直到所有 计算统一设备架构 操作都完成。
        # 因此，过度使用它可能会降低程序的性能，因为它会引入不必要的等待时间。
        # 通常，只有在确实需要同步操作时才应该使用它。在默认流（default stream）中，大多数 PyTorch 操作在返回之前都会自动同步，所以通常不需要显式调用 synchronize()。
        torch.cuda.synchronize()
        时间1 = time.time()
        间隔 = 时间1 - 时间0
        处理的字词 = 训练时加载器.批 * 训练时加载器.序 * 梯度累积的步数 * 分布式世界数量
        每秒字词 = 处理的字词 / 间隔
        if 主进程:
            print(
                f"第 {步数} 步，损失值：{累计损失.item():.6f}，学习率：{学习率:.4e}，范数：{范数:.4f}，时间间隔：{间隔:.2f}秒，字词/秒：{每秒字词:.2f}")
    if 分数并行:
        # 关闭和清理与进程组相关的资源。
        destroy_process_group()
