##  项目正在完善中
+ 2024年11月20日，当前只能从预训练模型中读取并使用。
+ 2024年11月21日，添加加了训练功能，大量的性能优化说明，但还没添加保存功能。

查看显卡信息
``` bash    
nvidia-smi

```

### 优化运行速度相关论文
+ 闪光注意力：快速且内存高效的精确注意力机制与IO感知：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
+ 闪光注意力-2：通过更好的并行性和工作划分实现更快的注意力机制：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
+ 用于软最大的在线归一化计算：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
+ 计算统一设备架构（CUDA）中的内容大部分是按照2的幂次方来实现的，所以最好选择8、16、64这类的数字