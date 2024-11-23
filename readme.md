##  项目正在完善中
+ 2024年11月20日，当前只能从预训练模型中读取并使用。
+ 2024年11月21日，添加加了训练功能，大量的性能优化说明，但还没添加保存功能。增加了一层循环，训练时间大幅延长。
+ 2024年11月22日，添加了多显卡训练，但只适合在linux系统下使用，不过你可以在Windows下安装wsl2的Ubuntu，当然，你也需要有多张显卡。
+ 2024年11月22日，添加抱抱脸渠道下载数据的方式，并且载入进行训练。
+ 2024年11月23日，添加保存功能，暂时完成了代码，但可能还有很多不足，请见谅。
## 相关内容
+ 论文下载网站：https://arxiv.org/
+ 英文源码地址：https://github.com/openai/gpt-2
+ gpt-2源码：https://github.com/openai/gpt-2/tree/master
### 可能需要能够访问外网
+ 模型下载地址：https://huggingface.co/openai-community/gpt2
+ 原作者视频讲解地址：https://www.youtube.com/watch?v=l8pRSuU81PU&t=920s
+ 原作者视频源码地址：https://github.com/karpathy/build-nanogpt/tree/master
### 查看显卡信息

``` bash    
nvidia-smi
```
## 关于数据集
+ 需要访问外网，精品网页：大规模挖掘网络上最精细的文本数据[ FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
+ 需要访问外网，[精品网页：教育](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
## 优化运行速度相关论文
+ 闪光注意力：快速且内存高效的精确注意力机制与IO感知：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
+ 闪光注意力-2：通过更好的并行性和工作划分实现更快的注意力机制：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
+ 用于软最大的在线归一化计算：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
+ 计算统一设备架构（CUDA）中的内容大部分是按照2的幂次方来实现的，所以最好选择8、16、64、128这类的数字或者这些数字的倍数