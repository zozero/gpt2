"""
精品网页-教育版
FineWeb-Edu 数据集（用于自监督表示学习（Self-supervised Representation Learning，srs）预训练）
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
下载数据并进行分词处理，然后将数据碎片保存到磁盘上。
只需简单地运行：
$ python 精品网页.py
将会把碎片保存到本地目录 "教育版精品网页10亿"。

以下有一个中文的教育版数据集
https://huggingface.co/datasets/opencsg/chinese-fineweb-edu
维基百科
https://huggingface.co/datasets/wikimedia/wikipedia
"""
import os.path
import multiprocessing as 多进程

import numpy as np
import tiktoken
# pip install datasets
from datasets import load_dataset
#  pip install tqdm
from tqdm import tqdm

# gpt2，它内部已经包含中文，其他类型也是包含中文了，所以我们这里不需要词汇表
编码器 = tiktoken.get_encoding("gpt2")
# 所以这里必须是这个<|endoftext|>，去表示文本的结尾
文末 = 编码器._special_tokens['<|endoftext|>']


def 切分字词(文档):
    """
    对单个文档进行分词，并返回一个包含16位无符号整数类型的标记的多维数组。
    :param 文档:
    :return:
    """
    # 特别的标记分隔所有文档
    字词 = [文末]
    字词.extend(编码器.encode_ordinary(文档["text"]))
    # 多维数组
    字词维组 = np.array(字词)
    assert (0 <= 字词维组).all() and (字词维组 < 2 ** 16).all(), "对于16位无符号整数来说字词字典太大了。"
    # 16位无符号整型
    无整16字词维组 = 字词维组.astype(np.uint16)
    return 无整16字词维组


def 写入数据至文件(文件名, 字词多维数组):
    """

    :param 文件名:
    :param 字词多维数组:
    :return:
    """
    np.save(文件名, 字词多维数组)


# 我这里将下载中文的数据集，你必须能够访问外网。这些数据是在抱抱脸上
if __name__ == '__main__':
    本地目录 = "中文数据"
    远程名 = "train"
    # 每个分片有1亿个字词，总共有100个分片
    # 我使用的数据集共分了388个分片
    分片长度 = int(1e7)

    # 如果不存在这个目录就创建一个用于存放缓存的本地目录
    数据缓存目录 = os.path.join(os.path.dirname(__file__), 本地目录)
    os.makedirs(数据缓存目录, exist_ok=True)

    # 下载数据集
    # revision，修订版，参数来指定分支名。下载后的路径会被缓存在C:\Users\你的用户名\.cache\huggingface\hub\xxxxx\snapshots\xxxxxx\default\partial-train
    # 建议用迅雷下载后放入指定位置，这里opencsg/chinese-fineweb-edu我需要的数据没有在主杆中
    数据集 = load_dataset("opencsg/chinese-fineweb-edu", revision="refs/convert/parquet", split="train")

    # 对所有文档进行分词，并写入输出碎片，每个碎片包含 分片长度 个字词（最后一个碎片包含剩余部分）。
    进程数量 = max(1, os.cpu_count() // 2)
    with 多进程.Pool(进程数量) as 池子:
        分片索引 = 0
        # 全部字词的多维数组
        # 预分配缓冲区以保存当前分片
        全字词维组 = np.empty(分片长度, dtype=np.uint16)
        字词计数 = 0
        进度条 = None
        # .imap是 multiprocessing 模块中的一个功能，用于在多个进程中并行处理输入序列。
        # chunksize，这个参数指定了每次迭代传递给 切分字词 函数的数据块的大小。
        for 字词 in 池子.imap(切分字词, 数据集, chunksize=16):
            # 当前分片中有足够的空间来容纳新的标记吗？
            if 字词计数 + len(字词) < 分片长度:
                # 一个简单的追加字词到当前分片中
                全字词维组[字词计数:字词计数 + len(字词)] = 字词
                字词计数 += len(字词)

                # 更新进度条
                if 进度条 is None:
                    进度条 = tqdm(total=分片长度, unit="字词", desc=f"分片 {分片索引}")
                进度条.update(len(字词))
            else:
                # 写入当前分片并开始一个新的分片
                分割 = "验证" if 分片索引 == 0 else "训练"
                文件名 = os.path.join(数据缓存目录, f"中文精品教育_{分割}_{分片索引:06d}")
                # 将文档分割成适合当前分片的部分；剩余部分放入下一个分片。
                剩余 = 分片长度 - 字词计数
                进度条.update(剩余)
                全字词维组[字词计数:字词计数 + 剩余] = 字词[:剩余]
                写入数据至文件(文件名, 全字词维组)
                分片索引 += 1
                进度条 = None
                # 用当前文档的剩余部分填充下一个分片。
                全字词维组[0:len(字词) - 剩余] = 字词[剩余:]
                字词计数 = len(字词) - 剩余
        # 将任何剩余的字词写入作为最后一个分片。
        if 字词计数 != 0:
            分割 = "验证" if 分片索引 == 0 else "训练"
            文件名 = os.path.join(数据缓存目录, f"中文精品教育_{分割}_{分片索引:06d}")
            写入数据至文件(文件名, 全字词维组[:字词计数])
