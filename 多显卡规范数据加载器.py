 # pip install tiktoken 它提供了一种简单的方式来计数和分割文本为tokens。
import os

import numpy as np
import tiktoken
import torch


def 载入字词(文件名):
    """

    :param 文件名:
    :return:
    """
    # 字词的多维数组
    字词维组 = np.load(文件名)
    # 多维数组转张量
    维转张 = torch.tensor(字词维组, dtype=torch.long)

    return 维转张


class 多显卡轻量规范数据加载器:
    def __init__(self, 批, 序, 进程班号, 进程数量, 主进程与否, 分割):
        """
        这是一个自定义的数据加载器
        相较于多显卡轻量数据加载器而言，它的数据是从网络上下载的，这些数据都有着相应的规范。
        :param 批:
        :param 序:
        :param 进程班号:
        :param 进程数量:
        :param 主进程与否:
        :param 分割:
        """
        self.批 = 批
        self.序 = 序
        self.进程数量 = 进程数量
        self.进程班号 = 进程班号
        assert 分割 in {"训练", "验证"}

        # 获得分片文件名
        数据根 = "中文数据"
        分片列表 = os.listdir(数据根)
        分片列表 = [片 for 片 in 分片列表 if 分割 in 片]
        分片列表 = sorted(分片列表)
        分片列表 = [os.path.join(数据根, 片) for 片 in 分片列表]
        self.分片列表 = 分片列表
        assert len(分片列表) > 0, f"分片中没有找到分割的{分割}集。"
        if 主进程与否:
            print(f"找到{len(分片列表)}分割的{分割}集的分片。")

        self.重置()

    def 下一批(self):
        """

        :return:
        """
        批 = self.批
        序 = self.序
        缓存 = self.字词[self.当前位置:self.当前位置 + 批 * 序 + 1]
        # 输入
        x = 缓存[:-1].view(批, 序)
        # 目标
        y = 缓存[1:].view(批, 序)
        # 在字词张量中前进的位置
        self.当前位置 += 批 * 序 * self.进程数量
        # 如果加载下一个批次将超出界限，则切换到下一个分片。
        if self.当前位置 + (批 * 序 * self.进程数量 + 1) > len(self.字词):
            self.当前分片 = (self.当前分片 + 1) % len(self.分片列表)
            self.字词 = 载入字词(self.分片列表[self.当前分片])
            self.当前位置 = self.批 * self.序 * self.进程班号
        return x, y

    def 重置(self):
        """

        :return:
        """
        self.当前分片 = 0
        self.字词 = 载入字词(self.分片列表[self.当前分片])
        self.当前位置 = self.批 * self.序 * self.进程班号

if __name__ == '__main__':
    数据根 = "中文数据"
    分片列表 = os.listdir(数据根)
    分片列表 = [片 for 片 in 分片列表 if "训练" in 片]
    分片列表 = sorted(分片列表)
    分片列表 = [os.path.join(数据根, 片) for 片 in 分片列表]
    for 当前分片 in range(len(分片列表)):
        字词 = 载入字词(分片列表[当前分片])
        print(len(字词))
