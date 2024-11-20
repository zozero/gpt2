# pip install tiktoken 它提供了一种简单的方式来计数和分割文本为tokens。
import tiktoken
import torch


class 轻量数据加载器:
    def __init__(self, 批, 序):
        """
        这是一个自定义的数据加载器
        :param 批:
        :param 序:
        """
        self.批 = 批
        self.序 = 序

        # 在初始化时，将字词从磁盘中载入到内存
        with open("英文训练集.txt", 'r', encoding="utf8") as 文件:
            文本 = 文件.read()

        编码器 = tiktoken.get_encoding("gpt2")
        字词 = 编码器.encode(文本)
        self.字词 = torch.tensor(字词)
        print(f"载入了 {len(self.字词)} 字词。")
        print(f"每轮有 {len(self.字词) // (批 * 序)} 批数据。")

        self.当前位置 = 0

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
        self.当前位置 += 批 * 序
        # 如果加载下一个批次超出范围，则重置
        if self.当前位置 + (批 * 序 + 1) > len(self.字词):
            self.当前位置 = 0
        return x, y
