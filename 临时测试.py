import matplotlib.pyplot as plt
import numpy as np

from 训练 import 获得学习率


def 学习率变化曲线():
    x = np.linspace(1, 60, 60)
    y=list(map(获得学习率,x))
    plt.figure(1)
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    学习率变化曲线()
