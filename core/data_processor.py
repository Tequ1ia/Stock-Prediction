import pandas as pd
import numpy as np


def normalise_window(window):
    normalised_window = []
    for col in range(window.shape[1]):
        normalised_col = [float(x) / float(window[0, col]) - 1 for x in window[:, col]]
        normalised_window.append(normalised_col)
    normalised_window = np.array(normalised_window).T
    return normalised_window


# TODO 读取整合相关股票指数信息
class DataProcessor:
    """
    前处理数据使其能够用于机器学习
    """

    # FIXME 初始化时cols【0】必须是价格，因为下面的代码都用window【-1，0】作为y
    def __init__(self, filename, split, cols):
        df = pd.read_csv(filename)
        self.data = df.get(cols).values
        ratio = list(map(lambda x: x / sum(split), split))
        self.train_len = int(len(self.data) * ratio[0])
        self.validation_len = int(len(self.data) * ratio[1])
        self.test_len = int(len(self.data) * ratio[2])

    def get_window(self, window_len, i, data_type):
        """
        从给定索引i处获取数据窗口
        :param window_len: 窗口长度
        :param i: 窗口索引
        :param data_type: 数据类型，train 或 validation 或 test
        :return: window
        """
        window = None
        if data_type == 'train':
            window = self.data[i: i + window_len]
        elif data_type == 'validation':
            window = self.data[self.train_len + i: self.train_len + i + window_len]
        elif data_type == 'test':
            window = self.data[self.validation_len + i: self.validation_len + i + window_len]
        return window

    def get_data(self, window_len, data_type):
        data_x = []
        data_y = []
        offset = 0
        if data_type == 'validation':
            offset = self.train_len
        elif data_type == 'test':
            offset = self.validation_len
        for i in range(offset, offset + self.train_len - window_len):
            window = self.get_window(window_len, i, 'train')
            window = normalise_window(window)
            data_x.append(window[:-1])
            data_y.append(window[-1, 0])
        return np.array(data_x), np.array(data_y)