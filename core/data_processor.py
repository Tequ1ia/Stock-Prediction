import pandas as pd
import numpy as np


def normalise_window(window):
    normalised_window = []
    for col in range(window.shape[1]):
        normalised_col = [float(x) / float(window[0, col]) - 1 for x in window[:, col]]
        normalised_window.append(normalised_col)
    normalised_window = np.array(normalised_window).T
    return normalised_window

def denormalise_price(normalised_price, price0):
    return float((normalised_price + 1) * price0)

# Z-score normalization
# 效果奇差
def normalise_data(data):
    for col in range(data.shape[1]):
        data[:, col] = (data[:, col] - np.mean(data[:, col])) / np.std(data[:, col])
    return data







# TODO 读取整合相关股票指数信息
class DataProcessor:
    """
    前处理数据使其能够用于机器学习
    数据的最基本单元为 1 个 window : 一个形如 (look_back, feature_len) 的数组
    """

    def __init__(self, filepath, split, feature_cols, label_cols):
        """
        读取csv文件，获取需要的列中的数据，分割训练、测试集
        :param filepath: 读取的csv文件目录
        :param split: 0-1 之间的数，代表训练集、测试集的比例
        :param feature_cols: 特征列名称的列表
        :param label_cols: 标签列名称的列表
        """

        df = pd.read_csv(filepath)
        self.feature_data = df.get(feature_cols).values
        self.label_data = df.get(label_cols).values
        self.data_len = len(self.feature_data)
        self.train_len = int(self.data_len * split)
        self.test_len = self.data_len - self.train_len

        self.normalised_feature = np.zeros(shape=self.feature_data.shape)
        self.normalised_label = np.zeros(shape=self.label_data.shape)

        for col in range(self.feature_data.shape[1]):
            self.normalised_feature[:, col] = (self.feature_data[:, col] - np.mean(self.feature_data[:, col]))\
                                              / np.std(self.feature_data[:, col])
        for col in range(self.label_data.shape[1]):
            self.normalised_label[:, col] = (self.label_data[:, col] - np.mean(self.label_data[:, col]))\
                                            / np.std(self.label_data[:, col])
            self.mean = np.mean(self.label_data[:, col])
            self.std = np.std(self.label_data[:, col])



    def get_train_data(self, window_len):
        train_x = []
        train_y = []
        for i in range(self.train_len - window_len):
            feature_window = self.normalised_feature[i: i + window_len]
            # feature_window = normalise_window(feature_window)
            label_window = self.normalised_label[i + window_len + 1]
            # label_window = normalise_window(label_window)

            train_x.append(feature_window)
            train_y.append(label_window)
        return np.array(train_x), np.array(train_y)

    def get_test_data(self, window_len):
        test_x = []
        test_y = []
        origin_price = []
        for i in range(self.test_len - window_len - 1):
            feature_window = self.normalised_feature[i + self.train_len: i + self.train_len + window_len]
            # feature_window = normalise_window(feature_window)
            label_window = self.normalised_label[i + self.train_len + window_len + 1]
            # label_window = normalise_window(label_window)

            # origin_price.append(self.label_data[i + self.train_len + window_len + 1])
            test_x.append(feature_window)
            test_y.append(label_window)
        # 把最后一个窗口加入到test数据集中（没有对应的真实价格）
        feature_window = self.normalised_feature[-window_len:]
        # feature_window = normalise_window(feature_window)
        label_window = None
        # origin_price.append(None)
        test_x.append(feature_window)
        test_y.append(label_window)
        # return np.array(test_x), np.array(test_y), origin_price
        return np.array(test_x), np.array(test_y)


    def get_data_for_predict(self, window_len):
        pass

    def get_origin_test_data(self):
        return self.label_data[self.train_len + 1:]

    def denormalise(self, normalised_data):
        denormalised_price = []
        for x in normalised_data:
            denormalised_price.append(x * self.std + self.mean)
        return denormalised_price

# df = DataProcessor('../data/AAPL_210201_210601.csv', 0.85, ['Close', 'Volume'], ['Close'])
# train_x, train_y = df.get_train_data(30)
# print(train_x.shape, train_y.shape)