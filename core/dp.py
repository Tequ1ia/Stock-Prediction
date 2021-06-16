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


class DataProcessor:
    def __init__(self, stock_data, feature_cols, label_col, day_num, split):
        df = pd.read_csv(stock_data)
        feature_df = df.get(feature_cols)
        label_df = df.get(label_col)

        self.feature_dfs = []
        self.label_dfs = []
        for i in range(len(feature_df) // day_num):
            self.feature_dfs.append((feature_df[i * day_num:(i + 1) * day_num]).reset_index(drop=True))
            self.label_dfs.append((label_df[i * day_num:(i + 1) * day_num]).reset_index(drop=True))

        self.data_len = len(self.feature_dfs[0])
        self.train_len = int(self.data_len * split)
        self.test_len = self.data_len - self.train_len

    def append_index_data(self, index_data, feature_cols):
        df = pd.read_csv(index_data)
        feature_df = df.get(feature_cols)
        for i in range(len(self.feature_dfs)):
            self.feature_dfs[i] = pd.concat([self.feature_dfs[i], feature_df], axis=1)

    def get_train_data(self, window_len):
        train_x = []
        train_y = []
        for i in range(self.train_len - window_len):
            for j in range(len(self.feature_dfs)):
                feature_window = (self.feature_dfs[j].values)[i: i + window_len + 1]
                feature_window = normalise_window(feature_window)
                # label_window = self.label_data[i + window_len + 1]
                # label_window = normalise_window(label_window)
                train_x.append(feature_window[:-1])
                train_y.append(feature_window[-1, 0])
        return np.array(train_x), np.array(train_y)

    def get_test_data(self, window_len):
        test_x = []
        test_y = []
        for i in range(self.test_len - window_len - 1):
            feature_window = (self.feature_dfs.values)[i + self.train_len: i + self.train_len + window_len + 1]
            feature_window = normalise_window(feature_window)
            # label_window = self.label_data[i + self.train_len + window_len + 1]
            # label_window = normalise_window(label_window)

            test_x.append(feature_window[:-1])
            test_y.append(feature_window[-1, 0])
        # 把最后一个窗口加入到test数据集中（没有对应的真实价格）
        feature_window = (self.feature_dfs.values)[-window_len:]
        feature_window = normalise_window(feature_window)
        label = None
        test_x.append(feature_window)
        test_y.append(label)
        return np.array(test_x), np.array(test_y)

# dp = DataProcessor('/home/tequlia/code/Stock-Prediction/data/data银行20192020.csv',['close', 'vol'], ['close'], 487, 0.85)
# dp.append_index_data('/home/tequlia/code/Stock-Prediction/data/datahs300.csv', ['close'])
# train_x, train_y = dp.get_train_data(50)
# print(train_x, train_y)