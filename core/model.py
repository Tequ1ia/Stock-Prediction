import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import os
import datetime as dt


class Model:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.input_shape = input_shape
        self.look_back = input_shape[0]

    def load_model(self, filepath):
        print('[Model] Loading Model from {}'.format(filepath))
        self.model = load_model(filepath)

    def build(self):
        self.model.add(LSTM(100, input_shape=self.input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')
        self.model.summary()
        print('[Model] Model Compiled')

    def train(self, x, y, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('{} epochs, {} batch size'.format(epochs, batch_size))

        save_path = os.path.join(save_dir, dt.datetime.now().strftime('%Y%m%d-%H%M%S.h5'))
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)  # 确定超参数用
        # self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        self.model.save(save_path)

        print('[Model] Training Completed, Model saved as {}'.format(save_path))
        return history

    def predict_point_by_point(self, data):
        print('[Model] Predicting Point by Point')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    # def predict_sequence_multiple(self, data, window_size, prediction_len):
    #     prediction_seqs = []
    #     for i in range(int(len(data) / prediction_len)):
    #         curr_frame = data[i * prediction_len]
    #         predicted = []
    #         for j in range(prediction_len):
    #             predicted.append(self.model.predict(curr_frame[np.newaxis, :, :])[0, 0])
    #             curr_frame = curr_frame[1:]
    #             curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
    #         prediction_seqs.append(predicted)
    #     return prediction_seqs

    def iterative_predict(self, data, iteration_len):
        '''
        迭代预测：用预测出的价格更新窗口，进行新一轮预测，返回一个迭代预测序列
        :param data: 要进行预测的数据
        :param iteration_len: 迭代长度
        :return:
        '''
        print('[Model] Predicting Iteratively')
        print('Iteration length {}'.format(iteration_len))

        predicted_seqs = []
        for i in range(len(data) // iteration_len):
            iter_window = data[i * iteration_len]
            predicted = []
            for j in range(iteration_len):
                predicted.append(self.model.predict(iter_window[np.newaxis, :, :])[0, 0])

                # 迭代预测用窗口：删除窗口中第一个数据点 在尾部添加一个预测出的数据
                # 由于模型预测结果只有一个价格值，所以新数据点中其他值用0填充
                new_data_point = [predicted[-1]]
                new_data_point += [0] * (data.shape[2] - 1)
                iter_window = iter_window[1:]
                iter_window = np.insert(iter_window, data.shape[1] - 1, new_data_point, axis=0)

            predicted_seqs.append(predicted)
        return predicted_seqs


