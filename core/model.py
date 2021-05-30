import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM
import os
import datetime as dt

class Model:
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading Model from {}'.format(filepath))
        self.model = load_model(filepath)

    def build(self):
        self.model.add(LSTM(100, input_shape=(49, 2), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')
        print('[Model] Model Compiled')

    def train(self, x, y, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('{} epochs, {} batch size'.format(epochs, batch_size))

        save_fname = os.path.join(save_dir, dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        self.model.save(save_fname)

        print('[Model] Training Completed, Model saved as {}'.format(save_fname))

    def predict_point_by_point(self, data):
        print('[Model] Predicting Point by Point')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence_multiple(self, data, window_size, prediction_len):
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[np.newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs