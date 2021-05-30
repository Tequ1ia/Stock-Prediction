from matplotlib import pyplot as plt
from core.model import Model
from core.data_processor import DataProcessor

# 数据预处理
dp = DataProcessor('./data/AAPL_080112_200112.csv', (85, 0, 15), ['Adj Close', 'Volume'])
data_x, data_y = dp.get_data(50, 'train')
test_x, test_y = dp.get_data(50, 'test')

# 训练模型
model = Model()
model.build()
model.train(data_x, data_y, epochs=10, batch_size=32, save_dir='./saved_model')

# 预测
predicted = model.predict_sequence_multiple(test_x, window_size=50, prediction_len=50)
plt.plot(test_y, label='True Data')
for i, data in enumerate(predicted):
    padding = [None for p in range(i * 50)]
    plt.plot(padding + data, label='Prediction')
    plt.legend()
plt.show()