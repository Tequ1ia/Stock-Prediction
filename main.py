from matplotlib import pyplot as plt
from core.model import Model
from core.data_processor import DataProcessor, denormalise_price


class Config:
    # 数据处理参数
    feature_columns = ['Close', 'Volume']  # 作为特征训练模型的列
    label_columns = ['Close']  # 模型要预测的列
    look_back = 30  # 用过去多少天的数据进行预测
    input_shape = (look_back, len(feature_columns))
    split = 0.85  # 训练集：测试集

    # 路径参数
    train_data_path = 'data/AAPL_080101_210131.csv'
    predict_data_path = 'data/AAPL_210201_210601.csv'
    model_save_dir = 'saved_model'
    saved_model_path = 'saved_model/20210611-232616.h5'

    # 训练参数
    batch_size = 64
    epochs = 14

    # 预测参数
    predict_len = 20


def plot_prediction(true_data, predicted_data):
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def main():
    config = Config()
    while True:
        x = eval(input('请选择模式 1.训练模式 2.预测模式\n'))
        if x == 1:
            dp = DataProcessor(filepath=config.train_data_path, split=config.split,
                               feature_cols=config.feature_columns, label_cols=config.label_columns)

            # 获取预处理后的数据用于训练模型
            train_x, train_y = dp.get_train_data(config.look_back)
            test_x, test_y = dp.get_test_data(config.look_back)

            model = Model(input_shape=config.input_shape)

            # 训练模型
            model.build()
            history = model.train(train_x, train_y,
                                  epochs=config.epochs,
                                  batch_size=config.batch_size,
                                  save_dir=config.model_save_dir)

            history_dict = history.history
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            plt.plot(range(1, config.epochs + 1), loss, 'bo', label='Training loss')
            plt.plot(range(1, config.epochs + 1), val_loss, 'b', label='Val Loss')
            plt.legend()
            plt.show()

            predicted = model.predict_point_by_point(test_x)
            predicted = dp.denormalise(predicted)
            # predicted = [i for j in predicted for i in j]  # 将预测得到的[[1], [2], ..., [n]] 转换为 [1, 2, ..., n]
            true_data = dp.get_origin_test_data()
            true_data = [i for j in true_data for i in j]
            true_data = true_data[config.look_back:]
            plot_prediction(true_data, predicted)

        elif x == 2:
            # 获取数据
            dp = DataProcessor(filepath=config.predict_data_path, split=0,
                               feature_cols=config.feature_columns, label_cols=config.label_columns)
            test_x, test_y = dp.get_test_data(config.look_back)

            # 加载模型
            model = Model(config.input_shape)
            model.load_model(config.saved_model_path)

            # 迭代预测
            predicted = model.predict_point_by_point(test_x)
            predicted = dp.denormalise(predicted)
            # predicted = [i for j in predicted for i in j]  # 将预测得到的[[1], [2], ..., [n]] 转换为 [1, 2, ..., n]
            true_data = dp.get_origin_test_data()
            true_data = [i for j in true_data for i in j]
            true_data = true_data[config.look_back:]
            plot_prediction(true_data, predicted)

        elif x == 3:
            break


if __name__ == '__main__':
    main()
