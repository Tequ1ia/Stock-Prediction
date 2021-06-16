import numpy as np
from matplotlib import pyplot as plt
from core.model import Model
from core.data_processor import DataProcessor, denormalise_price


class Config:
    # 路径参数
    train_data_path = 'data/bank.csv'
    saved_model_path = 'saved_model/20210616-133459.h5'
    index_data_path = ['data/hs300.csv', 'data/sh.csv']
    model_save_dir = 'saved_model'

    # 数据处理参数
    stock_feature_columns = ['close', 'vol']  # 作为特征训练模型的列
    label_columns = ['close']  # 模型要预测的列
    day_num = 487
    index_feature_columns = ['close']
    index_num = 2
    look_back = 20  # 用过去多少天的数据进行预测
    input_shape = (look_back, len(stock_feature_columns) + index_num * len(index_feature_columns))
    split = 0.85  # 训练集：测试集

    # 训练参数
    batch_size = 64
    epochs = 20

    # 预测参数
    predict_len = 20


def plot_prediction(true_data, predicted_data):
    # for i in range(7):
    #     plt.plot(true_data[i * 53:(i + 1) * 53], label='True Data')
    #     plt.plot(predicted_data[i * 53 + 1:(i + 1) * 53] + [None], label='Prediction')
    #     plt.legend()
    #     plt.show()

    # for i in range(6):
    #     true_data[(i + 1) * 53] = [None]
    #     predicted_data[(i + 1) * 53] = [None]

    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_iterable(true_data, predicted_data):
    for i in range(len(predicted_data) // 20 - 1):
        predicted_data[(i + 1) * 20] = [None]
    for i in range(7):
        plt.plot(true_data[i * 53:(i + 1) * 53], label='True Data')
        plt.plot(predicted_data[i * 53:(i + 1) * 53], label='Prediction')
        plt.legend()
        plt.show()
    # plt.plot(true_data, label='True Data')
    # plt.plot(predicted_data, label='Prediction')
    # plt.legend()
    # plt.show()


def main():
    config = Config()
    x = eval(input('请选择模式 1.训练模式 2.预测模式\n'))
    if x == 1:
        dp = DataProcessor(stock_data=config.train_data_path, split=config.split,
                           feature_cols=config.stock_feature_columns, label_col=config.label_columns,
                           day_num=config.day_num)
        for i in range(config.index_num):
            dp.append_index_data(config.index_data_path[i], config.index_feature_columns)

        # 获取预处理后的数据用于训练模型
        train_x, train_y = dp.get_train_data(config.look_back)
        test_x, test_y = dp.get_test_data(config.look_back)

        model = Model(input_shape=config.input_shape)

        # 构建并训练模型
        model.build()
        history = model.train(train_x, train_y,
                              epochs=config.epochs,
                              batch_size=config.batch_size,
                              save_dir=config.model_save_dir)

        # 模型是否过拟合
        # history_dict = history.history
        # loss = history_dict['loss']
        # val_loss = history_dict['val_loss']
        # plt.plot(range(1, config.epochs + 1), loss, 'bo', label='Training loss')
        # plt.plot(range(1, config.epochs + 1), val_loss, 'b', label='Val Loss')
        # plt.legend()
        # plt.show()

        # # 在测试集上验证模型（逐点预测）
        # predicted = model.predict_point_by_point(test_x)
        # de_predicted = dp.denormalise_predicted(predicted)
        # true_data = dp.get_origin_price()
        #
        # plot_prediction(true_data, de_predicted)

        # 在测试集上验证模型（迭代预测）
        predicted = model.iterative_predict(test_x, config.predict_len)
        predicted = [i for j in predicted for i in j]
        de_predicted = dp.denormalise_predicted(predicted)
        true_data = dp.get_origin_price()
        plot_iterable(true_data, de_predicted)


    elif x == 2:
        # 获取数据
        dp = DataProcessor(stock_data='data/test1.csv', split=0,
                           feature_cols=config.stock_feature_columns, label_col=config.label_columns,
                           day_num=269)
        dp.append_index_data('data/test_index_1.csv', 'close')
        dp.append_index_data('data/test_index_2.csv', 'close')
        # dp = DataProcessor(stock_data=config.train_data_path, split=config.split,
        #                    feature_cols=config.stock_feature_columns, label_col=config.label_columns,
        #                    day_num=config.day_num)
        # for i in range(config.index_num):
        #     dp.append_index_data(config.index_data_path[i], config.index_feature_columns)

        # 获取预处理后的数据用于训练模型
        test_x, test_y = dp.get_test_data(config.look_back)

        model = Model(input_shape=config.input_shape)
        model.load_model(config.saved_model_path)

        predicted = model.predict_point_by_point(test_x)
        de_predicted = dp.denormalise_predicted(predicted)
        true_data = dp.get_origin_price()
        plot_prediction(true_data, de_predicted)

        # 在测试集上验证模型（迭代预测）
        predicted = model.iterative_predict(test_x, config.predict_len)
        predicted = [i for j in predicted for i in j]
        de_predicted = dp.denormalise_predicted(predicted)
        true_data = dp.get_origin_price()
        plot_iterable(true_data, de_predicted)
    else:
        import pandas as pd
        # df = pd.read_csv('data/bank.csv')
        # stock = df.get('close').values
        # for i in range(7):
        #     plt.plot(stock[i * 487: (i + 1) * 487 - 74], 'b')
        #     padding = [None] * 413
        #     plt.plot(np.hstack((np.array(padding),stock[(i + 1) * 487 - 74: (i + 1) * 487])), 'r')
        # plt.show()

        dp = DataProcessor(stock_data='data/test1.csv', split=0,
                           feature_cols=config.stock_feature_columns, label_col=config.label_columns,
                           day_num=269)
        dp.append_index_data('data/test_index_1.csv', 'close')
        dp.append_index_data('data/test_index_2.csv', 'close')
        dp.f1()


        # df1 = pd.read_csv('data/hs300.csv')
        # df2 = pd.read_csv('data/sh.csv')
        # i1 = df1.get('close').values
        # i2 = df2.get('close').values
        # plt.plot(i1, label='hs300')
        # plt.plot(i2, label='sh')
        # plt.legend()
        # plt.show()
if __name__ == '__main__':
    main()
