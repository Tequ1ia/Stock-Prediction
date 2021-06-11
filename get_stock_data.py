import pandas as pd
import pandas_datareader.data as web
import datetime
import os


class StockData:
    def __init__(self, save_dir='./data'):
        self.save_dir = save_dir
        self.stock_df = None

    def get_stock_data(self, stock_name, start, end):
        print('[Data] Getting stock data of {} from {} to {}'.format(stock_name, start, end))
        self.stock_df = web.DataReader(stock_name, 'yahoo', start, end)
        save_fname = stock_name + '_' + start.strftime('%y%m%d') + '_' + end.strftime('%y%m%d') + '.csv'
        save_path = os.path.join(self.save_dir, save_fname)
        self.stock_df.to_csv(save_path)
        print('[Data] stock data has been saved to {}'.format(save_path))

    # TODO get multiple stock data
    # def get_stock_data(self, *stock_name, start, end):
    #     print('Getting stock data from yahoo')
    #     print('Stock list:{}'.format(*stock_name))
    #     print('From {} To {}'.format(start, end))
    #     self.stock_df = web.DataReader(*stock_name, 'yahoo', start, end)
    #     save_fname = self.save_dir.join(*stock_name)
    #     self.stock_df.to_csv(save_fname)


def main():
    stock_data = StockData()
    start = datetime.datetime(2008, 1, 1)
    end = datetime.datetime(2021, 1, 31)
    stock_name = 'AAPL'
    stock_data.get_stock_data(stock_name, start=start, end=end)


if __name__ == '__main__':
    main()
