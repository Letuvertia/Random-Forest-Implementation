import requests
import numpy as np
import random


class Dataset(object):
    # for training (estimating E_in)
    train_data_url = "https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw6/hw6_train.dat"
    # for testing (estimating E_out)
    test_data_url = "https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw6/hw6_test.dat"
    
    def __init__(self):
        self.fetch_data()
        
    def fetch_data(self):
        print('Fetching Data from HTlin')
        self.train_x, self.train_y = self.text_to_numpy(requests.get(self.train_data_url).text)
        self.test_x, self.test_y = self.text_to_numpy(requests.get(self.test_data_url).text)
        self.training_set_size = self.train_x.shape[0]
        self.testing_set_size = self.test_x.shape[0]
        print('Training size: x:{}; y:{}'.format(self.train_x.shape, self.train_y.shape))
        print('Testing size: x:{}; y:{}'.format(self.test_x.shape, self.test_y.shape))
        print()
    

    @staticmethod
    def text_to_numpy(raw_data):
        all_x, all_y = [], []
        for data in raw_data.split('\n')[:-1]:
            float_data = list(map(float, data.split(' ')))
            all_x.append(float_data[:-1])
            all_y.append(float_data[-1])
        return np.asarray(all_x), np.asarray(all_y)

    def get_data(self, train_or_test):
        if train_or_test == 'train':
            return self.train_x, self.train_y
        elif train_or_test == 'test':
            return self.test_x, self.test_y


    @staticmethod
    def couple_bagging_for_training(data_x, data_y, random_seed=1126, sample_rate=0.5):
        random.seed(random_seed + random.randint(1, 10000))
        assert data_x.shape[0] == data_y.shape[0]
        data_size = data_x.shape[0]
        random_idx = [random.randrange(0, data_size) for _ in range(int(data_size*sample_rate))]
        return data_x[random_idx], data_y[random_idx], random_idx