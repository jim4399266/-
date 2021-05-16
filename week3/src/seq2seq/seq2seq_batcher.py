# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19
from src.build_data.data_utils import load_dataset
import torch
import torch.utils.data as Data
from src.utils import config
from tqdm import tqdm


class My_Dataset(Data.Dataset):
    def __init__(self, x, y=None):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class My_Dataloader():
    def __init__(self, x_path, y_path, max_enc_len=200, max_dec_len=50, batch_size=8, sample_sum=None, shuffle=True):
        X, Y = load_dataset(x_path, y_path, max_enc_len, max_dec_len)
        self.data_size = len(X)
        if sample_sum:
            X = X[:sample_sum]
            Y = Y[:sample_sum]
        self.dataset = My_Dataset(X, Y)
        self.loader = Data.DataLoader(self.dataset, batch_size, shuffle)
        self.steps_per_epoch = len(X) // batch_size




'''def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=50, buffer_size=5, sample_sum=None):
    # 加载数据集
    train_X, train_Y = load_dataset(config.train_x_path, config.train_y_path,
                                    max_enc_len, max_dec_len)
    val_X, val_Y = load_dataset(config.test_x_path, config.test_y_path,
                                max_enc_len, max_dec_len)
    if sample_sum:
        train_X = train_X[:sample_sum]
        train_Y = train_Y[:sample_sum]
    print(f'total {len(train_Y)} examples ...')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X),
                                                                                   reshuffle_each_iteration=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_Y)).shuffle(len(val_X),
                                                                             reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size)
    train_steps_per_epoch = len(train_X) // batch_size
    val_steps_per_epoch = len(val_X) // batch_size
    return train_dataset, val_dataset, train_steps_per_epoch, val_steps_per_epoch'''



class Beam_Test_Dataloader():
    def __init__(self, x_path, y_path, beam_size=8, max_enc_len=200, max_dec_len=50, shuffle=False):
        X, Y = load_dataset(x_path, y_path, max_enc_len, max_dec_len)
        self.data_size = len(X)
        print(f'total {self.data_size} test examples ...')
        self.loader = self.generator(beam_size, X)

    def generator(self, beam_size, X):
        for row in tqdm(X, total=len(X), desc='Beam Search'):
            beam_search_data = torch.tensor([row for i in range(beam_size)])
            yield beam_search_data

'''def beam_test_batch_generator(beam_size, max_enc_len=200, max_dec_len=50):
    # 加载数据集
    test_X, _ = load_dataset(config.test_x_path, config.test_y_path,
                             max_enc_len, max_dec_len)
    print(f'total {len(test_X)} test examples ...')
    for row in tqdm(test_X, total=len(test_X), desc='Beam Search'):
        beam_search_data = tf.convert_to_tensor([row for i in range(beam_size)])
        yield beam_search_data'''


if __name__ == '__main__':
    task = 2
    if task == 1:
        my_loder = My_Dataloader(config.train_x_path, config.train_y_path)
        print('训练集大小:{}'.format(my_loder.data_size))
        print('steps_per_epoch:{}'.format(my_loder.steps_per_epoch))
        for item in my_loder.loader:
            print('训练集输入（batch_size, max_enc_len）:{}'.format(item[0].shape))
            print('训练集输入（batch_size, max_dec_len）:{}'.format(item[1].shape))
            # break
    elif task == 2:
        beam_loader = Beam_Test_Dataloader(config.test_x_path, config.test_y_path)
        print(next(beam_loader.loader))


