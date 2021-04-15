import pandas as pd
import os
from build_data.data_utils import sentences_proc
from utils import config
from utils.multi_proc_utils import parallelize


def split_train_eval(source_train_path, eval_rate=0.2, seed=1):
    '''
    将训练集划分为训练集和验证集
    :param train_data_path:
    :param eval_rate:
    :return:
    '''
    if os.path.exists(config.train_data_path) and os.path.exists(config.eval_data_path):
        train = pd.read_csv(config.train_data_path)
        eval = pd.read_csv(config.eval_data_path)

    else:
        source_data = pd.read_csv(source_train_path)
        print('total data size:{}'.format(len(source_data)))
        print('eval rate:{}'.format(eval_rate))

        eval = source_data.sample(frac=eval_rate, random_state=seed, axis=0)
        train = source_data.iloc[source_data.index.drop(eval.index)]

        train.to_csv(config.train_data_path, index=False, header=True)
        eval.to_csv(config.eval_data_path, index=False, header=True)

    return train, eval


def preprocess(train_data_path, test_data_path):
    if os.path.exists(config.train_seg_path) and \
        os.path.exists(config.train_seg_path) and \
        os.path.exists(config.merger_seg_path):
        train_df = pd.read_csv(config.train_seg_path, header=None)
        test_df = pd.read_csv(config.train_seg_path, header=None)

    else:
        # 1.加载数据
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        print('train data size {},test data size {}'.format(len(train_df), len(test_df)))
        # 2. 空值剔除
        train_df.dropna(subset=['Report'], inplace=True)
        test_df.dropna(subset=['Report'], inplace=True)

        train_df.fillna('', inplace=True)
        test_df.fillna('', inplace=True)
        # 3.多线程, 批量数据处理
        train_df = parallelize(train_df, sentences_proc)
        test_df = parallelize(test_df, sentences_proc)
        # 4. 合并训练测试集合
        train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
        test_df['merged'] = test_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
        merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
        print('train data size {},test data size {},merged_df data size {}'.format(len(train_df),
                                                                                   len(test_df),                                                                         len(merged_df)))
        # 5.保存处理好的 训练 测试集合
        train_df = train_df.drop(['merged'], axis=1)
        test_df = test_df.drop(['merged'], axis=1)
        train_df.to_csv(config.train_seg_path, index=False, header=False)
        test_df.to_csv(config.test_seg_path, index=False, header=False)
        # 6. 保存合并数据
        merged_df.to_csv(config.merger_seg_path, index=False, header=False)
    return train_df, test_df

if __name__ == '__main__':
    train_data, eval_data = split_train_eval(config.source_train_path)
    train_df, test_df = preprocess(config.train_data_path, config.eval_data_path)