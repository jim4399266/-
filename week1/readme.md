已跑通build_data里的三个文件。

1.在utils.config里添加初始路径
```python
# 初始数据路径
source_train_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
source_test_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
```

2.在preprocess文件中添加split_train_eval()函数将原始始数据划分为训练集和验证集。  
```python
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
```



3.在preprocess文件中添加对已有数据的判断。  
```python
    if os.path.exists(config.train_seg_path) and \
        os.path.exists(config.train_seg_path) and \
        os.path.exists(config.merger_seg_path):
        train_df = pd.read_csv(config.train_seg_path, header=None)
        test_df = pd.read_csv(config.train_seg_path, header=None)
```
4.在data_loader文件中添加对已有模型的判断。 
```python
    if os.path.exists(config.save_wv_model_path):
        wv_model = Word2Vec.load(config.save_wv_model_path)
    else:
        wv_model = build_w2v()
``` 


最终词向量效果：
```python
model.wv.most_similar('奔驰')
[('宝马', 0.6357235312461853),
 ('沃尔沃', 0.5677694082260132),
 ('E300', 0.5450246334075928),
 ('c200', 0.537631630897522),
 ('揽胜', 0.5347270965576172),
 ('c180', 0.5294455289840698),
 ('E260', 0.5253739356994629),
 ('s300', 0.5230761170387268),
 ('CLS', 0.5215107202529907),
 ('奥迪', 0.5187931060791016)]
 
 
 model.wv.most_similar('机油')
 [('消耗量', 0.6386330723762512),
 ('10w40', 0.6269619464874268),
 ('烧一升', 0.6211923360824585),
 ('6000km', 0.6148136854171753),
 ('勤换', 0.6112556457519531),
 ('尺看', 0.608181357383728),
 ('金桶', 0.6072831749916077),
 ('矿物油', 0.6057610511779785),
 ('七千五百', 0.6055890321731567),
 ('用金嘉护', 0.6035946607589722)]
```
