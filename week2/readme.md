## 利用pytorch改写了整个模型。  
完整项目代码（包括数据集、保存的模型）  
链接：https://pan.baidu.com/s/1KGwkO14ZAqXNX7JYgEYQfQ   
提取码：35kj   
  
### 1、先讲讲在用pytorch遇到的几个坑：   
（1）torch.tensor在赋值的时候需要使用clone()函数，如:
```python
a = torch.tensor([1,2,3])
b = a
```
这时候修改b，那么a也会跟着改变，所以：
```python
a = torch.tensor([1,2,3])
b = a.clone(）
```

(2)pytorch用来计算交叉熵损失的函数CrossEntropyLoss在方法上与tf有一些区别，这些区别导致我卡了很久，  
首先是CrossEntropyLoss使用了log_softmax来计算损失，所以传入的参数不需要进行softmax，不然会导致无法更新参数！！  
其次是CrossEntropyLoss对于输入的维度要求比较特殊，详细参考官方文档，我这里pred输入的是三维张量(batch_size, K, C)，  
target是二维张量(batch_size, K)， CrossEntropyLoss要求pred的shape是(batch_size, C, K)，即pred的最后一个维度和target的最后一个维度要相同  
**C是分类的数量，即vocab_size；K是网络的维度，即sen_len**。


### 2、模型测试过程
model_layers.py测试  
![model_layers.py测试](https://github.com/jim4399266/Text-Summarization/blob/main/week2/pic/model_layers.png)

seq2seq_model.py测试  
![seq2seq_model.py测试](https://github.com/jim4399266/Text-Summarization/blob/main/week2/pic/seq2seq_model.png)
 
seq2seq_batcher.py测试  
![seq2seq_batcher.py测试](https://github.com/jim4399266/Text-Summarization/blob/main/week2/pic/seq2seq_batcher.png)
  
### 3、模型训练过程
第一个epoch的loss已经很小了，完整信息查看log.txt   
![epoch1](https://github.com/jim4399266/Text-Summarization/blob/main/week2/pic/train_epoch1.png)  
  
