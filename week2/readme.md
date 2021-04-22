利用pytorch改写了整个模型。  
先讲讲在用pytorch遇到的几个坑：
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
