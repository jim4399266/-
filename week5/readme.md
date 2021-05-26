layers.transformer [传送门](https://github.com/jim4399266/Text-Summarization/blob/main/week5/src/pgn_transformer_tf2/layers/transformer.py)  
encoder.self_attention_encoder [传送门](https://github.com/jim4399266/Text-Summarization/blob/main/week5/src/pgn_transformer_tf2/encoders/self_attention_encoder.py)  
完整项目文件（包括data和checkpoints）[地址](https://pan.baidu.com/s/1ObndBG4cGjH4x916RBIJFQ) 提取码：4m27   

一、训练  
![image](https://github.com/jim4399266/Text-Summarization/blob/main/week5/pic/train_wo_cov.png)  
由于没有使用coverage，因此源码里没有计算P_gen，所以只有avg_loss。  

二、预测  
![image](https://github.com/jim4399266/Text-Summarization/blob/main/week5/pic/test_decode.png)  
这是decoder的输出结果。

三、rouge    
![image](https://github.com/jim4399266/Text-Summarization/blob/main/week5/pic/rouge.png)   
经过rouge评估的结果。
