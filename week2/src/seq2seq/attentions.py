import torch.nn as nn
import torch
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, att_hidden_size, att_units):
        '''
        :param hidden_size: RNN隐藏层的维度
        :param units: 线性变换后的维度
        '''
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(att_hidden_size, att_units)
        self.W2 = nn.Linear(att_hidden_size, att_units)
        self.V = nn.Linear(att_units, 1)
        self.gelu = nn.GELU() #将tanh换为gelu

    def forward(self, query, values):
        '''
        在seq2seq模型中，St是后面的query向量，而编码过程的隐藏状态hi是values。
        :param query: 为上次的GRU隐藏层,  dec_hidden  shape == (batch_size, hidden size)
        :param values: 为编码器的编码结果enc_output   shape == (batch_size, sen_len, hidden_size)
        :return:
        '''
        # dec_hidden shape == (batch_size, hidden size)
        # dec_hidden_with_time_axis shape == (batch_size, 1, hidden size)

        # we are doing this to perform addition to calculate the score
        dec_hidden_with_time_axis = torch.unsqueeze(query, dim=1)

        # score shape == (batch_size, sen_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, sen_len, units)

        ###################此处有作业################################
        """
        计算注意力权重值，得到score
        请写出你的代码
        """
        score_dec = self.W1(dec_hidden_with_time_axis)   # (batch_size, 1, units)
        score_enc = self.W2(values)                     # (batch_size, sen_len, units)
        score = self.V(self.gelu(score_dec + score_enc)) # (batch_size, sen_len, 1)

        """
        归一化score，得到 attention_weights
        your code
        """
        attention_weights = F.softmax(score, dim=1)  #(batch_size, sen_len, 1)

        ###################此处有作业################################

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = torch.matmul(attention_weights.transpose(1,2), values)   # (batch_size, 1, hidden_size)
        context_vector = context_vector.squeeze() # (batch_size, hidden_size)

        return context_vector, attention_weights

class LuongAttention(nn.Module):
    def __init__(self,att_hidden_size, attention_func):
        super(LuongAttention, self).__init__()
        self.hidden_size = att_hidden_size
        self.attention_func = attention_func.lower()

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Unknown attention score function! Must be either dot, general or cancat'
            )

        if attention_func == 'general':
            self.Wa = nn.Linear(att_hidden_size, att_hidden_size)
        elif attention_func == 'concat':
            self.Wa = nn.Linear(att_hidden_size * 2, att_hidden_size)
            self.gelu = nn.GELU()
            self.Va = nn.Linear(att_hidden_size, 1)

    def select_func(self, attention_func):
        def dot_func(query, values):
            '''
            :param query:   (batch_size, 1, hidden size)
            :param values:  (batch_size, sen_len, hidden_size)
            :return: (batch_size, 1, sen_len)
            '''
            score = torch.matmul(query, values.transpose(1, 2))
            return score

        def concat_func(query, values):
            '''
            :param query:   (batch_size, 1, hidden size)
            :param values:  (batch_size, sen_len, hidden_size)
            :return: (batch_size, 1, sen_len)
            '''
            query = query.repeat(1, values.shape[1], 1)
            tmp = torch.cat([query, values], dim=-1)  # (batch_size, sen_len, hidden_size * 2)
            tmp = self.Wa(tmp)  # (batch_size, sen_len, hidden_size)
            score = self.Va(self.gelu(tmp))  # # (batch_size, sen_len, 1)
            return score.transpose(1, 2)

        def general_func(query, values):
            '''
            :param query:   (batch_size, 1, hidden size)
            :param values:  (batch_size, sen_len, hidden_size)
            :return: (batch_size, 1, sen_len)
            '''
            score = (torch.matmul(query, self.Wa(values).transpose(1, 2)))
            return score

        func_dict = {
            'general': general_func,
            'dot': dot_func,
            'concat': concat_func
        }
        return func_dict[attention_func]



    def forward(self, query, values):
        '''
        在seq2seq模型中，St是后面的query向量，而编码过程的隐藏状态hi是values。
        :param query: 为上次的GRU隐藏层,  dec_hidden  shape == (batch_size, hidden size)
        :param values: 为编码器的编码结果enc_output   shape == (batch_size, sen_len, hidden_size)
        :return:
        '''
        # dec_hidden shape == (batch_size, hidden size)
        # dec_hidden_with_time_axis shape == (batch_size, 1, hidden size)

        # we are doing this to perform addition to calculate the score
        dec_hidden_with_time_axis = torch.unsqueeze(query, dim=1)

        score = self.select_func(self.attention_func)(query, values)  # (batch_size, 1, sen_len)
        attention_weights = F.softmax(score, dim=-1)  # (batch_size, 1, sen_len)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = torch.matmul(attention_weights, values)   # (batch_size, 1, hidden_size)
        context_vector = context_vector.squeeze() # (batch_size, hidden_size)

        return context_vector, attention_weights
