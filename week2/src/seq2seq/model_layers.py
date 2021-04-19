# -*- coding:utf-8 -*-
# Created by LuoJie at 11/23/19

from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab, load_embedding_matrix
import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, enc_units, rnn_type='gru', vocab_size=20000, embedding_dim=100, embedding_matrix=None):
        ###################此处有作业################################
        """
        定义Embedding层，加载预训练的词向量
        请写出你的代码
        """
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)


        """
        定义单向的RNN、GRU、LSTM层
        请写出你的代码
        """
        self.rnn_type = rnn_type.lower()
        rnn_model = self.get_rnn_model(self.rnn_type)
        self.model = rnn_model(
            input_size=self.embedding.embedding_dim,
            hidden_size=enc_units,
            batch_first=True
        )
        ###################此处有作业################################
    def get_rnn_model(self, rnn_type):
        types = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN,
        }
        return types[rnn_type]

    def forward(self, x):
        x = self.embedding(x)
        if self.rnn_type == 'lstm':
            output, (h, c) = self.model(x)
            return output, h, c
        else:
            output, state = self.model(x)
            return output, state



class BahdanauAttention(nn.Module):
    def __init__(self,hidden_size, units):
        '''
        :param hidden_size: RNN隐藏层的维度
        :param units: 线性变换后的维度
        '''
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, units)
        self.W2 = nn.Linear(hidden_size, units)
        self.V = nn.Linear(units, 1)
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
    def __init__(self,hidden_size, attention_func):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_func = attention_func.lower()

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Unknown attention score function! Must be either dot, general or cancat'
            )

        if attention_func == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size)
        elif attention_func == 'concat':
            self.Wa = nn.Linear(hidden_size * 2, hidden_size)
            self.gelu = nn.GELU()
            self.Va = nn.Linear(hidden_size, 1)

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



class Decoder(nn.Module):
    def __init__(self, dec_units, att_model,
                 rnn_type='gru', vocab_size=20000, embedding_dim=100, embedding_matrix=None, **att_model_config):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.rnn_type = rnn_type
        ###################此处有作业################################
        """
        定义Embedding层，加载预训练的词向量
        请写出你的代码
        """
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
            vocab_size = embedding_matrix.shape[0]
            embedding_dim = embedding_matrix.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        """
        定义单向的RNN、GRU、LSTM层
        请写出你的代码
        """
        self.rnn_type = rnn_type.lower()
        rnn_model = self.get_rnn_model(self.rnn_type)
        self.model = rnn_model(
            input_size=embedding_dim,
            hidden_size=dec_units,
            batch_first=True
        )

        """
        定义最后的fc层，用于预测词的概率
        请写出你的代码
        """
        self.fc_layer = nn.Sequential(
            nn.Linear(dec_units, vocab_size),
            nn.Softmax(dim=-1)
        )

        """
        注意力机制
        请写出你的代码
        """
        self.attention =self.select_att(att_model, **att_model_config)
        ###################此处有作业################################
        # used for attention

    def get_rnn_model(self, rnn_type):
        types = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN,
        }
        return types[rnn_type]

    def select_att(self, att_model, **config):
        if att_model in ['bahdanau', 'bahdanau_bttention', 'bahdanauattention',
                         'additive', 'additive_attention', 'additiveattention']:
            try:
                hidden_size = config['hidden_size']
                units = config['units']
            except KeyError:
                raise (
                'BahdanauAttention模型缺少参数：hidden_size， units！ 请于创建Decoder时传入！'
            )

            return BahdanauAttention(hidden_size, units)
        elif att_model in ['luong', 'luong_bttention', 'luongttention',
                         'multiplicative', 'multiplicative_attention', 'multiplicativeattention']:
            try:
                hidden_size = config['hidden_size']
                attention_func = config['attention_func']
            except KeyError:
                raise (
                'BahdanauAttention模型缺少参数：hidden_size， attention_func！ 请于创建Decoder时传入！'
            )
            return LuongAttention(hidden_size, attention_func)

    def forward(self, x, dec_hidden, enc_output):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(dec_hidden, enc_output)

        # x shape == (batch_size, 1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = torch.cat([torch.unsqueeze(context_vector, dim=1), x], dim=-1)

        # passing the concatenated vector to the GRU
        if self.rnn_type == 'lstm':
            output, (h, c) = self.model(x) # output shape == (batch_size, sen_len, hidden_size
            output = output[:, -1]  # (batch_size, hidden_size)
            # output shape == (batch_size, vocab)
            prediction = self.fc(output)
            return prediction, (h, c), attention_weights
        else:
            output, state = self.model(x) # output shape == (batch_size, sen_len, hidden_size
            output = output[:, -1]  # (batch_size, hidden_size)
            # prediction shape == (batch_size, vocab)
            prediction = self.fc(output)
            return prediction, state, attention_weights


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 获得参数
    params = get_params()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 计算vocab size
    vocab_size = vocab.count
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    input_sequence_len = 250
    batch_size = 64
    embedding_dim = 500
    units = 1024

    # 编码器结构 embedding_matrix, enc_units, batch_sz
    encoder = Encoder(embedding_matrix=embedding_matrix, enc_units=units, rnn_type='gru')
    # example_input
    example_input_batch = torch.ones(size=(batch_size, input_sequence_len), dtype=torch.int32)
    # sample input
    # sample_hidden = encoder.initialize_hidden_state()
    # sample_hidden = torch.zeros(size=(batch_size, units))

    sample_output, sample_hidden = encoder(example_input_batch)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, hidden_dim) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, hidden_dim) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(embedding_matrix, units, batch_size)
    sample_decoder_output, state, attention_weights = decoder(tf.random.uniform((64, 1)),
                                                              sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
