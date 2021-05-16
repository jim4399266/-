# -*- coding:utf-8 -*-
# Created by LuoJie at 11/23/19

# from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab, load_embedding_matrix
from src.seq2seq.attentions import BahdanauAttention, LuongAttention

import torch.nn as nn
import torch
import numpy as np



class Encoder(nn.Module):
    def __init__(self, batch_size, enc_units, rnn_type='gru', vocab_size=20000, embedding_dim=100, embedding_matrix=None):
        super().__init__()
        ###################此处有作业################################
        """
        定义Embedding层，加载预训练的词向量
        请写出你的代码
        """
        self.batch_size = batch_size
        self.enc_units = enc_units
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
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
        self.gru = nn.GRU(
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

    def forward(self, x, h=None, c=None):
        x_len = torch.where(x > 0, 1, 0).sum(-1).cpu() # (batch_size)
        x = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            x, x_len, batch_first=True, enforce_sorted=False
        )
        # r = torch.rand([32,341,300]).cuda()
        if self.rnn_type == 'lstm':
            # 隐状态输入维度为(num_layers * num_directions, batch, hidden_size), 默认为全0
            output, (h, c) = self.model(packed_embedded)
            # 此处为(1, batch, hidden_size)，将其降维
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output, h.squeeze(), c.squeeze()
        else:
            output, state = self.gru(packed_embedded)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output, state.squeeze()

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_size, self.enc_units)).cuda()




class Decoder(nn.Module):
    def __init__(self, dec_units, att_model,
                 rnn_type='gru', vocab_size=20000, embedding_dim=100, embedding_matrix=None, **att_model_config):
        '''
        :param dec_units:
        :param att_model: 可以是str，也可以是Attention类
        :param rnn_type:
        :param vocab_size:
        :param embedding_dim:
        :param embedding_matrix:
        :param att_model_config:
        '''
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
            input_size=embedding_dim+dec_units,
            hidden_size=dec_units,
            batch_first=True
        )

        """
        定义最后的fc层，用于预测词的概率
        请写出你的代码
        """
        self.fc = nn.Sequential(
            nn.Linear(dec_units, vocab_size),
            # nn.Softmax(dim=-1)
        )

        """
        注意力机制
        请写出你的代码
        """
        if type(att_model) == str:
            # 如果传入的是str，则需要根据str和config创建Attention类
            self.attention =self.select_att(att_model, **att_model_config)
        else:
            self.attention = att_model
        ###################此处有作业################################
        # used for attention


    def forward(self, x, dec_hidden, enc_output, c_hidden=None):
        '''
        :param x: 输入的token_id
        :param dec_hidden: 上个时刻的h_hidden == (batch_size, hidden_size)
        :param enc_output: encoder的所有时刻的h_hidden == (batch_size, max_length, hidden_size)
        :param c_hidden:   如果是lstm，则需要上个时刻的c_hidden == (batch_size, hidden_size)
        :return:
        '''
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
            dec_hidden = dec_hidden.unsqueeze(0)  # (1, batch_size, hidden_size)
            c_hidden = c_hidden.unsqueeze(0)

            output, (h, c) = self.model(x, dec_hidden, c_hidden) # output shape == (batch_size, sen_len, hidden_size)
            output = output[:, -1]  # (batch_size, hidden_size)
            # output shape == (batch_size, vocab)
            prediction = self.fc(output)
            return prediction, (h.squeeze(), c.squeeze()), attention_weights
        else:
            dec_hidden = dec_hidden.unsqueeze(0)  # (1, batch_size, hidden_size)
            output, h = self.model(x, dec_hidden) # output == (batch_size, sen_len, hidden_size);  h == (1, batch_size, hidden_size)
            output = output[:, -1]  # (batch_size, hidden_size)
            # prediction shape == (batch_size, vocab)
            prediction = self.fc(output)
            return prediction, h.squeeze(), attention_weights



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
                hidden_size = config['att_hidden_size']
                units = config['att_units']
            except KeyError:
                raise (
                'BahdanauAttention模型缺少参数：att_hidden_size， att_units！ 请于创建Decoder时传入！'
            )

            return BahdanauAttention(hidden_size, units)
        elif att_model in ['luong', 'luong_bttention', 'luongttention',
                         'multiplicative', 'multiplicative_attention', 'multiplicativeattention']:
            try:
                hidden_size = config['att_hidden_size']
                attention_func = config['attention_func']
            except KeyError:
                raise (
                'BahdanauAttention模型缺少参数：att_hidden_size， attention_func！ 请于创建Decoder时传入！'
            )
            return LuongAttention(hidden_size, attention_func)


if __name__ == '__main__':
    # GPU资源配置, pytorch 不配置
    # config_gpu()
    # 获得参数
    params = get_params()

    device = torch.device('cuda:0' if torch.cuda.is_available() and params['device']=='cuda' else 'cpu')
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
    encoder = Encoder(embedding_matrix=embedding_matrix, enc_units=units, rnn_type='gru').to(device)
    # print(encoder.state_dict())
    # example_input
    example_input_batch = torch.ones(size=(batch_size, input_sequence_len), dtype=torch.int32, device=device)
    # sample input
    # sample_hidden = encoder.initialize_hidden_state()
    # sample_hidden = torch.zeros(size=(batch_size, units))

    sample_output, sample_hidden = encoder(example_input_batch)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, hidden_dim) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, hidden_dim) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(units, units).to(device)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(embedding_matrix=embedding_matrix, dec_units=units, att_model=attention_layer).to(device)
    sample_decoder_output, state, attention_weights = decoder(torch.ones(size=(batch_size, 1),
                                                                         dtype=torch.int32, device=device),
                                                              sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
