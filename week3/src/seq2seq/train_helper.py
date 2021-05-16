# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import os
# from src.pgn_tf2.batcher import batcher
from src.seq2seq.seq2seq_batcher import My_Dataloader

import time
from functools import partial
from src.utils import config
from tqdm import tqdm



def train_model(model, vocab, params, device='cpu'):
    epochs = params['epochs']
    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    # 获取vocab大小
    params['vocab_size'] = vocab.count
    # for p in model.parameters():
    #     print(p.size())
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=params['learning_rate'])

    train_dataloader = My_Dataloader(config.train_x_path, config.train_y_path,
                                     max_enc_len=params['max_enc_len'], max_dec_len=params['max_dec_len'],
                                     batch_size=params['batch_size'])
    train_steps_per_epoch = train_dataloader.steps_per_epoch

    val_dataloader = My_Dataloader(config.test_x_path, config.test_y_path,
                                     max_enc_len=params['max_enc_len'], max_dec_len=params['max_dec_len'],
                                     batch_size=params['batch_size'])
    val_steps_per_epoch = val_dataloader.steps_per_epoch

    print(f'total batches:{train_dataloader.steps_per_epoch}')
    for epoch in tqdm(range(1, epochs+1), desc='epoch', ncols=80):
        start = time.time()
        # enc_hidden = model.encoder.initialize_hidden_state()

        # total_loss = torch.tensor(0., dtype=torch.float, device=device)
        # running_loss = torch.tensor(0., dtype=torch.float, device=device)
        total_loss = 0.
        running_loss = 0.
        for (batch, (inputs, target)) in enumerate(train_dataloader.loader, start=1):
            # 训练模式，dropout层发生作用
            model.zero_grad()
            model.train()
            optimizer.zero_grad()
            batch_loss = train_step(model, inputs.to(device), target.to(device),
                                    loss_function=partial(loss_function, pad_index=pad_index))
            # 反向传播
            # with torch.autograd.set_detect_anomaly(True):
            batch_loss.backward()
            # 梯度截断
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()

            if batch % 50 == 0:
                print('\nEpoch {} Batch {} Loss {:.4f}'.format(epoch,
                                                             batch,
                                                             (total_loss - running_loss) / 50), end='')
                # pytorch中的tensor要使用clone()进行赋值，不然running_loss和total_loss会一起变化
                # 不是tensor可以直接赋值
                running_loss = total_loss
        # saving (checkpoint) the model every 2 epochs
        if (epoch) % 1 == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            if not os.path.exists(params['checkpoint_dir']):
                os.makedirs(params['checkpoint_dir'])
            ckpt_save_path = os.path.join(params['checkpoint_dir'], f'epoch{epoch}.bin')

            torch.save(checkpoint, ckpt_save_path)
            print('Saving checkpoint for epoch {} at {}'.format(epoch,
                                                                ckpt_save_path))

        valid_loss = evaluate(model, val_dataloader, val_steps_per_epoch,
                              loss_func=partial(loss_function, pad_index=pad_index),
                              device=device)

        print('Epoch {} Loss {:.4f}; val Loss {:.4f}'.format(
            epoch, total_loss / train_steps_per_epoch, valid_loss)
        )

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# 定义损失函数
def loss_function(real, pred, pad_index):
    '''
    :param real: shape (batch_size, sen_len - 1, vocab_size)
    :param pred: shape (batch_size, sen_len - 1)
    :param pad_index:
    :return:
    '''
    loss_object = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_index)
    real = real.type(torch.long)
    p = torch.argmax(pred, dim=-1)
    pred = pred.transpose(1, 2)  # (batch_size, vocab_size, sen_len - 1)
    loss_ = torch.mean(loss_object(pred, real))
    return loss_


def loss_function_(real, pred, pad_index):
    '''
    :param real: shape (batch_size, sen_len - 1, vocab_size)
    :param pred: shape (batch_size, sen_len - 1)
    :param pad_index:
    :return:
    '''
    loss_object = nn.CrossEntropyLoss(reduction='none')
    mask = torch.not_equal(real, pad_index)
    # 类型转换
    real = real.type(torch.long)
    p = torch.argmax(pred, dim=-1)

    # 巨坑无比！！！！！！pytorch的CrossEntropyLoss的输入不需要经过softmax
    # pytorch的CrossEntropyLoss在input是三维的时候，要求的shape是(batch_size, C, K)
    # 即input的最后一个维度和target的最后一个维度要相同
    # C是分类的数量，K是网络的维度，即sen_len。
    # 参考官方文档
    pred = pred.transpose(1, 2) # (batch_size, vocab_size, sen_len - 1)

    loss_ = loss_object(pred, real)
    loss_ *= mask
    return torch.mean(loss_)


def train_step(model, enc_inp, dec_target, loss_function=None):
    enc_output, enc_hidden = model.encoder(enc_inp)
    # 第一个隐藏层输入
    dec_hidden = enc_hidden.clone()

    # 逐个预测序列
    predictions, _ = model.teacher_decoder(dec_hidden, enc_output, dec_target)
    batch_loss = loss_function(dec_target[:, 1:], predictions)

    return batch_loss

@torch.no_grad()
def evaluate(model, val_dataloader, val_steps_per_epoch, loss_func, device='cpu'):
    model.zero_grad()
    model.eval()
    print('\nStarting evaluate ...')
    total_loss = 0.
    # enc_hidden = model.encoder.initialize_hidden_state()
    for (batch, (inputs, target)) in enumerate(val_dataloader.loader, start=1):
        batch_loss = train_step(model, inputs.to(device), target.to(device),
                                loss_function=loss_func)
        total_loss += batch_loss
    return total_loss / val_steps_per_epoch
