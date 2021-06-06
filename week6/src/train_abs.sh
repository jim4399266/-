#/root/anaconda3/envs/tf2/bin/python3.7 train.py \
#  -mode train \
#  -accum_count 5 \
#  -batch_size 300 \
#  -bert_data_path ../bert_data \
#  -dec_dropout 0.1 \
#  -log_file ../logs/abs_bert \
#  -lr 0.05 \
#  -model_path ../models/bert_abs \
#  -save_checkpoint_steps 5000 \
#  -seed 777 \
#  -sep_optim false \
#  -train_steps 100000 \
#  -use_bert_emb true \
#  -use_interval true \
#  -warmup_steps 8000 \
#  -visible_gpus 0,1 \
#  -max_pos 512 \
#  -report_every 50 \
#  -enc_hidden_size 512 \
#  -enc_layers 6 \
#  -enc_ff_size 2048 \
#  -enc_dropout 0.1 \
#  -dec_layers 6 \
#  -dec_hidden_size 512 \
#  -dec_ff_size 2048 \
#  -encoder baseline \
#  -task abs

/root/anaconda3/envs/tf2/bin/python3.7 train.py \
  -task abs \
  -mode train \
  -bert_data_path ../bert_data \
  -dec_dropout 0.2 \
  -model_path ../models/bert_abs \
  -sep_optim true \
  -lr_bert 0.002 \
  -lr_dec 0.2 \
  -save_checkpoint_steps 2000 \
  -batch_size 140 \
  -train_steps 200000 \
  -report_every 50 \
  -accum_count 5 \
  -use_bert_emb true \
  -use_interval true \
  -warmup_steps_bert 20000 \
  -warmup_steps_dec 10000 \
  -max_pos 512 \
  -visible_gpus 2 \
  -log_file ../logs/abs_bert
