#!/bin/bash

model_name=PatchTST
seq_len=96
llama_layers=12
batch_size=128
d_model=32
d_ff=128
loss=MAE
heads=2
llm_model=GPT2
pool_type=avg

text_cd=0
use_fullmodel=0
top_k=4

patch_adaptive=0
adjsut=1
learning_rate=0.001

# 定义参数数组
pred_lens=(96 192 336 720)
e_layers=(2 3 3 3)
n_heads=(2 2 4 2)
dropouts=(0.1 0.1 0.3 0.3)
prompts=(2 3 4 5)

for prompt in "${prompts[@]}"; do
    for i in "${!pred_lens[@]}"; do
        pred_len=${pred_lens[$i]}
        e_layer=${e_layers[$i]}
        n_head=${n_heads[$i]}
        dropout=${dropouts[$i]}

        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --use_multi_gpu \
          --devices 0,1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh1.csv \
          --model_id ETTh1_${seq_len}_${pred_len} \
          --model $model_name \
          --data ETTh1 \
          --features M \
          --seq_len $seq_len \
          --label_len 0 \
          --pred_len $pred_len \
          --e_layers $e_layer \
          --d_layers 1 \
          --factor 3 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --n_heads $n_head \
          --d_model $d_model \
          --d_ff $d_ff \
          --batch_size $batch_size \
          --llm_layers $llama_layers \
          --loss $loss \
          --learning_rate $learning_rate \
          --llm_model $llm_model \
          --pool_type $pool_type \
          --use_fullmodel $use_fullmodel \
          --top_k 3 \
          --dropout $dropout \
          --text_cd $text_cd \
          --patch_adaptive $patch_adaptive \
          --patience 3 \
          --adjust $adjsut \
          --prompt $prompt \
          --itr 1
    done
done
    