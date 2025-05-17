#!/bin/bash

# 设置使用的 CUDA 设备
# export CUDA_VISIBLE_DEVICES=0
# 定义模型名称
model_name=PatchTST
# 定义 patch_len 数组，包含要遍历的 patch_len 值
patch_lens=(16 24 32)
# 定义 seq_len 数组，包含要遍历的 seq_len 值
seq_lens=(512)

# 外层循环遍历 patch_len 数组
for patch_len in "${patch_lens[@]}"
do
    # 内层循环遍历 seq_len 数组
    for seq_len in "${seq_lens[@]}"
    do
        # 执行预测长度为 96 的实验
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --use_multi_gpu \
          --devices 0,1 \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id ECL_${seq_len}_96 \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len $seq_len \
          --label_len 48 \
          --pred_len 96 \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --batch_size 16 \
          --patch_len $patch_len \
          --itr 1

        # 执行预测长度为 192 的实验
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --use_multi_gpu \
          --devices 0,1 \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id ECL_${seq_len}_192 \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len $seq_len \
          --label_len 48 \
          --pred_len 192 \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --batch_size 16 \
          --patch_len $patch_len \
          --itr 1

        # 执行预测长度为 336 的实验
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --use_multi_gpu \
          --devices 0,1 \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id ECL_${seq_len}_336 \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len $seq_len \
          --label_len 48 \
          --pred_len 336 \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --batch_size 16 \
          --patch_len $patch_len \
          --itr 1

        # 执行预测长度为 720 的实验
        # python -u run.py \
        #   --task_name long_term_forecast \
        #   --is_training 1 \
        #   --use_multi_gpu \
        #   --devices 0,1 \
        #   --root_path ./dataset/electricity/ \
        #   --data_path electricity.csv \
        #   --model_id ECL_${seq_len}_720 \
        #   --model $model_name \
        #   --data custom \
        #   --features M \
        #   --seq_len $seq_len \
        #   --label_len 48 \
        #   --pred_len 720 \
        #   --e_layers 2 \
        #   --d_layers 1 \
        #   --factor 3 \
        #   --enc_in 321 \
        #   --dec_in 321 \
        #   --c_out 321 \
        #   --des 'Exp' \
        #   --batch_size 16 \
        #   --patch_len $patch_len \
        #   --itr 1
    done
done