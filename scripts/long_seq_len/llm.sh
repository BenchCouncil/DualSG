#!/bin/bash

llm_models=("LLAMA2" "LLAMA3" "BERT" "DeepSeek" "QWEN")

model_name=DualSG
llama_layers=12
batch_size=128
d_model=32
d_ff=128
loss=MAE
pool_type=avg
top_k=10
adjsut=0
learning_rate=0.001

use_fullmodel_values=(0)
text_cd_values=(1)

for llm_model in "${llm_models[@]}"; do
  for use_fullmodel in "${use_fullmodel_values[@]}"; do
    for text_cd in "${text_cd_values[@]}"; do

      echo "=== Running model: $llm_model | use_fullmodel=$use_fullmodel | text_cd=$text_cd ==="

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --use_multi_gpu \
        --devices 0,1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_${llm_model}_96 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 720 \
        --label_len 0 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --n_heads 2 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size $batch_size \
        --llm_layers $llama_layers \
        --loss $loss \
        --learning_rate $learning_rate \
        --llm_model $llm_model \
        --pool_type $pool_type \
        --use_fullmodel $use_fullmodel \
        --text_cd $text_cd \
        --top_k $top_k \
        --dropout 0.3 \
        --patience 5 \
        --adjust $adjsut \
        --itr 1

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --use_multi_gpu \
        --devices 0,1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_${llm_model}_192 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 336 \
        --label_len 0 \
        --pred_len 192 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --n_heads 4 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size $batch_size \
        --llm_layers $llama_layers \
        --loss $loss \
        --learning_rate $learning_rate \
        --llm_model $llm_model \
        --pool_type $pool_type \
        --use_fullmodel $use_fullmodel \
        --text_cd $text_cd \
        --top_k $top_k \
        --dropout 0.3 \
        --patience 5 \
        --adjust $adjsut \
        --itr 1

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --use_multi_gpu \
        --devices 0,1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_${llm_model}_336 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 336 \
        --label_len 0 \
        --pred_len 336 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --n_heads 4 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size $batch_size \
        --llm_layers $llama_layers \
        --loss $loss \
        --learning_rate $learning_rate \
        --llm_model $llm_model \
        --pool_type $pool_type \
        --use_fullmodel $use_fullmodel \
        --text_cd $text_cd \
        --top_k $top_k \
        --dropout 0.3 \
        --patience 5 \
        --adjust $adjsut \
        --itr 1

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --use_multi_gpu \
        --devices 0,1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_${llm_model}_720 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 720 \
        --label_len 0 \
        --pred_len 720 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --n_heads 2 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size $batch_size \
        --llm_layers $llama_layers \
        --loss $loss \
        --learning_rate $learning_rate \
        --llm_model $llm_model \
        --pool_type $pool_type \
        --use_fullmodel $use_fullmodel \
        --text_cd $text_cd \
        --top_k $top_k \
        --dropout 0.3 \
        --patience 5 \
        --adjust $adjsut \
        --itr 1

    done
  done
done
