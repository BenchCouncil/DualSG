model_name=DualSG

seq_len=96
llama_layers=12
batch_size=32
d_model=32
d_ff=128
loss=MAE
heads=2
llm_model=GPT2
pool_type=avg

text_cd=1
use_fullmodel=0
top_k=10

adjsut=0
learning_rate=0.001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_${seq_len}_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --n_heads $heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate $learning_rate \
  --llm_model $llm_model \
  --pool_type $pool_type \
  --use_fullmodel $use_fullmodel\
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
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_${seq_len}_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --n_heads 4 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate $learning_rate \
  --llm_model $llm_model \
  --pool_type $pool_type \
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
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_${seq_len}_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --n_heads 4 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate $learning_rate \
  --llm_model $llm_model \
  --pool_type $pool_type \
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
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_${seq_len}_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --n_heads 2 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate $learning_rate \
  --llm_model $llm_model \
  --pool_type $pool_type \
  --top_k $top_k \
  --dropout 0.3 \
  --patience 5 \
  --adjust $adjsut \
  --itr 1