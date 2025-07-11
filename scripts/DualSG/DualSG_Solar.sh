model_name=DualSG

seq_len=96
llama_layers=12
batch_size=256
d_model=256
d_ff=512
loss=MAE
heads=2
llm_model=GPT2
pool_type=avg

text_cd=1
use_fullmodel=0
top_k=6

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_${seq_len}_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --n_heads $heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate 0.02 \
  --llm_model $llm_model \
  --pool_type $pool_type \
  --use_fullmodel $use_fullmodel\
  --text_cd $text_cd \
  --top_k $top_k \
  --dropout 0.4 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_${seq_len}_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --n_heads 16 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate 0.02 \
  --llm_model $llm_model \
  --pool_type $pool_type \
  --top_k 4 \
  --dropout 0.4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_${seq_len}_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --n_heads 16 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate 0.001 \
  --llm_model $llm_model \
  --pool_type $pool_type \
  --top_k $top_k \
  --dropout 0.4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_${seq_len}_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --n_heads 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --loss $loss\
  --learning_rate 0.01 \
  --llm_model $llm_model \
  --pool_type $pool_type \
  --top_k $top_k \
  --dropout 0.4 \
  --itr 1