model_name=PatchTST

seq_len=96
pred_len=12

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

patch_adaptive=1
adjsut=0
learning_rate=0.001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_${seq_len}${pred_len} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 5 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
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
  --dropout 0.1 \
  --patch_adaptive $patch_adaptive \
  --patience 5 \
  --adjust $adjsut \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_${seq_len}${pred_len} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 5 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
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
  --dropout 0.1 \
  --patch_adaptive $patch_adaptive \
  --patience 5 \
  --adjust $adjsut \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07_${seq_len}${pred_len} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 5 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
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
  --dropout 0.1 \
  --patch_adaptive $patch_adaptive \
  --patience 5 \
  --adjust $adjsut \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --use_multi_gpu \
  --devices 0,1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_${seq_len}${pred_len} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 5 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
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
  --dropout 0.1 \
  --patch_adaptive $patch_adaptive \
  --patience 5 \
  --adjust $adjsut \
  --itr 1