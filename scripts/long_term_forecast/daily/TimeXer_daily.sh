export CUDA_VISIBLE_DEVICES=0,1 # 这表示编号为 0 和编号为 1 的 GPU 是可见的。

model_name=TimeXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/daily/ \
  --data_path daily.csv \
  --model_id daily_96_96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --des 'Exp' \
  --use_multi_gpu \
  --use_gpu True \
  --itr 1
