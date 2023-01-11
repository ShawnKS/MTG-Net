export CUDA_VISIBLE_DEVICES=2


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_24 \
  --model Autoformer \
  --data ETTm2 \
  --dataset ETTm2\
  --batch_size 32\
  --train_epochs 10\
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --frac $2\
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_48 \
#   --model Autoformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_96 \
#   --model Autoformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_288 \
#   --model Autoformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 288 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_96_672 \
#   --model Autoformer \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 672 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1