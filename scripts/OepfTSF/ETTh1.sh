model_name=OepfTSF
ma_type=ema
alpha=0.3
beta=0.3
seq_len=512

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

pred_len=96
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 1024 \
    --learning_rate 0.0002 \
    --lradj 'sigmoid'\
    --alpha 0.3 \
    --beta 0.3 \
    --patience 15 \
    --horizon_weighting atan \
    --con_cls_1 0.25 \
    --con_cls_2 0.25 \
    --con_time 0.3 \
    --dist_sup_weight 0.25 \
    --label_sigma_ratio 1.0 \
    --support_reg_weight 0.25 \
    --support_min_gap_ratio 0.25 \
    --support_smooth_weight 0.1 \
    --grad_clip 1.0 \
    --use_amp \
    --amp_dtype bf16 \
    --beta $beta > logs/$ma_type/$model_name'_ETTh1_'$seq_len'_'$pred_len.log
