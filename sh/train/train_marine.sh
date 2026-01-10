#!/bin/bash
# train_marine.sh
# 功能：针对Marine海洋生物检测数据集启动训练。
# 作用：设置环境变量，调用train.py脚本，配置数据集路径和验证列表（MAS3K, RMAS）。

CUDA_VISIBLE_DEVICES="0" \
python train.py \
--exp_name "Marine" \
--data_path "../data/Marine" \
--valid_list MAS3K RMAS \
--hiera_path "../data/sam2.pt" \
--save_path "./logs" \
--epoch 500 \
--valid_interval 1 \
--lr 0.001 \
--batch_size 12 \
--weight_decay 5e-4
