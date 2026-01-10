#!/bin/bash
# train_salient.sh
# 功能：针对Salient显著性检测数据集启动训练。
# 作用：设置环境变量，调用train.py脚本，配置数据集路径和多个验证集列表。

CUDA_VISIBLE_DEVICES="0" \
python train.py \
--exp_name "Salient" \
--data_path "../data/Salient/Salient" \
--valid_list DUT-OMRON DUTS-TE ECSSD HKU-IS PASCAL-S \
--hiera_path "../data/sam2.pt" \
--save_path "./logs" \
--epoch 500 \
--valid_interval 1 \
--lr 0.001 \
--batch_size 12 \
--weight_decay 5e-4
