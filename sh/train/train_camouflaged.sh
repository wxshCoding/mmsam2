#!/bin/bash
# train_camouflaged.sh
# 功能：针对Camouflaged伪装目标检测数据集启动训练。
# 作用：设置环境变量（如CUDA设备），调用train.py脚本，并传入数据集路径、验证集列表、预训练模型路径、超参数等。

CUDA_VISIBLE_DEVICES="0" \
python train.py \
--exp_name "Camouflaged" \
--data_path "../data/Camouflaged/Camouflaged" \
--valid_list CAMO CHAMELEON COD10K NC4K \
--hiera_path "../data/sam2.pt" \
--save_path "./logs" \
--epoch 500 \
--valid_interval 1 \
--lr 0.001 \
--batch_size 12 \
--weight_decay 5e-4
