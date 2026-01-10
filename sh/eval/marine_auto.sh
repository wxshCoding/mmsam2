# marine_auto.sh
# 功能：针对Marine数据集的自动化评估脚本。
# 作用：调用test.py进行推理生成（针对RMAS等子集），并调用eval.py计算评估指标。

CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "7.pth" \
--test_image_path "./data/Marine/valid/RMAS/images/" \
--test_gt_path "./data/Marine/valid/RMAS/masks/" \
--save_path "./data/Marine/valid/RMAS/preds/"

python eval.py \
--dataset_name "RMAS" \
--pred_path "./data/Marine/valid/RMAS/preds/" \
--gt_path "./data/Marine/valid/RMAS/masks/"


CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "7.pth" \
--test_image_path "./data/Marine/valid/MAS3K/images/" \
--test_gt_path "./data/Marine/valid/MAS3K/masks/" \
--save_path "./data/Marine/valid/MAS3K/preds/"

python eval.py \
--dataset_name "MAS3K" \
--pred_path "./data/Marine/valid/MAS3K/preds/" \
--gt_path "./data/Marine/valid/MAS3K/masks/"
