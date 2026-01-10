# camouflaged_auto.sh
# 功能：针对Camouflaged数据集进行自动化的测试与评估流程。
# 作用：
# 1. 依次对多个测试子集（如CAMO, CHAMELEON等）调用test.py生成预测掩码。
# 2. 对生成的掩码调用eval.py进行性能评估。

CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "1.pth" \
--test_image_path "./data/Camouflaged/valid/CAMO/images/" \
--test_gt_path "./data/Camouflaged/valid/CAMO/masks/" \
--save_path "./data/Camouflaged/valid/CAMO/preds/"

python eval.py \
--dataset_name "CAMO" \
--pred_path "./data/Camouflaged/valid/CAMO/preds/" \
--gt_path "./data/Camouflaged/valid/CAMO/masks/"


CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "1.pth" \
--test_image_path "./data/Camouflaged/valid/CHAMELEON/images/" \
--test_gt_path "./data/Camouflaged/valid/CHAMELEON/masks/" \
--save_path "./data/Camouflaged/valid/CHAMELEON/preds/"

python eval.py \
--dataset_name "CHAMELEON" \
--pred_path "./data/Camouflaged/valid/CHAMELEON/preds/" \
--gt_path "./data/Camouflaged/valid/CHAMELEON/masks/"

CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "1.pth" \
--test_image_path "./data/Camouflaged/valid/COD10K/images/" \
--test_gt_path "./data/Camouflaged/valid/COD10K/masks/" \
--save_path "./data/Camouflaged/valid/COD10K/preds/"

python eval.py \
--dataset_name "COD10K" \
--pred_path "./data/Camouflaged/valid/COD10K/preds/" \
--gt_path "./data/Camouflaged/valid/COD10K/masks/"


CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "1.pth" \
--test_image_path "./data/Camouflaged/valid/NC4K/images/" \
--test_gt_path "./data/Camouflaged/valid/NC4K/masks/" \
--save_path "./data/Camouflaged/valid/NC4K/preds/"

python eval.py \
--dataset_name "NC4K" \
--pred_path "./data/Camouflaged/valid/NC4K/preds/" \
--gt_path "./data/Camouflaged/valid/NC4K/masks/"


