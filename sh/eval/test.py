import argparse
import os
import torch
import imageio
import numpy as np
'''
sh/eval/test.py
功能：模型推理测试脚本。
主要作用：
1. 加载训练好的模型检查点。
2. 读取测试集图像。
3. 如果有Ground Truth，读取对应的掩码。
4. 运行模型推理生成预测掩码，并将结果保存到指定目录。
'''
import torch.nn.functional as F
# from SAM2UNet import SAM2UNet
from mmsam2 import MMSAM2
from dataset import TestDataset
import shutil


parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint", type=str, default='./best-154.pth', help="path to the checkpoint of sam2-unet")
# parser.add_argument("--checkpoint", type=str, default='./7.pth', help="path to the checkpoint of sam2-unet")
# parser.add_argument("--test_image_path", type=str, default='./data/Polyp/valid/CVC-ColonDB/images/', help="path to the image files for testing")
# parser.add_argument("--test_gt_path", type=str, default='./data/Polyp/valid/CVC-ColonDB/masks/', help="path to the mask files for testing")
# parser.add_argument("--save_path", type=str, default='./data/Polyp/valid/CVC-ColonDB/preds/', help="path to save the predicted masks")

parser.add_argument("--checkpoint", type=str, default='./7.pth', help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, default='./data/Marine/valid/RMAS/images/', help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, default='./data/Marine/valid/RMAS/masks/', help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, default='./data/Marine/valid/RMAS/preds/', help="path to save the predicted masks")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
model = MMSAM2().to(device)


checkpoint = torch.load(args.checkpoint)


# missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)



# if 'memory_bank_state' in checkpoint:
#     memory_state = checkpoint['memory_bank_state']
#     model.memory_bank.memories = memory_state['memories']
#     model.memory_bank.max_size = memory_state['max_size']
#     model.memory_bank.min_size = memory_state['min_size']
#     model.memory_bank.similarity_threshold = memory_state['similarity_threshold']
#     model.memory_bank.decay_factor = memory_state['decay_factor']
#     model.memory_bank.usage_counts = memory_state['usage_counts']
#     model.memory_bank.timestamps = memory_state['timestamps']
#     model.memory_bank.current_time = memory_state['current_time']


#     # 加载记忆库状态
# if 'memory_bank_state' in checkpoint:
#     memory_state = checkpoint['memory_bank_state']
#     model.memory_bank.memories = memory_state.get('memories', [])
#     model.memory_bank.max_size = memory_state.get('max_size', 12)
#     model.memory_bank.min_size = memory_state.get('min_size', 6)
#     model.memory_bank.similarity_threshold = memory_state.get('similarity_threshold', 0.85)
#     model.memory_bank.decay_factor = memory_state.get('decay_factor', 0.98)
#     model.memory_bank.usage_counts = memory_state.get('usage_counts', [])
#     model.memory_bank.timestamps = memory_state.get('timestamps', [])
#     model.memory_bank.current_time = memory_state.get('current_time', 0)
#     print("Memory bank state loaded successfully")

model.eval()
model.cuda()
if os.path.exists(args.save_path):
    shutil.rmtree(args.save_path)
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name,click = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        click = torch.from_numpy(click).float().to(device)
        res, _, _ = model(image,click.unsqueeze(0))
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        # If you want to binarize the prediction results, please uncomment the following three lines. 
        # Note that this action will affect the calculation of evaluation metrics.
        # lambda = 0.5
        # res[res >= int(255 * lambda)] = 255
        # res[res < int(255 * lambda)] = 0
        # print("Saving " + name)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)
