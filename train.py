import os
import argparse
import torch
import torch.optim as opt
import torch.nn.functional as F
'''
train.py
功能：MMSAM2模型的训练脚本。
主要作用：
1. 解析命令行参数，设置实验名称、数据路径、预训练模型路径等。
2. 加载数据集（FullDataset_new）和模型（MMSAM2）。
3. 定义包括多尺度记忆库改进、ImageEncoder MFBFpnNeck替换、基于Mamba的解码器改进等模型训练流程。
4. 执行训练循环，计算损失，更新模型权重，并保存模型检查点。
'''
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset_new
from mmsam2 import MMSAM2
import _function as ff

# import random
# import numpy as np
# import tqdm

# 1、进行记忆库的改进改进：将编码后的记忆，进行多尺度变化，基于mamba架构思路，采用门控系统，对记忆库中特征进行与当前特征进行加权移动平均的方式进行组合。
# 2、ImageEncoder MFBFpnNeck替换掉FpnNeck
    #  - Captures multi-scale context
    #  - More powerful feature representation
    #  - Enhanced receptive field diversity
    #  - Potentially better performance for detecting objects at different scales
# 3、解码器改进： 基于mamba的思路，将经过记忆库的sam_mask_decoder预测内容与直接编码部分进行结合，采用门控系统对两部分进行加权组合，进行总的预测输出。

parser = argparse.ArgumentParser("mmsam2 training")

parser.add_argument("--exp_name", type=str, default="Polyp",  help="path to the sam2 pretrained hiera")
parser.add_argument("--data_path", type=str, default="./data/Polyp",  help="path to the image")
parser.add_argument("--valid_list", nargs='+',type=str, default=['CVC-300','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','Kvasir'], help="path to the image")

#parser.add_argument("--exp_name", type=str, default="Marine",  help="path to the sam2 pretrained hiera")
#parser.add_argument("--data_path", type=str, default="../data/Marine",   help="path to the image")
#parser.add_argument("--valid_list", nargs='+',type=str, default=['MAS3K','RMAS'], help="path to the image")

# parser.add_argument("--exp_name", type=str, default="Camouflaged",  help="path to the sam2 pretrained hiera")
# parser.add_argument("--data_path", type=str, default="../data/Camouflaged/Camouflaged",     help="path to the image")
# parser.add_argument("--valid_list", nargs='+',type=str, default=['CAMO','CHAMELEON','COD10K','NC4K'], help="path to the image")

#parser.add_argument("--exp_name", type=str, default="Salient_n", help="path to the sam2 pretrained hiera")
#parser.add_argument("--data_path", type=str, default="../data/Salient/Salient", help="path to the image")
#parser.add_argument("--valid_list", nargs='+',type=str, default=['DUT-OMRON','DUTS-TE','ECSSD','HKU-IS','PASCAL-S'], help="path to the image")



parser.add_argument("--hiera_path", type=str, default="./sam2.pt",  help="path to the sam2 pretrained hiera")
parser.add_argument('--save_path', type=str, default='./logs', help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=500,  help="training epochs")
parser.add_argument("--valid_interval", type=int, default=1,  help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main(args):    
    train_dataset = FullDataset_new(args.data_path, 352, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=False)
    device = torch.device("cuda")
    model = MMSAM2(args.hiera_path)
    model.to(device)

    MFBFpnNeck_params = (
                        []
                            + list(model.model.image_encoder.neck.mrb_convs.parameters())
                        )
    model_laste_params    =  (
                        []
                            + list(model.model.memory_attention.parameters())
                            + list(model.model.sam_mask_decoder.parameters())
                            + list(model.model.sam_prompt_encoder.parameters())
                        )
    
    model_params = list(
                            param for param in model.parameters() if id(param) not in {id(p) for p in MFBFpnNeck_params}
                        )
    
    model_params = list(
                            param for param in model_params if id(param) not in {id(p) for p in model_laste_params}
                       )


    optim = opt.AdamW([{"params":model_params,"initia_lr": args.lr},
                       {"params":MFBFpnNeck_params,"weight_decay": 1e-4},
                       {"params":model_laste_params,"weight_decay": 0},
                       ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    log_dir = ff.set_log_dir(os.path.join(args.save_path, args.exp_name))
    logger = ff.create_logger(log_dir['log'], phase='train')
  
    logger.info("-----------------------Training starts-----------------------")
    logger.info("Training with {} images".format(len(train_dataloader)))
    logger.info("args:{}".format(args))

    best_mdice, best_miou= {}, {}
    for _list  in args.valid_list:
        best_mdice[_list] = best_miou[_list] =  0.0
    for epoch in range(args.epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):
            x = batch['image']
            target = batch['label']
            point = batch['point']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x,point)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                logger.info("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            temp_mdice, temp_miou = {}, {}

            for _list  in args.valid_list:
                valid_dataset = FullDataset_new(args.data_path, 352, mode='valid',valid_file = _list)
                valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=False)
                temp_mdice[_list] = temp_miou[_list] = 0.0 

                mdice, miou = ff.valid_model_O(model, valid_dataloader, device)
                temp_mdice[_list] = mdice
                temp_miou[_list] =  miou
                logger.info(f'Epoch: {epoch+1}, Dataset: {_list}, mDice: {mdice:.4f}, mIou: {miou:.4f}')

            mdice_str = "_".join([f"{_list}:{temp_mdice[_list]:.4f}" for _list in args.valid_list])  
            miou_str = "_".join([f"{_list}:{temp_miou[_list]:.4f}" for _list in args.valid_list])  

            torch.save(model.state_dict(), os.path.join(log_dir['checkpoints'],f'mDice: {mdice_str},mIou: {miou_str}.pth'))
        
import random
import numpy as np
def seed_torch(seed=1024):

	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # seed_torch(1024)
    main(args)
