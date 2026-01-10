import os
from datetime import datetime
import time
import numpy as np
import torch
'''
_function.py
功能：提供训练和评估过程中的辅助函数。
主要作用：
1. 初始化评估指标（init_metrics），包括F-measure, S-measure, E-measure等。
2. 提供保存检查点、调整学习率、模型验证等辅助功能（如果包含）。
3. 封装与py_sod_metrics库的交互，用于计算显著性目标检测（SOD）相关的性能指标。
'''
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from dataset import TestDataset
from torch.utils.data import DataLoader
import imageio
import shutil
import logging
import cv2
import py_sod_metrics


def init_metrics():
    FM = py_sod_metrics.Fmeasure()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    MSIOU = py_sod_metrics.MSIoU()

    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
    overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)

    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
            "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
            "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
            "rec": py_sod_metrics.RecallHandler(**sample_gray),
            "fpr": py_sod_metrics.FPRHandler(**sample_gray),
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
            "dice": py_sod_metrics.DICEHandler(**sample_gray),
            "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
            "ber": py_sod_metrics.BERHandler(**sample_gray),
            "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
            "kappa": py_sod_metrics.KappaHandler(**sample_gray),
            "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
            "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
            "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
            "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
            "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
            "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
            "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
            "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
            "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
            "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
            "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
            "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
            "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
            "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
            "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
            "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
            "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
            "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
            "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
            "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
            "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
            "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
        }
    )
    return FM, WFM, SM, EM, MAE, FMv2


def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    device = pred.device
    gt = gt.to(device)

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        # TP = torch.Tensor([1],device=device)
        TP = torch.ones(1, device=device)

    Dice = 2 * TP / (2 * TP + FP + FN)
    IoU = TP / (TP + FP + FN)
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    Mae = ( FP+FN ) / (TP + FP + TN + FN)
    return Dice, IoU
    # return Dice, IoU, Acc, Mae


class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics


class EvalDataset:
    def __init__(self, pred_root, gt_root):
        self.preds = [pred_root + f for f in os.listdir(pred_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.preds = sorted(self.preds)
        self.gts = sorted(self.gts)
        self.size = len(self.preds)
        self.transform = transforms.ToTensor()
        self.index = 0

    def load_data(self):
        pred = self.transform(self.binary_loader(self.preds[self.index]))
        gt = self.transform(self.binary_loader(self.gts[self.index]))
        self.index += 1
        return pred, gt

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') # convert the gray picture 

def set_log_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    if not os.path.exists(os.path.join(save_path, timestamp)):
        os.makedirs(os.path.join(save_path, timestamp), exist_ok=True)
    prefix = os.path.join(save_path, timestamp)
    if not os.path.exists(os.path.join(prefix, "logs")):
        os.makedirs(os.path.join(prefix, "logs"), exist_ok=True)
    logs_path = os.path.join(prefix, "logs")
    if not os.path.exists(os.path.join(prefix, "checkpoints")):
        os.makedirs(os.path.join(prefix, "checkpoints"), exist_ok=True)
    checkpoints_path = os.path.join(prefix, "checkpoints")
    return {"log":logs_path,"checkpoints":checkpoints_path}


def pred_mask(model,valid_path,device):

    img  = os.path.join(valid_path, "image/")
    mask = os.path.join(valid_path, "mask/")
    pred = os.path.join(valid_path, "pred/")
    if os.path.exists(pred):
        shutil.rmtree(pred)
    os.makedirs(pred)
    loader = TestDataset(img, mask, 352)
    for i in range(loader.size):
        with torch.no_grad():
            image, gt, name = loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.to(device)
            res, _, _ = model(image)
            # fix: duplicate sigmoid
            # res = torch.sigmoid(res)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu()
            res = res.numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)
            # print("Saving " + name)
            imageio.imsave(os.path.join(pred, name[:-4] + ".png"), res)

    return mask,pred

    # img  = os.path.join(valid_path, "image/")
    # mask = os.path.join(valid_path, "mask/")
    # pred = os.path.join(valid_path, "pred/")
    # if os.path.exists(pred):
    #     shutil.rmtree(pred)
    # os.makedirs(pred)
    # valid_dataset = FullDataset(img, mask, 352, mode='valid')
    # valid_dataloader = DataLoader(valid_dataset, batch_size=12, shuffle=True, num_workers=8,drop_last=True)
    # model.eval()
    # # loader = TestDataset(img, mask, 352)
    # with torch.no_grad():
    #     for i, batch in enumerate(valid_dataloader):
    #         # image, gt, name = loader.load_data()
    #         x = batch["image"].to(device)
    #         y = batch["label"]
    #         name =  batch["name"]
    #         res, _, _ = model(x)
    #         out_size = (y.shape[2],y.shape[3]) 
    #         res = F.upsample(res, size=out_size, mode='bilinear', align_corners=False)
    #         res = res.sigmoid().data.cpu()
    #         res = res.numpy().squeeze()
    #         res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    #         res = (res * 255).astype(np.uint8)
    #         imageio.imsave(os.path.join(pred, name + ".png"), res)
    # return mask,pred

def valid_model(model, valid_path,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    mask_path = os.path.join(valid_path, "mask/")
    pred_path = os.path.join(valid_path, "pred/")
    mask_path , pred_path = pred_mask(model=model,valid_path=valid_path, device=device)
    valid_loader =  EvalDataset(mask_path, pred_path)
    model.eval()
    metrics = Metrics(['Dice', 'IoU', 'Sen', 'Spe', 'Acc'])
    for _ in range(valid_loader.size):
        with torch.no_grad():
            pred, gt = valid_loader.load_data()
            _Dice, _IoU, _Sen, _Spe, _Acc = evaluate(pred, gt)
            metrics.update(Dice=_Dice, IoU=_IoU, Sen=_Sen,
                        Spe=_Spe, Acc=_Acc)
    metrics_result = metrics.mean(valid_loader.size)
    return metrics_result['Dice']


def valid_model_new_contrast(model, valid_load, device):
    
    # metrics = Metrics(['Dice', 'IoU', 'Sen', 'Spe', 'Acc'])
    metrics = Metrics(['Dice', 'IoU', 'Acc', 'Mae'])
    model.eval()
    with torch.no_grad():
        # for  batch in valid_load:
        for i, batch in enumerate(valid_load):
            image, gt = batch['image'], batch['label']
            point = batch['point']
            image = image.to(device)
            gt = gt.to(device)
            # res = model(image)
            # res, _, _ = model(image,point)
            res, _, _ = model(image)
            # _Dice, _IoU, _Sen, _Spe, _Acc = evaluate(res, gt)
            _Dice, _IoU, _Acc, _Mae = evaluate(res, gt)
            metrics.update(Dice=_Dice, IoU=_IoU, Acc=_Acc, Mae=_Mae)

    metrics_result = metrics.mean(len(valid_load))
    # return metrics_result['Dice'], metrics_result['IoU'], metrics_result['Sen'], metrics_result['Spe'], metrics_result['Acc']
    return metrics_result['Dice'], metrics_result['IoU'], metrics_result['Acc'], metrics_result['Mae']
   
def valid_model_O(model, valid_load, device):
    
    # metrics = Metrics(['Dice', 'IoU', 'Sen', 'Spe', 'Acc'])
    metrics = Metrics(['Dice', 'IoU'])
    model.eval()
    with torch.no_grad():
        # for  batch in valid_load:
        for i, batch in enumerate(valid_load):
            image, gt = batch['image'], batch['label']
            point = batch['point']
            image = image.to(device)
            gt = gt.to(device)
            # res = model(image)
            res, _, _ = model(image,point)
            # res, _, _ = model(image)
            _Dice, _IoU = evaluate(res, gt)
            # _Dice, _IoU, _Acc, _Mae = evaluate(res, gt)
            # metrics.update(Dice=_Dice, IoU=_IoU, Acc=_Acc, Mae=_Mae)
            metrics.update(Dice=_Dice, IoU=_IoU)

    metrics_result = metrics.mean(len(valid_load))
    # return metrics_result['Dice'], metrics_result['IoU'], metrics_result['Sen'], metrics_result['Spe'], metrics_result['Acc']
    return metrics_result['Dice'], metrics_result['IoU']


def valid_model_new(model, valid_load,path, device):
    
    metrics = Metrics(['Dice', 'IoU', 'Sen', 'Spe', 'Acc'])
    model.eval()
    with torch.no_grad():
        # path_conccat = os.path.join(path,'test')
        path_preds = os.path.join(path,'test','preds')
        path_masks = os.path.join(path,'test','masks')
        
        if os.path.exists(path_preds):
            shutil.rmtree(path_preds)
        os.makedirs(path_preds, exist_ok=True)

        if os.path.exists(path_masks):
            shutil.rmtree(path_masks)
        os.makedirs(path_masks, exist_ok=True)

        # for  batch in valid_load:
        for i, batch in enumerate(valid_load):
            image, gt = batch['image'], batch['label']
            point = batch['point']
            image = image.to(device)
            gt_0 = np.asarray(gt, np.float32)
            gt = gt.to(device)
            res, _, _ = model(image,point)

          # Save predictions and ground truth as images
            for batch_idx in range(res.shape[0]):  # Iterate through batch
                # Save prediction: [1, 352, 352]
                single_pred = res[batch_idx].unsqueeze(0)  # Shape: [1, 352, 352]
                single_gt = gt_0[batch_idx].squeeze(0)  # Shape: [1, 352, 352] or [352, 352]

                res_save = F.upsample(single_pred, size=single_gt.shape, mode='bilinear', align_corners=False)
                res_save = res_save.sigmoid().data.cpu()
                res_save = res_save.numpy().squeeze()
                res_save = (res_save - res_save.min()) / (res_save.max() - res_save.min() + 1e-8)
                res_save = (res_save * 255).astype(np.uint8)
                gt_normalized = (single_gt * 255).astype(np.uint8)
                
                # Save images with descriptive names
                pred_name = f"batch_{i:03d}_sample_{batch_idx:02d}.png"
                gt_name = f"batch_{i:03d}_sample_{batch_idx:02d}.png"
                
                imageio.imsave(os.path.join(path_preds, pred_name), res_save)
                imageio.imsave(os.path.join(path_masks, gt_name), gt_normalized)
            _Dice, _IoU, _Sen, _Spe, _Acc = evaluate(res, gt)
            metrics.update(Dice=_Dice, IoU=_IoU, Sen=_Sen, Spe=_Spe, Acc=_Acc)
        metrics_result = metrics.mean(len(valid_load))
    


        mask_name_list = sorted(os.listdir(path_preds))
        FM, WFM, SM, EM, MAE, FMv2 = init_metrics()        
        for i, mask_name in enumerate(mask_name_list):
            mask_path = os.path.join(path_preds, mask_name)
            pred_path = os.path.join(path_masks, mask_name[:-4] + '.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            MAE.step(pred=pred, gt=mask)
            FMv2.step(pred=pred, gt=mask)
        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = MAE.get_results()["mae"]
        fmv2 = FMv2.get_results()


        curr_results = {
            "meandice": fmv2["dice"]["dynamic"].mean(),
            "meaniou": fmv2["iou"]["dynamic"].mean(),
            'Smeasure': sm,
            "wFmeasure": wfm,  # For Marine Animal Segmentation
            "adpFm": fm["adp"], # For Camouflaged Object Detection
            "meanEm": em["curve"].mean(),
            "MAE": mae,
        }
        return  curr_results['meandice'],curr_results['meaniou'],curr_results['Smeasure'],curr_results['wFmeasure'],curr_results['adpFm'],curr_results['meanEm'],curr_results['MAE'] ,metrics_result  
    
def create_logger(log_dir, phase='train'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    file = logging.FileHandler(filename=final_log_file)
    # logging.getLogger('').addHandler(console)
    logging.getLogger('').addHandler(file)
    return logger



def test_logs():
    # Example usage
    path = './test'
    set_log_dir(path)
    print(set_log_dir(path))

if __name__ == "__main__":
    pass

    # from SAM2UNet import SAM2UNet
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SAM2UNet().to(device)
    # model.load_state_dict(torch.load("./best.pth"), strict=True)
    # model.eval()
    # model.cuda()


    # # valid_dataset = FullDataset("./data/valid/image/", "./data/valid/mask/", 352, mode='valid')
    # valid_dataset = FullDataset_new("./data/valid/image/", "./data/valid/mask/", 352, mode='valid')
    # # valid_dataset = TestDataset("./data/valid/image/", "./data/valid/mask/", 352)
    # valid_dataloader = DataLoader(
    #     valid_dataset, 
    #     batch_size=5, 
    #     shuffle=True,  # 设置为False保证顺序一致
    #     num_workers=1,
    #     drop_last=False  # 设置为False确保所有样本都被评估
    # )

    # # aa  =  valid_model_new(model=model,valid_load = valid_dataloader,device=device)
    # # bb  =  valid_model(model=model,valid_path = "./data/valid",device=device)

    # # print(aa)
    # # print(bb)

    # for _ in range(5):
    #     aa  =  valid_model_new(model=model,valid_load = valid_dataloader,device=device)
    #     print(aa)
    # bb  =  valid_model(model=model,valid_path = "./data/valid",device=device)
    # print(bb)
    # 0.919026024201337
    # 0.920538057461381
    # pass
    # test_logs()