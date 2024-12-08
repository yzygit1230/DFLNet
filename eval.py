import torch.utils.data
from statistics import mean
from tqdm import tqdm
from util.AverageMeter import RunningMetrics, compute_assd
from util.transforms import test_transforms 
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from glob import glob
from dataset import BreastData

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_path = "./datasets/BUSI-WHU/test/"
path = 'runs/train/1/checkpoint_epoch_1.pt'

test_data = pd.DataFrame({'images': sorted(glob(test_path + "img" + "/*.bmp")),
              'masks': sorted(glob(test_path + "gt" + "/*.bmp"))})
test_dataset = BreastData(df = test_data, transforms=test_transforms) 
test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=1, shuffle=False)

model = torch.load(path, map_location={'cuda:0':'cuda:0'})
model.eval()

running_metrics =  RunningMetrics(2)
c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
test_epoch_iou = []
np_assd = []
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img, labels, paths in tbar:
        batch_img = batch_img.float().to(dev)
        labels = labels.long().to(dev)
       
        cd_preds = model(batch_img)
        _, cd_preds = torch.max(cd_preds, 1)
        running_metrics.update(labels.data.cpu().numpy(),cd_preds.data.cpu().numpy())
        
        labels_np = labels.data.cpu().numpy()
        preds_np = cd_preds.data.cpu().numpy()
        assd = compute_assd(preds_np, labels_np)
        np_assd.append(assd)
        tp = np.sum((labels_np == 1) & (preds_np == 1))
        tn = np.sum((labels_np == 0) & (preds_np == 0))
        fp = np.sum((labels_np == 0) & (preds_np == 1))
        fn = np.sum((labels_np == 1) & (preds_np == 0))
        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp
       

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)
IOU0 = tn / (tn + fp + fn)
IOU1 = tp / (tp + fp + fn)
mIOU = (IOU0 + IOU1) / 2
OA = (tp + tn) / (tp + fp + tn + fn)
p0 = OA
pe = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / (tp + fp + tn + fn) ** 2
Kappa = (p0 - pe) / (1 - pe)

print('Precision: {}\nRecall: {}\nF1-Score: {} \nmIOU:{}'.format(P, R, F1, mIOU))
print('Kappa: {}\nASSD: {}'.format(Kappa, mean(np_assd)))

