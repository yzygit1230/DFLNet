import torch
from util.helpers import get_criterion, initialize_metrics
import os
import logging
import json
import pandas as pd
from util.AverageMeter import RunningMetrics
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import DataLoader
from models.Base import dropblock_step
from util.common import check_dirs, CosOneCycle, ScaleInOutput
import argparse
from models.DFLNet import DFLNet
from util.transforms import train_transforms, test_transforms 
from glob import glob
import os
from dataset import BreastData

parser = argparse.ArgumentParser('DFLNet Train')
parser.add_argument("--network", type=str, default="DFLNet")
parser.add_argument("--inplanes", type=int, default=64)
parser.add_argument("--loss_function", type=str, default="hybrid", choices=['hybrid', 'bce', 'dice'])
parser.add_argument("--input_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=int, default=0.00035)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--pretrain_pth", type=str, default='models/DFLNet_pretrain.pth')

opt = parser.parse_args()
metadata = dict()

device = torch.device("cuda:0")
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=123)

best_metrics = {'precision_1': -1, 'recall_1': -1, 'F1_1': 0.75, 'Overall_Acc': -1,'Mean_IoU': -1}

train_path = "./datasets/BUSI-WHU/train/"
val_path = "./datasets/BUSI-WHU/valid/"

train_data = pd.DataFrame({
              'images': sorted(glob(train_path + "img" + "/*.bmp")),
              'masks': sorted(glob(train_path + "gt" + "/*.bmp"))})
val_data = pd.DataFrame({
              'images': sorted(glob(val_path + "img" + "/*.bmp")),
              'masks': sorted(glob(val_path + "gt" + "/*.bmp"))})

train_dataset = BreastData(df=train_data, transforms=train_transforms)
val_dataset = BreastData(df=val_data, transforms=test_transforms)

train_loader = DataLoader(dataset=train_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, num_workers=opt.num_workers, batch_size=1, shuffle=False)

save_path = check_dirs()

model = DFLNet(opt)
model = model.to(device)

criterion = get_criterion('hybrid')
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate ,weight_decay=0.0001)
scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs, up_rate=0)

logging.info('STARTING training')
total_step = -1
scale = ScaleInOutput(opt.input_size)

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    model.train()
    batch_iter = 0
    tbar = tqdm(train_loader)
    i=1
    train_running_metrics =  RunningMetrics(2)
    for batch_img, labels, paths in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
        batch_img= batch_img.float().to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        batch_img, batch_img2 = scale.scale_input((batch_img, batch_img))  
        result = model(batch_img)
        result = scale.scale_output(result)
        loss = criterion(result, labels, device)
        loss.backward()
        optimizer.step()
        result = result[-1]
        _, cd_preds = torch.max(result, 1)

        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.input_size**2)))
        train_running_metrics.update(labels.data.cpu().numpy(),cd_preds.data.cpu().numpy())

        del batch_img, labels
      
        
    scheduler.step()
    dropblock_step(model)
    mean_train_metrics = train_running_metrics.get_scores()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    model.eval()
    val_running_metrics =  RunningMetrics(2)
    with torch.no_grad():
        for batch_img ,labels, paths in val_loader:
            batch_img = batch_img.float().to(device)
            labels = labels.long().to(device)
            batch_img, batch_img2 = scale.scale_input((batch_img, batch_img))

            body = model(batch_img)

            body = scale.scale_output(body)
            body = body[-1]
            _, cd_preds = torch.max(body, 1)

            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.input_size**2)))

            val_running_metrics.update(labels.data.cpu().numpy(),cd_preds.data.cpu().numpy())
           
            del batch_img, labels

        
        mean_val_metrics = val_running_metrics.get_scores()
        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

        if mean_val_metrics['F1_1'] > best_metrics['F1_1']:
            logging.info('updata the model')
            metadata['validation_metrics'] = mean_val_metrics
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path+'/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)
            torch.save(model, save_path+'/checkpoint_epoch_'+str(epoch)+'.pt')
            best_metrics = mean_val_metrics