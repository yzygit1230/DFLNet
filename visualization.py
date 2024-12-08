import torch.utils.data
from dataset import BreastData
from tqdm import tqdm
from util.transforms import test_transforms 
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from glob import glob
import cv2
import os

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
full_to_colour = {1: (255, 255, 255), 2: (0, 0, 0), 3: (0, 0, 255), 4: (0, 255, 0)} 

test_path = "./datasets/BUSI/test/"
path = 'runs/train/1/checkpoint_epoch_1.pt'
base_path = './output_imgs/'

model = torch.load(path, map_location={'cuda:0':'cuda:0'})

test_data = pd.DataFrame({'images': sorted(glob(test_path + "img" + "/*.bmp")),
              'masks': sorted(glob(test_path + "gt" + "/*.bmp"))})


if not os.path.exists(base_path):
    os.mkdir(base_path)

test_dataset = BreastData(df=test_data, transforms=test_transforms)
test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img, labels, path in tbar:
        batch_img = batch_img.float().to(dev)
        labels = labels.long().to(dev)

        body = model(batch_img)
        _, cd_preds = torch.max(body, 1)

        labels_np = labels.data.cpu().numpy()
        preds_np = cd_preds.data.cpu().numpy()

        b, _, h, w = batch_img.size()
        tp = np.array((labels_np == 1) & (preds_np == 1)).astype(np.int8)
        tn = np.array((labels_np == 0) & (preds_np == 0)).astype(np.int8)
        fp = np.array((labels_np == 0) & (preds_np == 1)).astype(np.int8)
        fn = np.array((labels_np == 1) & (preds_np == 0)).astype(np.int8)
        img = tp * 1 + tn * 2 + fp * 3 + fn * 4

        img_colour = torch.zeros(b, 3, h, w)
        img_r = torch.zeros(1, h, w)
        img_g = torch.zeros(1, h, w)
        img_b = torch.zeros(1, h, w)
        img = img.reshape(1, 1, h, -1).squeeze(0)

        for k, v in full_to_colour.items():
            img_r[(img == k)] = v[0]
            img_g[(img == k)] = v[1]
            img_b[(img == k)] = v[2]
            img_colour = torch.cat((img_r, img_g, img_b), 0)
            img_colour = img_colour.data.cpu().numpy()
            img_colour = np.transpose(img_colour, (1, 2, 0))

        img_paths = str(path)
        img_paths = img_paths[2:]
        img_paths = img_paths[:-3]
        file_path = base_path + img_paths
        cv2.imwrite(file_path, img_colour.astype(np.uint8))



