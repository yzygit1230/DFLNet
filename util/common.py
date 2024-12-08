import os
import torch
import random
import numpy as np
import matplotlib
import torch.nn.functional as F
import seaborn
matplotlib.use('Agg')

def check_dirs():
    print("\n"+"-"*30+"Check Dirs"+"-"*30)
    if not os.path.exists('./runs/train'):
        os.makedirs('./runs/train')
      
    file_names = os.listdir('./runs/train')
    file_names = [int(i) for i in file_names] + [0]
    new_file_name = str(max(file_names) + 1)

    save_path = './runs/train/' + new_file_name
    os.mkdir(save_path)
    
    print("checkpoints & results are saved at: {}".format(save_path))

    return save_path


def compute_p_r_f1_miou_oa(tn_fp_fn_tps):   
    p, r, f1, miou, oa = [], [], [], [], []
    for tn_fp_fn_tp in tn_fp_fn_tps:
        tn, fp, fn, tp = tn_fp_fn_tp
        p_tmp = tp / (tp + fp)
        r_tmp = tp / (tp + fn)
        miou_tmp = 0.5 * tp / (tp + fp + fn) + 0.5 * tn / (tn + fp + fn)
        oa_tmp = (tp + tn) / (tp + tn + fp + fn)

        p.append(p_tmp)
        r.append(r_tmp)
        f1.append(2 * p_tmp * r_tmp / (r_tmp + p_tmp))
        miou.append(miou_tmp)
        oa.append(oa_tmp)

    return np.array(p), np.array(r), np.array(f1), np.array(miou), np.array(oa)

class CosOneCycle:  
    def __init__(self, optimizer, max_lr, epochs, min_lr=None, up_rate=0.3):  
        self.optimizer = optimizer

        self.max_lr = max_lr
        if min_lr is None:
            self.min_lr = max_lr / 10
        else:
            self.min_lr = min_lr
        self.final_lr = self.min_lr / 50

        self.new_lr = self.min_lr

        self.step_i = 0
        self.epochs = epochs
        self.up_rate = up_rate 
        assert up_rate < 0.5, "up_rate should be smaller than 0.5"

    def step(self):
        self.step_i += 1
        if self.step_i < (self.epochs*self.up_rate):
            self.new_lr = 0.5 * (self.max_lr - self.min_lr) * (
                        np.cos((self.step_i/(self.epochs*self.up_rate) + 1) * np.pi) + 1) + self.min_lr
        else:
            self.new_lr = 0.5 * (self.max_lr - self.final_lr) * (np.cos(
                ((self.step_i - self.epochs * self.up_rate) / (
                            self.epochs * (1 - self.up_rate))) * np.pi) + 1) + self.final_lr

        if len(self.optimizer.state_dict()['param_groups']) == 1:
            self.optimizer.param_groups[0]["lr"] = self.new_lr
        elif len(self.optimizer.state_dict()['param_groups']) == 2:  
            self.optimizer.param_groups[0]["lr"] = self.new_lr / 10
            self.optimizer.param_groups[1]["lr"] = self.new_lr
        else:
            raise Exception('Error. You need to add a new "elif". ')

    def plot_lr(self):
        all_lr = []
        for i in range(self.epochs):
            all_lr.append(self.new_lr)
            self.step()
        fig = seaborn.lineplot(x=range(self.epochs), y=all_lr)
        fig = fig.get_figure()
        fig.savefig('./lr_schedule.jpg', dpi=200)
        self.step_i = 0
        self.new_lr = self.min_lr

class ScaleInOutput:
    def __init__(self, input_size=512):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.output_size = None

    def scale_input(self, imgs: tuple):
        assert isinstance(imgs, tuple), "Please check the input type. It should be a 'tuple'."
        imgs = list(imgs)
        self.output_size = imgs[0].shape[2:]

        for i, img in enumerate(imgs):
            imgs[i] = F.interpolate(img, self.input_size, mode='bilinear', align_corners=True)

        return tuple(imgs)

    def scale_output(self, outs: tuple):
        if type(outs) in [torch.Tensor]:
            outs = (outs,)
        assert isinstance(outs, tuple), "Please check the input type. It should be a 'tuple'."
        outs = list(outs)

        assert self.output_size is not None, \
            "Please call 'scale_input' function firstly, to make sure 'output_size' is not None"

        for i, out in enumerate(outs):
            outs[i] = F.interpolate(out, self.output_size, mode='bilinear', align_corners=True)

        return tuple(outs)



