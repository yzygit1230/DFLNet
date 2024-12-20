from scipy.spatial import cKDTree
import numpy as np
eps = np.finfo(float).eps


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores, cls_iu, m_1 = cm2score(self.sum)
        scores.update(cls_iu)
        scores.update(m_1)
        return scores


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    acc_cls_ = tp / (sum_a1 + np.finfo(np.float32).eps)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*acc_cls_ * precision / (acc_cls_ + precision + np.finfo(np.float32).eps)
    # ---------------------------------------------------------------------- #
    # 2. Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    cls_iu = dict(zip(range(n_class), iu))



    return {'Overall_Acc': acc,
            'Mean_IoU': mean_iu}, cls_iu, \
           {
        'precision_1': precision[1],
        'recall_1': acc_cls_[1],
        'F1_1': F1[1],}


class RunningMetrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def __fast_hist(self, label_gt, label_pred):
        mask = (label_gt >= 0) & (label_gt < self.num_classes)
        hist = np.bincount(self.num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_gts, label_preds):
        for lt, lp in zip(label_gts, label_preds):
            self.confusion_matrix += self.__fast_hist(lt.flatten(), lp.flatten())

    def reset(self):

        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_cm(self):
        return self.confusion_matrix

    def get_scores(self):
        hist = self.confusion_matrix
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)

        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
        acc_cls_ = tp / (sum_a1 + np.finfo(np.float32).eps)
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)
        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        cls_iu = dict(zip(range(self.num_classes), iu))

        F1 = 2 * acc_cls_ * precision / (acc_cls_ + precision + np.finfo(np.float32).eps)

        scores = {'Overall_Acc': acc,
                'Mean_IoU': mean_iu}
        scores.update(cls_iu)
        scores.update({'precision_1': precision[1],
                       'recall_1': acc_cls_[1],
                       'F1_1': F1[1]})
        return scores

def compute_assd(segmentation, ground_truth):
    segmentation_points = np.argwhere(segmentation)
    ground_truth_points = np.argwhere(ground_truth)
    if segmentation_points.size == 0 or ground_truth_points.size == 0:
        return 0
    seg_tree = cKDTree(segmentation_points)
    gt_tree = cKDTree(ground_truth_points)
    seg_to_gt_distances, _ = seg_tree.query(ground_truth_points)
    gt_to_seg_distances, _ = gt_tree.query(segmentation_points)
    assd = (np.mean(seg_to_gt_distances) + np.mean(gt_to_seg_distances)) / 2
    return assd