import logging
from util.losses import hybrid_loss, bce_loss, dice_loss

logging.basicConfig(level=logging.INFO)

def initialize_metrics():
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics


def get_criterion(opt):
    if opt == 'hybrid':
        criterion = hybrid_loss
    elif opt == 'bce':
        criterion = bce_loss
    elif opt == 'dice':
        criterion = dice_loss
    

    return criterion



