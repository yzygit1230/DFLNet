from util.metrics import BCELoss, DiceLoss

def hybrid_loss(predictions, target,device):
    loss = 0
    bceloss = BCELoss(gamma=0, alpha=0)
    for prediction in predictions:
        bce = bceloss(prediction, target)
        dice = DiceLoss(prediction, target, device)
        loss += bce + dice

    return loss

def bce_loss(predictions, target, device):
    loss = 0
    bceloss = BCELoss(gamma=0, alpha=0)
    for prediction in predictions:
        bce = bceloss(prediction, target)
        loss += bce

    return loss

def dice_loss(predictions, target, device):
    loss = 0
    for prediction in predictions:
        dice = DiceLoss(prediction, target, device)
        loss += dice

    return loss