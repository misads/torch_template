import torch
import torch.nn.functional as F


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        target = target
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt+1e-5) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt+1e-5)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    # pdb.set_trace()
    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def bce_loss(prediction, target):
    """
        prediction and target should have the same size.
        :param prediction:
        :param target:
        :return:
    """
    return F.binary_cross_entropy_with_logits(prediction, target)