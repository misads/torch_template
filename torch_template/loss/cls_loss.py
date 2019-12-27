import torch.nn.functional as F


def cross_entropy(prediction, target):
    """
        prediction should be probabilities [N × C], target should be label index (0 ~ C-1)
        :param prediction:   size: [N, C]
        :param target:  size: [N]
        :return:
    """
    return F.cross_entropy(prediction, target)