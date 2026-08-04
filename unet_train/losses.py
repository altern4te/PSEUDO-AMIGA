
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def compute_class_weights(dataset, num_classes, max_weight=50.0, sample_limit=None):
    """Inverse-frequency class weights, estimated by scanning masks in `dataset`.

    Runs once before training starts (one pass over the training set). For very
    large datasets, pass sample_limit to subsample instead of scanning everything.
    Weights are clamped to max_weight to avoid numerical blowup if a class is
    extremely rare (e.g. weeds only appear in a handful of frames).
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    n = len(dataset)
    indices = range(n)
    if sample_limit and sample_limit < n:
        indices = torch.randperm(n)[:sample_limit].tolist()

    for i in indices:
        _, mask = dataset[i]
        binc = torch.bincount(mask.flatten(), minlength=num_classes)
        counts += binc.double()

    counts = torch.clamp(counts, min=1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = torch.clamp(weights, max=max_weight)
    return weights.float()
