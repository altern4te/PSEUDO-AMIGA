
import torch


class ConfusionMatrixTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, preds, targets):
        preds = preds.flatten().cpu()
        targets = targets.flatten().cpu()
        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]
        idx = targets * self.num_classes + preds
        binc = torch.bincount(idx, minlength=self.num_classes**2)
        self.cm += binc.reshape(self.num_classes, self.num_classes)

    def iou_per_class(self):
        intersection = torch.diag(self.cm).float()
        union = (self.cm.sum(1) + self.cm.sum(0) - torch.diag(self.cm)).float()
        return intersection / union.clamp(min=1)

    def mean_iou(self):
        return self.iou_per_class().mean()

    def dice_per_class(self):
        intersection = torch.diag(self.cm).float()
        denom = (self.cm.sum(1) + self.cm.sum(0)).float()
        return (2 * intersection) / denom.clamp(min=1)

    def pixel_accuracy(self):
        return torch.diag(self.cm).float().sum() / self.cm.sum().float().clamp(min=1)
