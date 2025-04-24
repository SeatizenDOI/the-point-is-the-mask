import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth


    def dice_loss(self, probs, target):
        """
        Compute Dice Loss.

        Args:
            probs (torch.Tensor): Softmaxed predictions (probabilities), shape (B, C, H, W)
            target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        """

        # Convert target to one-hot encoding
        num_classes = probs.shape[1]  # Number of classes (C)
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # Shape: (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)

        # Compute Dice loss
        intersection = (probs * target_one_hot).sum(dim=(2, 3))  # Sum over spatial dims
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # Return Dice loss


    def forward(self, logits, target):
        """
        Compute the Dice loss.

        Args:
            logits (torch.Tensor): Raw logits from the model, shape (B, C, H, W)
            target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        """
        # Convert logits to softmax probabilities for Dice loss
        probs = torch.softmax(logits, dim=1)  # Convert to probabilities

        return self.dice_loss(probs, target)
