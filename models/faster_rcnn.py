import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterRCNN(nn.Module):

    def __init__(self,
                 pretrained=True,
                 progress=True,
                 num_classes=36,
                 pretrained_backbone=True,
                 trainable_backbone_layers=3,
                ) -> None:
        super().__init__()

        self.frcnn = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            progress=progress,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    def forward(self, images, targets):
        """
        Args:
            images: List[torch.Tensor]
            targets: List[Dict[str, torch.Tensor]]
        """
        return self.frcnn(images, targets)
