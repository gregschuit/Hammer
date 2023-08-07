import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import densenet121


class TorchDenseNet121(nn.Module):
    
    def __init__(
        self,
        n_classes,
        weights = None,
    ):
        """
        Args:
            n_classes (int): Number of classes in the dataset.
            weights (str): One the following:
                - 'imagenet' (pre-training on ImageNet).
                - Path to model weights.
                - None (random initialization).

        """
        super(TorchDenseNet121, self).__init__()

        use_imagenet = weights == 'imagenet'
        use_custom_weights = weights is not None and not use_imagenet

        self.model = densenet121(
            pretrained=use_imagenet,
            progress=False,
        )

        self.model.classifier = nn.Sequential(
            nn.Linear(1024, n_classes),
            nn.Sigmoid()
        )

        if use_custom_weights:
            self._load_state_dict(torch.load(weights))

    def _load_state_dict(self, state_dict):
 
        current_state = self.state_dict()

        for name, param in state_dict.items():
            striped_name = name.replace('model.', '')
            if striped_name not in current_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            current_state[name].copy_(param)

    def forward(self, x):
        # Model expects 3-channel input.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.model(x).squeeze()
        x = F.one_hot(torch.tensor(x.round(), dtype=torch.int64), num_classes=2)
        return x
