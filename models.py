import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights


class MTLModel(nn.Module):
    def __init__(self, num_tasks, num_classes, shared_layers):
        """
        Base class for MTL models

        Args:
            num_tasks (int): Number of tasks.
            num_classes (int): Number of classes per task.
            shared_layers (nn.Module): Shared layers for feature extraction.
        """
        super(MTLModel, self).__init__()
        self.shared_layers = shared_layers
        self.task_specific_layers = nn.ModuleList([nn.Linear(self._shared_output_dim(), num_classes, bias=False) for _ in range(num_tasks)])

    def _shared_output_dim(self):
        raise NotImplementedError("This function should be implemented by subclasses")

    def forward(self, x, task_id):
        # task_id is the index of the task in the dataset
        shared_features = self.shared_layers(x)
        return self.task_specific_layers[task_id](shared_features)

    def forward_emb(self, x):
        return self.shared_layers(x)


class MTLVision(MTLModel):
    def __init__(self, num_tasks=128, num_classes=2, pretrained=False, arch="resnet34", bottleneck_dim=None):
        self.bottleneck = bool(bottleneck_dim)
        
        # pick MTL backbone
        if arch == "resnet18":
            model_init = resnet18
            weight_init = ResNet18_Weights
        elif arch == "resnet34":
            model_init = resnet34
            weight_init = ResNet34_Weights
        elif arch == "resnet50":
            model_init = resnet50
            weight_init = ResNet50_Weights
        elif arch == "resnet101":
            model_init = resnet101
            weight_init = ResNet101_Weights
        elif arch == "wrn50":
            model_init = wide_resnet50_2
            weight_init = Wide_ResNet50_2_Weights

        if pretrained:
            shared_layers = model_init(weights=weight_init.IMAGENET1K_V1)
        else:
            shared_layers = model_init(num_classes=num_classes)

        if bottleneck_dim:
            bigger_dimension = shared_layers.fc.in_features

            # create learned bottleneck
            bottleneck_layer = nn.Linear(bigger_dimension, bottleneck_dim, bias=False)
            shared_layers.fc = nn.Identity()
            shared_layers = nn.Sequential(
                    shared_layers,
                    bottleneck_layer,
                    nn.Dropout(0.3),
            )

            self.hidden_dim = bottleneck_dim
        else:
            self.hidden_dim = shared_layers.fc.in_features
            shared_layers.fc = nn.Identity()

        super(MTLVision, self).__init__(num_tasks, num_classes, shared_layers)

    def _shared_output_dim(self):
        # embedding dimension
        return self.hidden_dim  

    def forward_emb(self, x):
        # query representation
        return self.shared_layers(x)


