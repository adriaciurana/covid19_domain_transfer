from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet34


import torch.nn as nn
def ModelCreator(inference=True, arch='resnet34'):
    if arch == 'resnet34':
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = IntermediateLayerGetter(
                    resnet34(pretrained=True), #, replace_stride_with_dilation=[False, True, True]
                        return_layers={'layer4': 'out'}
                )

            def forward(self, x):
                return self.features(x)['out']
        model = Model()
        discriminator = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        return model, classifier, discriminator

