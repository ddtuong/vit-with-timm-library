from configuration import *

class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model(VIT_PRETRAINED_MODEL, pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

        # for block in self.model.blocks[:10]:  # freezing layers
        #     for param in block.parameters():
        #         param.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        return x
