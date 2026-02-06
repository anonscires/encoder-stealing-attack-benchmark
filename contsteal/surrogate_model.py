import torch
import torch.nn as nn
import torchvision

class Surrogate_model(nn.Module):
    def __init__(self,out_dim,num_classes,model_type):
        super(Surrogate_model,self).__init__()
        print(model_type)
        if(model_type == 'resnet18'):
            self.encoder = torchvision.models.resnet18(pretrained=False,num_classes=out_dim)
        if(model_type == 'resnet34'):
            self.encoder = torchvision.models.resnet34(pretrained=False,num_classes=out_dim)
        if(model_type == 'resnet50'):
            self.encoder = torchvision.models.resnet50(pretrained=False,num_classes=out_dim)
     
        # self.encoder = torchvision.models.resnet18(pretrained=False,num_classes=512)
        self.num_classes = num_classes
        self.linear = nn.Linear(self.encoder.fc.out_features,self.num_classes)

    def forward(self,x):
        x = self.encoder(x)
        x = self.linear(x)
        return x
