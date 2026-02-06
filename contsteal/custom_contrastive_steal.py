import sys
import os

# Add the parent directory of 'new_attacks' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import argparse
from surrogate_model import Surrogate_model
from src.contsteal.contsteal_utils import load_dataset
from Loss import ContrastiveLoss
from custom_train_representation import train_representation,train_represnetation_linear
from train_posteriors import train_posterior
from test_target import test_for_target
from test_last import test_onehot
from train_onehot import train_onehot
from train_posteriors import train_posterior
from test_target import test_for_target
import numpy as np
# from utils import load_target_model,load_dataset
import dataloader
from test_target import test_for_target
import torchvision
from Linear import linear
import os
from PIL import Image
import requests
import timm

# from utils import load_victim
from src.models.resnet import ResNetEncoder
from src.models.dino import DinoEncoder
from src.models.clip import CLIPVisionEncoder

def main():
    torch.set_num_threads(1)   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type',default='simclr',type=str)
    parser.add_argument('--pretrain',default='cifar10',type=str)
    parser.add_argument('--target-dataset',default='cifar10',type=str)
    parser.add_argument('--surrogate-dataset',default='cifar10',type=str)
    parser.add_argument('--augmentation',default=2,type=int)
    parser.add_argument('--surrogate-arch',default='resnet18',type=str, choices=['resnet18', 'resnet34', 'vitb16'])
    parser.add_argument('--epoch',default= 100, type = int)
    
    parser.add_argument('--victim-arch', default='clip', type=str, choices=['clip', 'clip-vitb32', 'clip-vitb16', 'dino-vitb16'], help='Victim model architecture')
    parser.add_argument('--victim-head', default="False", type=str, help='To use victim emebedding head or not.')
    parser.add_argument('--surrogate-head', default="False", type=str, help='To use surrogate emebedding head or not.')
    parser.add_argument('--out-dim', default=128, type=int, help='Embedding dimension size of victim model')
    parser.add_argument('--victim-path', type=str, help='Path to victim model.')
    parser.add_argument('--save-path', type=str, help='Path to stolen model.')
    parser.add_argument('--gpu-index', type=int, default=0, help='GPU index to use.')
    parser.add_argument('--gradcam', type=str, default="False", help='Use GradCam for victim model.')


    args = parser.parse_args()
    print("Arguments:", args)
    device = torch.device(f"cuda:{args.gpu_index}") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Device:{device} will be used.')
    catagory_num = 10
    # surrogate_model = Surrogate_model(args.out_dim, catagory_num,args.surrogate_arch).to(device)
    # surrogate_model = ResNetEncoder(base_model=args.surrogate_arch,
    #                                 out_dim=args.out_dim, include_mlp = args.victim_head).to(device)
    # victim_model = ResNetEncoder(base_model=args.victim_arch,
    #                                 out_dim=args.out_dim, include_mlp = args.victim_head).to(device)

    victim_head = (args.victim_head == "True")
    if "clip" in args.victim_arch:
        if args.gradcam == "True":
            raise NotImplementedError("GradCam not implemented for CLIP models")
        else:
            victim_model = CLIPVisionEncoder(base_model=args.victim_arch, out_dim=args.out_dim, loss=None, include_mlp=victim_head, pretrained=True)
            victim_model.to(device).eval()
    
    elif "vit" in args.victim_arch:
        if args.gradcam == "True":
            raise NotImplementedError("GradCam not implemented for ViT models")
        else:
            victim_model = DinoEncoder(base_model=args.victim_arch, out_dim=args.out_dim, loss=None, include_mlp = victim_head, pretrained=True)
            victim_model.to(device).eval()
            

    surrogate_head = (args.surrogate_head == "True")
    if 'resnet' in args.surrogate_arch:
        surrogate_model = ResNetEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim, loss=None, include_mlp=surrogate_head)
    elif 'vit' in args.surrogate_arch:
        surrogate_model = DinoEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim, loss=None, include_mlp=surrogate_head, pretrained=False)
    surrogate_model.to(device)

    batch_size = 128


    # victim_model,target_linear = load_target_model(args.model_type,args.pretrain,args.target_dataset)
    train_dataset,test_dataset,linear_dataset = load_dataset(args.pretrain,args.target_dataset,args.surrogate_dataset,args.augmentation,1)#args.split)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    linear_loader = torch.utils.data.DataLoader(
        linear_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    
    criterion = ContrastiveLoss(batch_size, device)
    # optimizer = torch.optim.Adam(surrogate_model.encoder.parameters(), lr=3e-4)
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=3e-4)
    # criterion2 = torch.nn.CrossEntropyLoss()
    # optimizer2 = torch.optim.Adam(surrogate_model.linear.parameters(), lr=3e-4)
    for i in range(args.epoch):
        print(f"Epoch: {i+1}/{args.epoch}")
        train_representation(victim_model,surrogate_model,train_loader,criterion,optimizer,device, args)
    # for i in range(args.epoch):
    #     train_represnetation_linear(surrogate_model,victim_model,linear_loader,criterion2,optimizer2,device)
    #     agreement,accuracy = test_onehot(victim_model,surrogate_model,test_loader)
    os.makedirs(args.save_path, exist_ok=True)
    torch.save({
                'epoch': args.epoch,
                'arch': args.surrogate_arch,
                'state_dict': surrogate_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            args.save_path+'/srgt_of_'+args.victim_arch+'_epoch_'+str(args.epoch)+'.pth.tar'
            )
    print(f"Model saved to {args.save_path}/srgt_of_{args.victim_arch}_epoch_{args.epoch}.pth.tar")
if __name__ == "__main__":
    main()
    
