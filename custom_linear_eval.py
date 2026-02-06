import sys
import os

# Add the parent directory of 'new_attacks' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import argparse
from torch.utils.data import DataLoader
from models.resnet import ResNetEncoder, ResNet18, ResNet34, ResNet50
import torchvision.transforms as transforms
import logging
from torchvision import datasets
from src.models.clip import CLIPVisionEncoder
from src.models.dino import DinoEncoder, DinoGuardCamEncoder
from tqdm import tqdm
from src.dataset.sicap import SicapSingleDataset

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--datasettest', default='cifar10',
                    help='dataset to run downstream task on', choices=['stl10', 'cifar10', 'svhn', 'sicap'])
parser.add_argument('-a', '--arch', metavar='ARCH',
        choices=['clip', 'dino-vitb16'], help='model architecture')
parser.add_argument('-n', '--num-labeled', default=50000,type=int,
                     help='Number of labeled examples to train on')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs surrogate model was trained with')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--lr', default=1e-4, type=float, 
                    help='learning rate to train the model with.')
parser.add_argument('--modeltype', default='victim', type=str,
                    help='Type of model to evaluate', choices=['victim', 'surrogate'])
parser.add_argument('--save', default='False', type=str,
                    help='Save final model', choices=['True', 'False'])
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use.')
parser.add_argument('--head', default='False', type=str,
                    help='surrogate model was trained using recreated head.', choices=['True', 'False'])
parser.add_argument('--defence', default='False', type=str,
                    help='Use defence on the victim side by perturbing outputs', choices=['True', 'False'])
parser.add_argument('--sigma', default=0.5, type=float,
                    help='standard deviation used for perturbations')
parser.add_argument('--mu', default=5, type=float,
                    help='mean noise used for perturbations')
parser.add_argument('--clear', default='True', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--watermark', default='False', type=str,
                    help='Watermark used when training the model', choices=['True', 'False'])
parser.add_argument('--entropy', default='False', type=str,
                    help='Additional softmax layer when training the model', choices=['True', 'False'])
parser.add_argument('--freeze-encoder', default="True", type=str,
                    help='Whether to freeze the encoder weights or not', choices=["True", "False"])
parser.add_argument('--out-dim', default=512, type=int, help='Output size of embedding dim')
parser.add_argument('--save-path', default=None, type=str, help='Path to the save the log and model at')
parser.add_argument('--gradcam', default="False", type=str, help='Using gradcam blur or not.', choices=['True', 'False'])
parser.add_argument('--gpu', default=0, type=int, help="GPU Index to use.")
args = parser.parse_args()


def get_stl10_data_loaders(download, dataset = None, shuffle=False, batch_size=args.batch_size, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()])):
    train_dataset = datasets.STL10(f"./datadir/", split='train', download=download,
                                  transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10(f"./datadir/", split='test', download=download,
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, dataset = None, shuffle=False, batch_size=args.batch_size, transform=transforms.ToTensor()):
    train_dataset = datasets.CIFAR10(f"./datadir/", train=True, download=download,
                                    transform=transform)
    test_dataset = datasets.CIFAR10(f"./datadir/",
                                    train=False, download=download,
                                    transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_svhn_data_loaders(download, dataset = None, shuffle=False, batch_size=args.batch_size, transform=transforms.ToTensor()):
    train_dataset = datasets.SVHN(f"./datadir/SVHN",
                                    split='train', download=download,
                                    transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.SVHN(f"./datadir/SVHN",
                                    split='test', download=download,
                                    transform=transform)
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_sicap_data_loaders(download=None, dataset = None, shuffle=False, batch_size=args.batch_size, transform=transforms.ToTensor()):
    train_dataset = SicapSingleDataset(
                                root='/work/hdd/bcvd/sraisharma/datasets/sicapv2/SICAPv2',
                                train=True,
                                transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = SicapSingleDataset(
                                root='/work/hdd/bcvd/sraisharma/datasets/sicapv2/SICAPv2',
                                train=False,
                                transform=transform)
    indxs = list(range(len(test_dataset) - 1000, len(test_dataset)))
    test_dataset = torch.utils.data.Subset(test_dataset,
                                           indxs)  # only select last 1000 samples to prevent overlap with queried samples.
    test_loader = DataLoader(test_dataset, batch_size=64,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    print("Using device:", device)

    log_dir = args.save_path

    
    logname = f'testing{args.modeltype}{args.datasettest}{args.num_queries}.log'

    if args.clear == "True":
        if os.path.exists(os.path.join(log_dir, logname)):
            os.remove(os.path.join(log_dir, logname))
    logging.basicConfig(
        filename=os.path.join(log_dir, logname),
        level=logging.DEBUG)
    print(f"logging to {os.path.join(log_dir, logname)}")

    if args.losstype == "symmetrized":
        raise NotImplementedError('Symmetrized loss not implemented.')
        if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False,
                                                num_classes=10).to(device)
        elif args.arch == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False,
                                                num_classes=10).to(device)
        elif args.arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False,
                                                num_classes=10).to(device)


    include_mlp = args.head == "True"
    if 'clip' in args.arch:
        # load pretrained clip model as victim
        if args.modeltype == "victim":
            encoder = CLIPVisionEncoder(base_model=args.arch, out_dim=args.out_dim, loss=args.losstype, include_mlp = include_mlp, pretrained=True).to(device)
            transform = transforms.Compose([
                                transforms.Resize((224, 224)),  # CLIP expects 224x224 images
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                            ])
        else:
            raise NotImplementedError("Custom Surrogate model not implemented for CLIP.")
    elif 'dino' in args.arch:
        if args.modeltype == "victim":
            if args.gradcam == "True":
                encoder = DinoGuardCamEncoder(base_model=args.arch, out_dim=args.out_dim, loss=args.losstype, include_mlp = include_mlp, pretrained=True).to(device)
                transform = DinoGuardCamEncoder.get_transform()
            else:
                encoder = DinoEncoder(base_model=args.arch, out_dim=args.out_dim, loss=args.losstype, include_mlp = include_mlp, pretrained=True).to(device)
                transform = DinoEncoder.get_transform()
        else:
            raise NotImplementedError("Custom Surrogate model not implemented for DINO.")
    else:
        raise NotImplementedError(f"Model {args.arch} not implemented for linear evaluation.")
        
    model = torch.nn.Sequential(encoder, torch.nn.Linear(args.out_dim, 10))
    model.to(device)

    if args.datasettest == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(download=True, dataset=args.datasettest, batch_size=args.batch_size, transform=transform)
    elif args.datasettest == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=True, dataset = args.datasettest, batch_size=args.batch_size, transform=transform)
    elif args.datasettest == "svhn":
        train_loader, test_loader = get_svhn_data_loaders(download=True, dataset = args.datasettest, batch_size=args.batch_size, transform=transform)
    elif args.datasettest == "sicap":
        train_loader, test_loader = get_sicap_data_loaders(download=True, dataset = args.datasettest, batch_size=args.batch_size, transform=transform)
    
    if args.gradcam == "False":
        if(args.freeze_encoder == "True"):
            print(f"Freeze Encoder Flag = {args.freeze_encoder}. Encoder Weights will be freezed")
            # freeze all layers but the last fc 
            for name, param in model[0].named_parameters():
                param.requires_grad = False
        else:
            print(f"Freeze Encoder Flag = {args.freeze_encoder}. Encoder Weights will not be freezed")
    else:
        print(f"Gradcam Flag = {args.gradcam}. Encoder Weights will not be freezed")

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    if args.gradcam == "False":
        if(args.freeze_encoder == "True" and args.head == "False"):
            parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            assert len(parameters) == 2  # fc.weight, fc.bias

    if args.modeltype == "victim":
        args.lr = 3e-4 
        optimizer = torch.optim.Adam(params_to_update, lr=args.lr, weight_decay=0.0008) 
        criterion = torch.nn.CrossEntropyLoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        optimizer = torch.optim.Adam(params_to_update, lr=args.lr,
                                    weight_decay=0.0008)
        criterion = torch.nn.CrossEntropyLoss().to(device)
    epochs = args.epochs

    ## Trains the representation model with a linear classifier to measure the accuracy on the test set labels of the victim/surrogate model

    logging.info(f"Evaluating {args.modeltype} model on {args.datasettest} dataset. Model trained using {args.losstype}.")
    logging.info(f"Args: {args}")
    for epoch in range(epochs):
        top1_train_accuracy = 0
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
        for counter, (x_batch, y_batch) in train_iter:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (counter+1) * x_batch.shape[0] >= args.num_labeled:
                break

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        test_loss = 0
        test_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch} [Test]")
        for counter, (x_batch, y_batch) in test_iter:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
            test_loss += criterion(logits, y_batch)

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        logging.debug(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

    if args.save == "True":
        if args.modeltype == "surrogate":
            torch.save(model.state_dict(), os.path.join(log_dir, f"surrogate_linear_{args.datasettest}.pth.tar"))
        else:
            torch.save(model.state_dict(), os.path.join(log_dir, f"victim_linear_{args.datasettest}_head{args.head}.pth.tar"))
