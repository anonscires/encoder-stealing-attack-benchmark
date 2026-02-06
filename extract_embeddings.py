import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from src.models.resnet import ResNetEncoder
from src.models.dino import DinoEncoder
from src.models.clip import CLIPVisionEncoder
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
from src.sslsteal.sslsteal_utils import load_victim

parser = argparse.ArgumentParser(description='Extract Embeddings from Encoders')
parser.add_argument('--model-path', required=False, type=str, 
                    help='Path to the model checkpoint')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=['resnet18', 'resnet34', 'resnet50', 'vitb16', 'clip', 'clip-vitb16', 'clip-vitb32'], 
                    help='model architecture')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('--split', default='test', type=str,
                    choices=['train', 'test'],
                    help='dataset split to extract embeddings from')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--out-dim', default=512, type=int, 
                    help='Output size of embedding dim')
parser.add_argument('--modeltype', default='surrogate', type=str,
                    help='Type of model to load', choices=['victim', 'surrogate'])
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function used during training')
parser.add_argument('--head', default='False', type=str,
                    help='Whether model has MLP head', choices=['True', 'False'])
parser.add_argument('--output-dir', default='./embeddings', type=str,
                    help='Directory to save embeddings')
parser.add_argument('--output-name', default=None, type=str,
                    help='Name for output file (without extension). If not provided, auto-generated')
parser.add_argument('--gpu', default=0, type=int, 
                    help="GPU Index to use.")

args = parser.parse_args()

# Setup device
if torch.cuda.is_available():
    if torch.cuda.device_count() > args.gpu:
        device = f'cuda:{args.gpu}'
        torch.cuda.set_device(args.gpu)
    else:
        print(f"GPU index {args.gpu} not available. Using default CUDA device.")
        device = 'cuda'
else:
    device = 'cpu'
print("Using device:", device)


def get_stl10_data_loader(download, split='test', shuffle=False, batch_size=256, 
                          transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()])):
    """Get STL10 data loader"""
    dataset = datasets.STL10(f"./datadir/", split=split, download=download,
                            transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                       num_workers=4, drop_last=False, shuffle=shuffle)
    return loader


def get_cifar10_data_loader(download, split='test', shuffle=False, batch_size=256, 
                           transform=transforms.ToTensor()):
    """Get CIFAR10 data loader"""
    is_train = (split == 'train')
    dataset = datasets.CIFAR10(f"./datadir/", train=is_train, download=download,
                              transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size,
                       num_workers=4, drop_last=False, shuffle=shuffle)
    return loader


def get_svhn_data_loader(download, split='test', shuffle=False, batch_size=256, 
                        transform=transforms.ToTensor()):
    """Get SVHN data loader"""
    dataset = datasets.SVHN(f"./datadir/SVHN",
                           split=split, download=download,
                           transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size,
                       num_workers=4, drop_last=False, shuffle=shuffle)
    return loader


def load_encoder(args, device):
    """Load encoder model based on architecture and model type"""
    include_mlp = args.head == "True"
    
    if 'resnet' in args.arch:
        if args.modeltype == "victim":
            print(f"Loading ResNet Encoder for victim model: {args.arch}")
            encoder = ResNetEncoder(base_model=args.arch, out_dim=args.out_dim,
                                  loss=args.losstype, include_mlp=include_mlp).to(device)
            encoder = load_victim(args.model_path, encoder, device=device, discard_mlp=True)
        else:
            print(f"Loading ResNet Encoder for surrogate model: {args.arch}")
            encoder_state_dict = torch.load(args.model_path, map_location=device)
            encoder = ResNetEncoder(base_model=args.arch, out_dim=args.out_dim, 
                                  loss=args.losstype, include_mlp=include_mlp).to(device)
            encoder.load_state_dict(encoder_state_dict['state_dict'], strict=True)
    
    elif 'clip' in args.arch:
        print(f"Loading CLIP Encoder for victim model: {args.arch}")
        encoder = CLIPVisionEncoder(base_model=args.arch, out_dim=args.out_dim, 
                                      loss=args.losstype, include_mlp=include_mlp, 
                                      pretrained=True).to(device)
    
    elif 'vitb16' in args.arch:
        if args.modeltype == "victim":
            print(f"Loading DINO Encoder for victim model: {args.arch}")
            encoder = DinoEncoder(base_model=args.arch, out_dim=args.out_dim, 
                                loss=args.losstype, include_mlp=include_mlp, 
                                pretrained=False).to(device)
            encoder = load_victim(args.model_path, encoder, device=device, discard_mlp=True)
        else:
            print(f"Loading DINO Encoder for surrogate model: {args.arch}")
            encoder_state_dict = torch.load(args.model_path, map_location=device)
            encoder = DinoEncoder(base_model=args.arch, out_dim=args.out_dim,
                                loss=args.losstype, include_mlp=include_mlp, 
                                pretrained=False).to(device)
            encoder.load_state_dict(encoder_state_dict['state_dict'], strict=True)
    else:
        raise NotImplementedError(f"Model {args.arch} not implemented for embedding extraction.")
    
    return encoder


def get_transform(arch, dataset):
    """Get appropriate transform based on architecture and dataset"""
    if 'vitb16' in arch:
        # Use DINO transform
        transform = DinoEncoder.get_transform()
    elif 'clip' in arch:
        # Use CLIP transform
        transform = CLIPVisionEncoder.get_transform()
    else:
        # Use ResNet transform
        if dataset == 'stl10':
            transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        else:
            transform = transforms.ToTensor()
    
    return transform


def extract_embeddings(encoder, data_loader, device):
    """Extract embeddings from the encoder for all data"""
    encoder.eval()
    
    all_embeddings = []
    all_labels = []
    
    print(f"Extracting embeddings from {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Extracting embeddings")):
            images = images.to(device)
            
            # Get embeddings
            embeddings = encoder(images)
            
            # Move to CPU and convert to numpy
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Extracted embeddings shape: {all_embeddings.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return all_embeddings, all_labels


def save_embeddings(embeddings, labels, output_path):
    """Save embeddings and labels to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(output_path, 
             embeddings=embeddings, 
             labels=labels)
    
    print(f"Embeddings saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("Extracting Embeddings")
    print("=" * 80)
    print(f"Model: {args.arch}")
    print(f"Model Type: {args.modeltype}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Model Path: {args.model_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output Dimension: {args.out_dim}")
    print("=" * 80)
    
    # Get appropriate transform
    transform = get_transform(args.arch, args.dataset)
    
    # Get data loader
    if args.dataset == 'cifar10':
        data_loader = get_cifar10_data_loader(download=True, split=args.split, 
                                             batch_size=args.batch_size, transform=transform)
    elif args.dataset == 'stl10':
        data_loader = get_stl10_data_loader(download=True, split=args.split,
                                           batch_size=args.batch_size, transform=transform)
    elif args.dataset == 'svhn':
        data_loader = get_svhn_data_loader(download=True, split=args.split,
                                          batch_size=args.batch_size, transform=transform)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Load encoder
    encoder = load_encoder(args, device)
    encoder.to(device)
    encoder.eval()
    
    print("Model loaded successfully!")
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(encoder, data_loader, device)
    
    # Generate output filename if not provided
    if args.output_name is None:
        output_name = f"{args.modeltype}_{args.arch}_{args.dataset}_{args.split}_embeddings.npz"
    else:
        output_name = args.output_name if args.output_name.endswith('.npz') else f"{args.output_name}.npz"
    
    output_path = os.path.join(args.output_dir, output_name)
    
    # Save embeddings
    save_embeddings(embeddings, labels, output_path)
    
    print("=" * 80)
    print("Embedding extraction complete!")
    print("=" * 80)
    print(f"Summary:")
    print(f"  - Total samples: {len(labels)}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print(f"  - Number of classes: {len(np.unique(labels))}")
    print(f"  - Output file: {output_path}")
    print("=" * 80)
