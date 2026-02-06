import sys
import os

# Add the parent directory of 'new_attacks' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from src.dataset.contrastive_learning_dataset import ContrastiveLearningDataset, CustomRegularDataset
from src.dataset.view_generator import ContrastiveLearningViewGenerator
from src.models.simsiam import SimSiamEncoder
from src.models.resnet import ResNetEncoder
from src.models.clip import CLIPVisionEncoder
from src.models.dino import DinoEncoder, DinoGuardCamEncoder, DinoGradCamEncoder, DinoGradCamEncoderWithLogits
from custom_simclr import CustomSimCLR
from src.sslsteal.sslsteal_utils import load_victim
import os
from torch.nn import DataParallel

def comma_separated_ints(arg_string):
    return [int(item) for item in arg_string.split(',')]

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--datadir', metavar='DIR', default=f"./datadir",
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name')
parser.add_argument('--datasetsteal', default='cifar10',
                    help='dataset used for querying the victim', choices=['stl10', 'cifar10', 'svhn', 'sicap'])
parser.add_argument('-a', '--victim-arch', metavar='ARCH', default='clip',
                    choices=["clip-vitb32", "clip-vitb16", "clip", "dino-vitb16", "dino-vits16"],
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--surrogate-arch', default='resnet34',
                    choices=model_names + ["vitb16", "vits16"],
                    help='stolen model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet34)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                    help='number of epochs victim was trained with')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=200, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--temperaturesn', default=100, type=float,
                    help='temperature for soft nearest neighbors loss')
parser.add_argument('--num_queries', default=9000, type=int, metavar='N',
                    help='Number of queries to steal the model.')
parser.add_argument('--n-views', default=1, type=int, metavar='N',  # 2 to use multiple augmentations.
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default="0", type=comma_separated_ints, help='Gpu index.')
parser.add_argument('--logdir', default='test', type=str,
                    help='Log directory to save output to.')
parser.add_argument('--losstype', default='infonce', type=str,
                    help='Loss function to use')
parser.add_argument('--lossvictim', default='infonce', type=str,
                    help='Loss function victim was trained with')
parser.add_argument('--victim-head', default='False', type=str,
                    help='Access to victim head while (g) while getting representations', choices=['True', 'False'])
parser.add_argument('--surrogate-head', default='False', type=str,
                    help='Use an additional head while training the stolen model.', choices=['True', 'False'])
parser.add_argument('--defence', default='False', type=str,
                    help='Use defence on the victim side by perturbing outputs', choices=['True', 'False'])
parser.add_argument('--sigma', default=0.5, type=float,
                    help='standard deviation used for perturbations')
parser.add_argument('--mu', default=5, type=float,
                    help='mean noise used for perturbations')
parser.add_argument('--clear', default='True', type=str,
                    help='Clear previous logs', choices=['True', 'False'])
parser.add_argument('--watermark', default='False', type=str,
                    help='Evaluate with watermark model from victim', choices=['True', 'False'])
parser.add_argument('--entropy', default='False', type=str,
                    help='Use entropy victim model', choices=['True', 'False'])
parser.add_argument('--force', default='False', type=str,
                    help='Use cifar10 training set when stealing from cifar10 victim model.', choices=['True', 'False'])
parser.add_argument('--ckptdir', type=str, help='Directory to save checkpoints')
parser.add_argument('--gradcam', default="False", choices=['True', 'False', 'gradcam', 'guardcam', 'gradcamwithlogits'], help='Use gradcam defense.')
parser.add_argument('--ckptpath', type=str, default=None, help='Path to load the checkpoint from.')


def main():

    args = parser.parse_args()

    print('Checkpoints will be saved to:', args.ckptdir)
    
    if torch.cuda.is_available():
        print(f"Using GPU {args.gpu_index} for training")
        print(f"Base GPU index: {args.gpu_index[0]}")
        args.device = torch.device(f'cuda:{args.gpu_index[0]}')
        cudnn.deterministic = True
        cudnn.benchmark = True
        torch.cuda.set_device(args.gpu_index[0])
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.dataset not in ["cifar10", "stl10", "svhn"]:
        print(f"Dataset '{args.dataset}' is provided as args.dataset for CustomStealing")

    if args.losstype in  ["infonce", "softnn", "supcon", "barlow"]:
        # args.batch_size = 256
        args.weight_decay = 1e-4
        print(f"Using batch size {args.batch_size} and weight decay 1e-4 for contrastive losses")
    if args.losstype == "infonce":
        args.lr = 0.0003
    if args.losstype == "supcon":
        args.lr = 0.05
    if args.losstype == "softnn":
        args.lr = 0.001
    if args.losstype == "symmetrized":
        args.batch_size = 256
        args.lr = 0.05
        args.out_dim = 512
        args.n_views = 2
        args.surrogate_head = "True"
    if args.losstype in ["mse", "softce", "wassersein"]:
        args.n_views = 1
    # if args.surrogate_head == "True" and args.victim_head == "False":
    #     args.out_dim = 512 
    if args.n_views == 1:
        if args.victim_arch in ["clip-vitb32", "clip-vitb16"]:
            victim_transform = CLIPVisionEncoder.get_transform()
        elif args.victim_arch in ["dino-vitb16", "dino-vits16"]:
            if args.gradcam == "guardcam":
                victim_transform = DinoGuardCamEncoder.get_transform()
            elif args.gradcam == "True" or args.gradcam == "gradcam":
                victim_transform = DinoGradCamEncoder.get_transform()
            elif args.gradcam == "gradcamwithlogits":
                victim_transform = DinoGradCamEncoderWithLogits.get_transform()
            else:
                victim_transform = DinoEncoder.get_transform()
        else:
            raise NotImplementedError(f"Victim transform for arch {args.victim_arch} not implemented yet for CustomStealing")

        if args.surrogate_arch in ["clip-vitb32", "clip-vitb16"]:
            raise NotImplementedError("Surrogate CLIP Vision Encoder not implemented yet for CustomStealing")
            # surrogate_transform = CLIPVisionEncoder.get_transform()
        elif args.surrogate_arch in ["dino-vitb16", "dino-vits16"]:
            raise NotImplementedError("Surrogate DINO Encoder not implemented yet for CustomStealing")
        elif args.surrogate_arch in ["vitb16", "vits16"]:
            surrogate_transform = DinoEncoder.get_transform()
        else:
            surrogate_transform = ContrastiveLearningViewGenerator(CustomRegularDataset.get_simclr_pipeline_transform(32), n_views=1)

        print("Using following transforms for victim and surrogate model:")
        print("Victim transform:", victim_transform)
        print("Surrogate transform:", CustomRegularDataset.get_simclr_pipeline_transform(32))
        
        dataset = CustomRegularDataset(args.datadir, victim_transform=victim_transform, surrogate_transform=surrogate_transform) # using data augmentation for queries
    elif args.n_views == 2:
        raise NotImplementedError("2 views not implemented yet for CustomStealing")
        dataset = ContrastiveLearningDataset(args.datadir) # using data augmentation for queries

    if args.n_views > 1:
        raise NotImplementedError("More than one view not implemented yet for CustomStealing")
    
    print("Args for Custom Stealing", args)


    if args.datasetsteal != args.dataset or args.force == "True":
        query_dataset = dataset.get_dataset(args.datasetsteal, args.n_views)
        indxs = list(range(0, len(query_dataset)))
        query_dataset = torch.utils.data.Subset(query_dataset,
                                               indxs)
    else:
        query_dataset = dataset.get_test_dataset(args.datasetsteal,
                                                 args.n_views)
        indxs = list(range(0, len(query_dataset) - 1000))

        query_dataset = torch.utils.data.Subset(query_dataset,
                                                indxs)  # query set (without last 1000 samples as they are used in the test set)

    print(f"Batch size: {args.batch_size}, Number of queries: {len(query_dataset)}")
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    victim_head = (args.victim_head == "True")

    print("Initializing Victim Model")
    if args.victim_arch in ["clip-vitb32", "clip-vitb16"]:
        if args.gradcam == "True":
            raise NotImplementedError("Gradcam not implemented yet for CustomStealing with CLIP Vision Encoder")
        else:
            victim_model = CLIPVisionEncoder(base_model=args.victim_arch, out_dim=args.out_dim, loss=args.lossvictim, include_mlp = victim_head, pretrained=True).to(args.device)    
    elif args.victim_arch in ["dino-vitb16", "dino-vits16"]:
        if args.gradcam == "guardcam":
            victim_model = DinoGuardCamEncoder(base_model=args.victim_arch, out_dim=args.out_dim, loss=args.lossvictim, include_mlp = victim_head, pretrained=True).to(args.device)
        elif args.gradcam == "True" or args.gradcam == "gradcam":
            victim_model = DinoGradCamEncoder(base_model=args.victim_arch, out_dim=args.out_dim, loss=args.lossvictim, include_mlp = victim_head, pretrained=True, batch_size=args.batch_size).to(args.device)
            victim_model.cam.device = args.device
        elif args.gradcam == "gradcamwithlogits":
            if args.ckptpath is None:
                raise ValueError("Please provide a checkpoint path for gradcam with logits.")
            print(f"Loading checkpoint from {args.ckptpath}")
            victim_model = DinoGradCamEncoderWithLogits(base_model=args.victim_arch, out_dim=args.out_dim, loss=args.lossvictim, include_mlp = victim_head, pretrained=True, batch_size=args.batch_size, ckpt_path=args.ckptpath).to(args.device)
            victim_model.cam.device = args.device
        else:
            victim_model = DinoEncoder(base_model=args.victim_arch, out_dim=args.out_dim, loss=args.lossvictim, include_mlp = victim_head, pretrained=True).to(args.device)
    else:
        raise NotImplementedError(f"Victim model {args.victim_arch} not implemented yet for CustomStealing")


    print("Initializing Surrogate Model")
    surrogate_head = (args.surrogate_head == "True")
    if args.surrogate_arch in ["clip-vitb32", "clip-vitb16"]:
        raise NotImplementedError("Surrogate CLIP Vision Encoder not implemented yet for CustomStealing")
    #     model = CLIPVisionEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim,loss=args.lossvictim, include_mlp = surrogate_head, pretrained=False).to(args.device)
    elif args.surrogate_arch in ["dino-vitb16", "dino-vits16"]:
        raise NotImplementedError("Surrogate DINO Encoder not implemented yet for CustomStealing")
    elif args.surrogate_arch in ["vitb16", "vits16"]:
        model = DinoEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim,loss=args.lossvictim, include_mlp = surrogate_head, pretrained=False).to(args.device)
    else:
        model = ResNetEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim, loss=args.losstype, include_mlp = surrogate_head)
    
    if args.losstype == "symmetrized":
        model = SimSiamEncoder(models.__dict__[args.victim_arch], args.out_dim, args.out_dim)


    victim_model = victim_model.to(args.device)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,  
                                 weight_decay=args.weight_decay)

    if args.losstype in ["supcon", "symmetrized"]:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
        query_loader), eta_min=0,last_epoch=-1)
    with torch.cuda.device(args.gpu_index[0]):
        if args.defence == "True":
            raise NotImplementedError("Defence not implemented yet for CustomStealing")
            simclr = CustomSimCLR(stealing=True, victim_model=victim_model, victim_head=victim_head,entropy_model=entropy_model,
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            args=args, logdir=args.ckptdir, loss=args.losstype)
            simclr.steal(query_loader, args.num_queries)
        # elif args.watermark == "True":
        #     simclr = SimCLR(stealing=True, victim_model=victim_model,
        #                     watermark_mlp=watermark_mlp,
        #                     model=model, optimizer=optimizer,
        #                     scheduler=scheduler,
        #                     args=args, logdir=args.logdir, loss=args.losstype)
        #     simclr.steal(query_loader, args.num_queries, watermark_loader)
        else:
            simclr = CustomSimCLR(stealing=True, victim_model=victim_model,
                            model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            args=args, logdir=args.ckptdir, loss=args.losstype)
            simclr.steal(query_loader, args.num_queries)


if __name__ == "__main__":
    main()
    