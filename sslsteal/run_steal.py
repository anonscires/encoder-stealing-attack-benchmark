import sys
import os

# Add the parent directory of 'new_attacks' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from src.dataset.contrastive_learning_dataset import ContrastiveLearningDataset, RegularDataset
from src.models.simsiam import SimSiamEncoder
from src.models.resnet import ResNetEncoder, ResNetGradCamEncoder
from src.models.dino import DinoEncoder
from simclr import SimCLR
from src.sslsteal.sslsteal_utils import load_victim
import os
import numpy as np
import time
import psutil
import threading
from collections import defaultdict
import subprocess
import platform
import torch.nn.functional as F
import torch.nn as nn
import json

# Try importing optional dependencies
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from colorama import init, Fore
    init(autoreset=True)
except ImportError:
    class DummyFore:
        CYAN = ''
        YELLOW = ''
    Fore = DummyFore()

# Global variables for performance monitoring
performance_metrics = defaultdict(list)
cpu_usage_data = []
memory_usage_data = []
gpu_memory_data = []
monitoring_active = False

def monitor_system_resources():
    """Background thread to monitor CPU and memory usage"""
    global monitoring_active, cpu_usage_data, memory_usage_data, gpu_memory_data
    
    while monitoring_active:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_usage_data.append(cpu_percent)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usage_data.append(memory_info.percent)
        
        # GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            gpu_memory_data.append({
                'allocated': gpu_memory_allocated,
                'reserved': gpu_memory_reserved
            })
        
        time.sleep(1)  # Sample every second

def start_monitoring():
    """Start system resource monitoring"""
    global monitoring_active
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()
    return monitor_thread

def stop_monitoring():
    """Stop system resource monitoring"""
    global monitoring_active
    monitoring_active = False

def get_model_flops(model, input_shape):
    """Estimate FLOPs for a model (simplified estimation)"""
    def flop_count(module, input, output):
        if isinstance(module, nn.Linear):
            return module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            return (module.in_channels * module.out_channels * 
                   module.kernel_size[0] * module.kernel_size[1] * 
                   output.shape[-2] * output.shape[-1])
        elif isinstance(module, nn.BatchNorm2d):
            return output.numel()
        else:
            return 0
    
    model.eval()
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        total_flops += flop_count(module, input, output)
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).cuda()
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops

def get_model_stats(model):
    """Get model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size
    }

def get_system_info():
    """Get CPU and GPU information"""
    system_info = {}
    
    # CPU Information - Enhanced detection
    try:
        # Try multiple methods to get CPU model name
        cpu_model = None
        
        # Method 1: Try /proc/cpuinfo (Linux)
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_model = line.split(':')[1].strip()
                        break
        except:
            pass
        
        # Method 2: Try platform.processor() as fallback
        if not cpu_model or cpu_model == '':
            cpu_model = platform.processor()
        
        # Method 3: Try using lscpu command (Linux)
        if not cpu_model or 'x86_64' in cpu_model or cpu_model == '':
            try:
                result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
                for line in result.stdout.split('\n'):
                    if 'Model name:' in line:
                        cpu_model = line.split(':', 1)[1].strip()
                        break
            except:
                pass
        
        # Final fallback
        if not cpu_model or cpu_model == '' or 'x86_64' in cpu_model:
            cpu_model = "Unknown CPU Model"
        
        system_info['cpu_model'] = cpu_model
        system_info['cpu_cores'] = psutil.cpu_count(logical=False)
        system_info['cpu_threads'] = psutil.cpu_count(logical=True)
        system_info['cpu_frequency'] = psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
        
    except Exception as e:
        system_info['cpu_model'] = f"Error getting CPU info: {str(e)}"
        system_info['cpu_cores'] = psutil.cpu_count(logical=False)
        system_info['cpu_threads'] = psutil.cpu_count(logical=True)
    
    # GPU Information
    try:
        if torch.cuda.is_available():
            system_info['gpu_count'] = torch.cuda.device_count()
            system_info['gpu_names'] = []
            system_info['gpu_memory'] = []
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                system_info['gpu_names'].append(gpu_name)
                system_info['gpu_memory'].append(f"{gpu_memory:.1f}GB")
            
            # Current GPU being used
            current_gpu = torch.cuda.current_device()
            system_info['current_gpu'] = system_info['gpu_names'][current_gpu]
            system_info['current_gpu_memory'] = system_info['gpu_memory'][current_gpu]
        else:
            system_info['gpu_count'] = 0
            system_info['gpu_names'] = ["No CUDA GPU available"]
            system_info['current_gpu'] = "No CUDA GPU available"
    except Exception as e:
        system_info['gpu_info_error'] = f"Error getting GPU info: {str(e)}"
    
    # System Information
    system_info['platform'] = platform.platform()
    system_info['python_version'] = platform.python_version()
    system_info['pytorch_version'] = torch.__version__
    
    # Memory Information
    memory = psutil.virtual_memory()
    system_info['total_ram_gb'] = memory.total / 1024**3
    
    return system_info

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--datadir', metavar='DIR', default=f"./datadir",
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('--datasetsteal', default='cifar10',
                    help='dataset used for querying the victim', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('-a', '--victim-arch', metavar='ARCH', default='resnet34',
                    choices=model_names + ['vitb16'],
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--surrogate-arch', default='resnet34',
                    choices=model_names + ['vitb16'],
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
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
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
parser.add_argument('--victim-path', type=str, help='Path to the victim model')
parser.add_argument('--ckptdir', type=str, help='Directory to save checkpoints')
parser.add_argument('--gradcam', default="False", choices=['True', 'False'], help='Use gradcam defense.')

# Add logging arguments similar to RDA and ContraSteal
parser.add_argument('--log-results-dir', default='logs/sslsteal', type=str, help='Path to save logs')
# parser.add_argument('--seed', default=100, type=int, help='Random seed for reproducibility')

def main():
    args = parser.parse_args()

    # Dump args
    with open(args.log_results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    
    # Set random seeds for reproducibility
    # import random
    # random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
        torch.cuda.set_device(args.gpu_index)
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

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
        dataset = RegularDataset(args.datadir)
    elif args.n_views == 2:
        dataset = ContrastiveLearningDataset(args.datadir) # using data augmentation for queries

    print("Arguments:", args)
    
    # Create log directory
    if not os.path.exists(args.log_results_dir):
        os.makedirs(args.log_results_dir, exist_ok=True)

    # ========================= COMPREHENSIVE PERFORMANCE MONITORING =========================
    print("="*80)
    print("Starting Comprehensive Performance Monitoring for SSL Steal Attack")
    print("="*80)
    
    # Get system information
    system_info = get_system_info()
    
    # Display system information
    print("System Information:")
    print(f"  CPU: {system_info.get('cpu_model', 'Unknown')}")
    print(f"  CPU Cores: {system_info.get('cpu_cores', 'Unknown')} cores, {system_info.get('cpu_threads', 'Unknown')} threads")
    if system_info.get('cpu_frequency') != "Unknown":
        print(f"  CPU Frequency: {system_info['cpu_frequency']:.0f} MHz")
    print(f"  Total RAM: {system_info.get('total_ram_gb', 0):.1f} GB")
    
    if system_info.get('gpu_count', 0) > 0:
        print(f"  GPU Count: {system_info['gpu_count']}")
        for i, (name, memory) in enumerate(zip(system_info['gpu_names'], system_info['gpu_memory'])):
            print(f"  GPU {i}: {name} ({memory})")
        print(f"  Current GPU: {system_info.get('current_gpu', 'Unknown')}")
        print(f"  Current GPU Memory: {system_info.get('current_gpu_memory', 'Unknown')}")
    else:
        print(f"  GPU: {system_info.get('current_gpu', 'No GPU available')}")
    
    print(f"  Platform: {system_info.get('platform', 'Unknown')}")
    print(f"  Python: {system_info.get('python_version', 'Unknown')}")
    print(f"  PyTorch: {system_info.get('pytorch_version', 'Unknown')}")
    print("-"*80)
    
    # Initialize timing and performance metrics
    overall_start_time = time.time()
    timing_metrics = defaultdict(list)
    
    # Store system information in timing metrics
    timing_metrics['system_info'] = system_info
    
    # Start system monitoring
    print("Starting system resource monitoring...")
    monitor_thread = start_monitoring()
    
    # ========================= MODEL SETUP AND INITIALIZATION =========================
    setup_start_time = time.time()

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

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    victim_head = (args.victim_head == "True")
    if "resnet" in args.victim_arch:
        if args.gradcam == "True":
            victim_model = ResNetGradCamEncoder(base_model=args.victim_arch,
                                                out_dim=args.out_dim,loss=args.lossvictim, include_mlp = victim_head).to(args.device)
        else:
            victim_model = ResNetEncoder(base_model=args.victim_arch,
                                        out_dim=args.out_dim,loss=args.lossvictim, include_mlp = victim_head).to(args.device)
        victim_model = load_victim(args.victim_path, victim_model, device=args.device, discard_mlp = True)
    
    elif "vit" in args.victim_arch:
        if args.gradcam == "True":
            raise NotImplementedError("GradCam not implemented for ViT models")
        else:
            victim_model = DinoEncoder(base_model=args.victim_arch, out_dim=args.out_dim,loss=args.lossvictim, include_mlp = victim_head).to(args.device)
            victim_model = load_victim(args.victim_path, victim_model, device=args.device, discard_mlp = True)
            victim_model.eval()
    
    if args.defence == "True": # Use the model head as part of the defence.
        if args.entropy == "True":
            victim_head = ResNetEncoder(base_model=args.victim_arch,
                                            out_dim=args.out_dim,
                                            entropy=args.entropy).to(args.device)
        else:
            victim_head = ResNetEncoder(base_model=args.victim_arch,
                                            out_dim=args.out_dim,
                                            loss=args.lossvictim,
                                            include_mlp=True).to(args.device)

        victim_head = load_victim(args.victim_path, victim_model, device=args.device)
        # model to be used for entropy calculation (assumes specific downstream task being used)
        entropy_model = models.resnet50(pretrained=False,
                                            num_classes=10).to(args.device)
        entropy_model.load_state_dict(torch.load(f"{args.ckptdir}/victim_linear_{args.datasetsteal}.pth.tar"))
    
    surrogate_head = (args.surrogate_head == "True")
    if 'resnet' in args.surrogate_arch:
        model = ResNetEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim, loss=args.losstype, include_mlp=surrogate_head)
    elif 'vit' in args.surrogate_arch:
        model = DinoEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim, loss=args.losstype, include_mlp=surrogate_head)

    if args.losstype == "symmetrized":
        model = SimSiamEncoder(models.__dict__[args.victim_arch], args.out_dim, args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,  
                                 weight_decay=args.weight_decay)

    if args.losstype in ["supcon", "symmetrized"]:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(
        query_loader), eta_min=0,last_epoch=-1)
    
    setup_end_time = time.time()
    model_setup_time = setup_end_time - setup_start_time

    # Get model statistics
    victim_stats = get_model_stats(victim_model)
    surrogate_stats = get_model_stats(model)
    
    print(f"Model setup completed in {model_setup_time:.2f} seconds")
    print(f"Victim Model Stats - Params: {victim_stats['total_params']:,}, Size: {victim_stats['model_size_mb']:.2f}MB")
    print(f"Surrogate Model Stats - Params: {surrogate_stats['total_params']:,}, Size: {surrogate_stats['model_size_mb']:.2f}MB")
    
    timing_metrics['victim_model_stats'] = victim_stats
    timing_metrics['surrogate_model_stats'] = surrogate_stats
    timing_metrics['model_setup_time'] = model_setup_time
    
    # ========================= TRAINING/STEALING PHASE =========================
    print(Fore.CYAN + "Phase: SSL Steal Training")
    print(f"Note: Victim querying pattern depends on loss type: {args.losstype}")
    
    with torch.cuda.device(args.gpu_index):
        if args.defence == "True":
            simclr = SimCLR(stealing=True, victim_model=victim_model, victim_head=victim_head,entropy_model=entropy_model,
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            args=args, logdir=args.ckptdir, loss=args.losstype, timing_metrics=timing_metrics)
            simclr.steal(query_loader, args.num_queries)
        else:
            simclr = SimCLR(stealing=True, victim_model=victim_model,
                            model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            args=args, logdir=args.ckptdir, loss=args.losstype, timing_metrics=timing_metrics)
            simclr.steal(query_loader, args.num_queries)
    
    # Access updated timing_metrics from simclr
    timing_metrics.update(simclr.timing_metrics)

    # ========================= FINAL CLEANUP AND COMPREHENSIVE LOGGING =========================
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    
    # Stop system monitoring
    stop_monitoring()
    
    # Calculate system resource statistics
    avg_cpu_usage = np.mean(cpu_usage_data) if cpu_usage_data else 0
    max_cpu_usage = np.max(cpu_usage_data) if cpu_usage_data else 0
    avg_memory_usage = np.mean(memory_usage_data) if memory_usage_data else 0
    max_memory_usage = np.max(memory_usage_data) if memory_usage_data else 0
    
    avg_gpu_allocated = np.mean([d['allocated'] for d in gpu_memory_data]) if gpu_memory_data else 0
    max_gpu_allocated = np.max([d['allocated'] for d in gpu_memory_data]) if gpu_memory_data else 0
    avg_gpu_reserved = np.mean([d['reserved'] for d in gpu_memory_data]) if gpu_memory_data else 0
    max_gpu_reserved = np.max([d['reserved'] for d in gpu_memory_data]) if gpu_memory_data else 0
    
    # Calculate cumulative statistics from SimCLR
    total_victim_flops = timing_metrics.get('total_victim_flops', 0)
    total_surrogate_flops = timing_metrics.get('total_surrogate_flops', 0)
    total_flops = total_victim_flops + total_surrogate_flops
    avg_training_loss = np.mean(timing_metrics['training_losses']) if timing_metrics.get('training_losses') else 0
    final_training_loss = timing_metrics['training_losses'][-1] if timing_metrics.get('training_losses') else 0
    
    # Comprehensive performance summary
    performance_summary = {
        # System information
        'system_info': timing_metrics['system_info'],
        
        # Overall timing
        'total_execution_time': total_execution_time,
        'model_setup_time': timing_metrics['model_setup_time'],
        'total_training_time': timing_metrics.get('total_training_time', 0),
        
        # Training breakdown
        'avg_epoch_time': np.mean(timing_metrics['epoch_times']) if timing_metrics.get('epoch_times') else 0,
        'total_victim_forward_time': timing_metrics.get('total_victim_forward_time', 0),
        'total_surrogate_forward_time': timing_metrics.get('total_surrogate_forward_time', 0),
        'total_loss_computation_time': timing_metrics.get('total_loss_computation_time', 0),
        'total_backward_pass_time': timing_metrics.get('total_backward_pass_time', 0),
        
        # Model statistics
        'victim_model_stats': timing_metrics['victim_model_stats'],
        'surrogate_model_stats': timing_metrics['surrogate_model_stats'],
        
        # Computational metrics
        'victim_flops_per_sample': timing_metrics.get('victim_flops_per_sample', 0),
        'surrogate_flops_per_sample': timing_metrics.get('surrogate_flops_per_sample', 0),
        'total_victim_flops': total_victim_flops,
        'total_surrogate_flops': total_surrogate_flops,
        'total_flops': total_flops,
        'total_victim_queries': timing_metrics.get('total_victim_queries', 0),
        
        # System resource usage
        'system_resources': {
            'avg_cpu_usage_percent': avg_cpu_usage,
            'max_cpu_usage_percent': max_cpu_usage,
            'avg_memory_usage_percent': avg_memory_usage,
            'max_memory_usage_percent': max_memory_usage,
            'avg_gpu_memory_allocated_gb': avg_gpu_allocated,
            'max_gpu_memory_allocated_gb': max_gpu_allocated,
            'avg_gpu_memory_reserved_gb': avg_gpu_reserved,
            'max_gpu_memory_reserved_gb': max_gpu_reserved,
        },
        
        # Training statistics
        'training_stats': {
            'num_epochs': timing_metrics.get('num_epochs', args.epochs),
            'avg_training_loss': avg_training_loss,
            'final_training_loss': final_training_loss,
            'target_dataset': args.dataset,
            'surrogate_dataset': args.datasetsteal,
            'victim_architecture': args.victim_arch,
            'surrogate_architecture': args.surrogate_arch,
            'query_limit': args.num_queries,
            'loss_type': args.losstype,
        }
    }
    
    # Print comprehensive summary (SAME FORMAT AS CONTRASTIVE STEAL)
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY - SSL STEAL")
    print("="*80)
    
    # System Information Summary
    print("System Configuration:")
    print(f"  CPU: {system_info.get('cpu_model', 'Unknown')}")
    print(f"  Current GPU: {system_info.get('current_gpu', 'Unknown')} ({system_info.get('current_gpu_memory', 'Unknown')})")
    print(f"  RAM: {system_info.get('total_ram_gb', 0):.1f} GB")
    print(f"  Platform: {system_info.get('platform', 'Unknown')}")
    print()
    
    print(f"Total Execution Time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
    print(f"  - Model Setup: {performance_summary['model_setup_time']:.2f}s")
    print(f"  - Training/Stealing: {performance_summary['total_training_time']:.2f}s")
    print(f"\nVictim Querying Details:")
    print(f"  - Victim queries: {timing_metrics.get('total_victim_queries', 0):,} forward passes")
    print(f"  - Victim FLOPs: {total_victim_flops:,} ({total_victim_flops/1e9:.2f} GFLOPs)")
    print(f"  - Loss type: {args.losstype}")
    print(f"\nSurrogate Training Details:")
    print(f"  - Training epochs: {timing_metrics.get('num_epochs', args.epochs)}")
    print(f"  - Avg epoch time: {performance_summary['avg_epoch_time']:.2f}s")
    print(f"  - Total surrogate FLOPs: {total_surrogate_flops:,} ({total_surrogate_flops/1e9:.2f} GFLOPs)")
    print(f"  - Combined FLOPs (victim + surrogate): {total_flops:,} ({total_flops/1e9:.2f} GFLOPs)")
    if final_training_loss > 0:
        print(f"  - Final training loss: {final_training_loss:.6f}")
    print(f"\nSystem Resources:")
    print(f"  - Average CPU usage: {avg_cpu_usage:.1f}%")
    print(f"  - Peak CPU usage: {max_cpu_usage:.1f}%")
    print(f"  - Average GPU memory: {avg_gpu_allocated:.2f}GB")
    print(f"  - Peak GPU memory: {max_gpu_allocated:.2f}GB")
    
    # Save detailed timing metrics (SAME FORMAT AS CONTRASTIVE STEAL)
    timing_filename = f'{args.log_results_dir}/comprehensive_timing_metrics_seed{args.seed}.json'
    with open(timing_filename, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_compatible_metrics = {}
        for key, value in performance_summary.items():
            if isinstance(value, np.ndarray):
                json_compatible_metrics[key] = value.tolist()
            elif isinstance(value, np.number):
                json_compatible_metrics[key] = value.item()
            else:
                json_compatible_metrics[key] = value
        json.dump(json_compatible_metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {timing_filename}")
    
    # Save per-epoch timing data if available
    if timing_metrics.get('epoch_times'):
        epoch_data = {
            'epoch': range(1, len(timing_metrics['epoch_times']) + 1),
            'total_time': timing_metrics['epoch_times'],
            'victim_forward_time': timing_metrics.get('victim_forward_times', [0] * len(timing_metrics['epoch_times'])),
            'surrogate_forward_time': timing_metrics.get('surrogate_forward_times', [0] * len(timing_metrics['epoch_times'])),
            'loss_computation_time': timing_metrics.get('loss_computation_times', [0] * len(timing_metrics['epoch_times'])),
            'backward_pass_time': timing_metrics.get('backward_pass_times', [0] * len(timing_metrics['epoch_times'])),
            'training_loss': timing_metrics.get('training_losses', [0] * len(timing_metrics['epoch_times'])),
            'victim_queries': timing_metrics.get('victim_queries_per_epoch', [0] * len(timing_metrics['epoch_times'])),
            'victim_flops': timing_metrics.get('victim_flops_per_epoch', [0] * len(timing_metrics['epoch_times'])),
            'surrogate_flops': timing_metrics.get('surrogate_flops_per_epoch', [0] * len(timing_metrics['epoch_times'])),
        }
        epoch_data['total_flops'] = [v + s for v, s in zip(epoch_data['victim_flops'], epoch_data['surrogate_flops'])]
        
        if pd is not None:
            epoch_timing_df = pd.DataFrame(epoch_data)
            epoch_timing_filename = f'{args.log_results_dir}/per_epoch_timing_seed{args.seed}.csv'
            epoch_timing_df.to_csv(epoch_timing_filename, index=False)
            print(f"Per-epoch timing data saved to: {epoch_timing_filename}")
        else:
            # Save as JSON if pandas is not available
            epoch_timing_filename = f'{args.log_results_dir}/per_epoch_timing_seed{args.seed}.json'
            with open(epoch_timing_filename, 'w') as f:
                json.dump(epoch_data, f, indent=2)
            print(f"Per-epoch timing data saved to: {epoch_timing_filename} (JSON format - pandas not available)")
    
    print("="*80)



if __name__ == "__main__":
    main()
    