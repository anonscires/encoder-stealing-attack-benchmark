import sys
import os

# Add the parent directory of 'new_attacks' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import argparse
from surrogate_model import Surrogate_model
from src.contsteal.contsteal_utils import load_dataset
from Loss import ContrastiveLoss
from train_representation import train_representation,train_represnetation_linear
from train_posteriors import train_posterior
from test_target import test_for_target
from test_last import test_onehot
from train_onehot import train_onehot
from train_posteriors import train_posterior
from test_target import test_for_target
import numpy as np
from src.contsteal.contsteal_utils import load_target_model,load_dataset
import dataloader
from test_target import test_for_target
import torchvision
from Linear import linear
import os
from PIL import Image
import requests
import timm
from tqdm import tqdm
import json
import time
import psutil
import threading
from collections import defaultdict
import subprocess
import platform
import torch.nn.functional as F
import torch.nn as nn

# from models.resnet import ResNetEncoder
from src.models.resnet import ResNetGradCamEncoder, ResNetEncoder
from src.models.dino import DinoEncoder
from src.sslsteal.sslsteal_utils import load_victim

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


def enhanced_train_representation(target_encoder, surrogate_model, train_loader, criterion, optimizer, device, epoch, timing_metrics, args):
    """Enhanced training with comprehensive metrics similar to RDA"""
    surrogate_model.train()
    target_encoder.eval()
    
    loss_epoch = 0
    mse_epoch = 0
    total_queries = 0
    
    # Timing metrics for this epoch
    epoch_start_time = time.time()
    data_loading_time = 0
    forward_pass_time = 0
    victim_forward_time = 0
    loss_computation_time = 0
    backward_pass_time = 0
    
    victim_queries_this_epoch = 0
    victim_flops_this_epoch = 0
    surrogate_flops_this_epoch = 0
    
    # Estimate FLOPs for victim and surrogate models (done once per epoch)
    if epoch == 1:
        sample_batch = next(iter(train_loader))[0]
        sample_input = sample_batch[:1]  # Single sample
        victim_flops_per_sample = get_model_flops(target_encoder, sample_input.shape)
        surrogate_flops_per_sample = get_model_flops(surrogate_model, sample_input.shape)
        timing_metrics['victim_flops_per_sample'] = victim_flops_per_sample
        timing_metrics['surrogate_flops_per_sample'] = surrogate_flops_per_sample
        print(f"Victim model FLOPs per sample: {victim_flops_per_sample:,}")
        print(f"Surrogate model FLOPs per sample: {surrogate_flops_per_sample:,}")
    
    for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Training Epoch [{epoch}] (ContraSteal - Querying victim every batch)")):
        batch_start_time = time.time()
        
        x = x.to(device)
        y = y.to(device)
        data_load_time = time.time() - batch_start_time
        data_loading_time += data_load_time
        
        batch_size = len(x)
        
        # Victim forward pass timing (QUERIES VICTIM EVERY BATCH)
        victim_start_time = time.time()
        with torch.no_grad():
            target_encoder.requires_grad = False
            re = target_encoder(x)
        victim_forward_time += time.time() - victim_start_time
        victim_queries_this_epoch += batch_size
        if 'victim_flops_per_sample' in timing_metrics:
            victim_flops_this_epoch += timing_metrics['victim_flops_per_sample'] * batch_size
        
        # Surrogate forward pass timing
        forward_start_time = time.time()
        optimizer.zero_grad()
        su_output = surrogate_model(x)
        forward_pass_time += time.time() - forward_start_time
        if 'surrogate_flops_per_sample' in timing_metrics:
            surrogate_flops_this_epoch += timing_metrics['surrogate_flops_per_sample'] * batch_size
        
        # Loss computation timing
        loss_start_time = time.time()
        loss = criterion(su_output, re)
        mse = F.mse_loss(su_output, re)
        loss_computation_time += time.time() - loss_start_time
        
        # Backward pass timing
        backward_start_time = time.time()
        loss.backward()
        optimizer.step()
        backward_pass_time += time.time() - backward_start_time
        
        loss_epoch += loss.item()
        mse_epoch += mse.item()
        total_queries += batch_size
    
    epoch_total_time = time.time() - epoch_start_time
    avg_loss = loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
    avg_mse = mse_epoch / len(train_loader) if len(train_loader) > 0 else 0
    
    # Store timing metrics
    timing_metrics['epoch_times'].append(epoch_total_time)
    timing_metrics['data_loading_times'].append(data_loading_time)
    timing_metrics['forward_pass_times'].append(forward_pass_time)
    timing_metrics['victim_forward_times'].append(victim_forward_time)
    timing_metrics['loss_computation_times'].append(loss_computation_time)
    timing_metrics['backward_pass_times'].append(backward_pass_time)
    timing_metrics['training_losses'].append(avg_loss)
    timing_metrics['training_mse'].append(avg_mse)
    timing_metrics['victim_queries_per_epoch'].append(victim_queries_this_epoch)
    timing_metrics['victim_flops_per_epoch'].append(victim_flops_this_epoch)
    timing_metrics['surrogate_flops_per_epoch'].append(surrogate_flops_this_epoch)
    
    print(f'Training Epoch {epoch} - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}', flush=True)
    print(f'Epoch {epoch} timing - Total: {epoch_total_time:.2f}s, Data: {data_loading_time:.2f}s, Victim: {victim_forward_time:.2f}s, Surrogate: {forward_pass_time:.2f}s, Loss: {loss_computation_time:.2f}s, Backward: {backward_pass_time:.2f}s', flush=True)
    print(f'Victim queries this epoch: {victim_queries_this_epoch:,}, Total FLOPs: {victim_flops_this_epoch + surrogate_flops_this_epoch:,} ({(victim_flops_this_epoch + surrogate_flops_this_epoch)/1e9:.2f} GFLOPs)', flush=True)
    print()
    
    return total_queries


def main():
    torch.set_num_threads(1)   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type',default='simclr',type=str)
    parser.add_argument('--pretrain',default='cifar10',type=str)
    parser.add_argument('--target-dataset',default='cifar10',type=str)
    parser.add_argument('--surrogate-dataset',default='cifar10',type=str)
    parser.add_argument('--augmentation',default=2,type=int)
    parser.add_argument('--surrogate-arch',default='resnet18',type=str)
    parser.add_argument('--epoch',default= 100, type = int)

    parser.add_argument('--victim-arch', default='resnet34', type=str, choices=['resnet18', 'resnet34'])
    parser.add_argument('--victim-head', default="False", type=str, help='To use victim emebedding head or not.')
    parser.add_argument('--surrogate-head', default="False", type=str, help='To use surrogate emebedding head or not.')
    parser.add_argument('--out-dim', default=128, type=int, help='Embedding dimension size of victim model')
    parser.add_argument('--victim-path', type=str, help='Path to victim model.')
    parser.add_argument('--save-path', type=str, help='Path to stolen model.')
    parser.add_argument('--gpu-index', type=int, default=0, help='GPU index to use.')
    parser.add_argument('--gradcam', type=str, default="False", help='Use GradCam for victim model.')
    parser.add_argument('--num-query', type=int, default=9000, help='Number of query samples to use.')

    # Add logging arguments similar to RDA
    parser.add_argument('--log-results-dir', default='logs/contsteal', type=str, help='Path to save logs')
    parser.add_argument('--seed', default=100, type=int, help='Random seed for reproducibility')

    args = parser.parse_args()
    print("Arguments:", args)
    
    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    device = torch.device(f"cuda:{args.gpu_index}") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Device:{device} will be used.')
    
    # Create log directory
    if not os.path.exists(args.log_results_dir):
        os.makedirs(args.log_results_dir, exist_ok=True)

    # ========================= COMPREHENSIVE PERFORMANCE MONITORING =========================
    print("="*80)
    print("Starting Comprehensive Performance Monitoring for ContraSteal Attack")
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
    
    catagory_num = 10

    victim_head = (args.victim_head == "True")
    if "resnet" in args.victim_arch:
        if args.gradcam == "True":
            victim_model = ResNetGradCamEncoder(base_model=args.victim_arch,
                                                out_dim=args.out_dim,loss=None, include_mlp = victim_head).to(device)
        else:
            victim_model = ResNetEncoder(base_model=args.victim_arch,
                                        out_dim=args.out_dim,loss=None, include_mlp = victim_head).to(device)
        victim_model = load_victim(args.victim_path, victim_model, device=device, discard_mlp = True)
    
    elif "vit" in args.victim_arch:
        if args.gradcam == "True":
            raise NotImplementedError("GradCam not implemented for ViT models")
        else:
            victim_model = DinoEncoder(base_model=args.victim_arch, out_dim=args.out_dim,loss=None, include_mlp = victim_head)
            victim_model = load_victim(args.victim_path, victim_model, device=device, discard_mlp = True)
            victim_model.to(device).eval()
    
    target_encoder = load_victim(args.victim_path, victim_model, device=device, discard_mlp = not victim_head)
    target_encoder.to(device)

    surrogate_head = (args.surrogate_head == "True")
    if 'resnet' in args.surrogate_arch:
        surrogate_model = ResNetEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim, loss=None, include_mlp=surrogate_head)
    elif 'vit' in args.surrogate_arch:
        surrogate_model = DinoEncoder(base_model=args.surrogate_arch, out_dim=args.out_dim, loss=None, include_mlp=surrogate_head)
    surrogate_model.to(device)

    setup_end_time = time.time()
    model_setup_time = setup_end_time - setup_start_time

    # Get model statistics
    victim_stats = get_model_stats(target_encoder)
    surrogate_stats = get_model_stats(surrogate_model)
    
    print(f"Model setup completed in {model_setup_time:.2f} seconds")
    print(f"Victim Model Stats - Params: {victim_stats['total_params']:,}, Size: {victim_stats['model_size_mb']:.2f}MB")
    print(f"Surrogate Model Stats - Params: {surrogate_stats['total_params']:,}, Size: {surrogate_stats['model_size_mb']:.2f}MB")
    
    timing_metrics['victim_model_stats'] = victim_stats
    timing_metrics['surrogate_model_stats'] = surrogate_stats
    timing_metrics['model_setup_time'] = model_setup_time
    
    # Initialize timing metric lists
    timing_metrics['epoch_times'] = []
    timing_metrics['data_loading_times'] = []
    timing_metrics['forward_pass_times'] = []
    timing_metrics['victim_forward_times'] = []
    timing_metrics['loss_computation_times'] = []
    timing_metrics['backward_pass_times'] = []
    timing_metrics['training_losses'] = []
    timing_metrics['training_mse'] = []
    timing_metrics['victim_queries_per_epoch'] = []
    timing_metrics['victim_flops_per_epoch'] = []
    timing_metrics['surrogate_flops_per_epoch'] = []

    # ========================= DATA LOADING =========================
    data_loading_start_time = time.time()
    
    train_dataset,test_dataset,linear_dataset = load_dataset(args.pretrain,args.target_dataset,args.surrogate_dataset,args.augmentation,1,num_query=args.num_query)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    linear_loader = torch.utils.data.DataLoader(
        linear_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    
    data_loading_end_time = time.time()
    data_loading_setup_time = data_loading_end_time - data_loading_start_time
    
    print(f"Data loading completed in {data_loading_setup_time:.2f} seconds")
    timing_metrics['data_loading_setup_time'] = data_loading_setup_time

    # ========================= TRAINING PHASE =========================
    print(Fore.CYAN + "Phase: ContraSteal Training")
    print("Note: Victim is queried EVERY EPOCH during training (unlike RDA)")
    
    criterion = ContrastiveLoss(128, device)
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=3e-4)
    
    training_start_time = time.time()
    total_victim_queries = 0
    
    for i in range(1, args.epoch + 1):
        queries_this_epoch = enhanced_train_representation(
            target_encoder, surrogate_model, train_loader, criterion, optimizer, device, i, timing_metrics, args
        )
        total_victim_queries += timing_metrics['victim_queries_per_epoch'][-1]
        
        if i == 10:
            avg_epoch_time = sum(timing_metrics['epoch_times'][:10]) / 10
            print(f'Average time per epoch (first 10): {avg_epoch_time:.2f} seconds', flush=True)
    
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    print(f"Total training completed in {total_training_time:.2f} seconds")
    timing_metrics['total_training_time'] = total_training_time
    timing_metrics['total_victim_queries'] = total_victim_queries
    
    # ========================= MODEL SAVING =========================
    os.makedirs(args.save_path, exist_ok=True)
    torch.save({
                'epoch': args.epoch,
                'arch': args.surrogate_arch,
                'state_dict': surrogate_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            args.save_path+'/srgt_of_'+args.victim_path.split('/')[-1])

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
    
    # Calculate cumulative statistics
    total_victim_flops = sum(timing_metrics['victim_flops_per_epoch'])
    total_surrogate_flops = sum(timing_metrics['surrogate_flops_per_epoch'])
    total_flops = total_victim_flops + total_surrogate_flops
    avg_training_loss = np.mean(timing_metrics['training_losses'])
    final_training_loss = timing_metrics['training_losses'][-1]
    avg_training_mse = np.mean(timing_metrics['training_mse'])
    final_training_mse = timing_metrics['training_mse'][-1]
    
    # Comprehensive performance summary
    performance_summary = {
        # System information
        'system_info': timing_metrics['system_info'],
        
        # Overall timing
        'total_execution_time': total_execution_time,
        'model_setup_time': timing_metrics['model_setup_time'],
        'data_loading_setup_time': timing_metrics['data_loading_setup_time'],
        'total_training_time': timing_metrics['total_training_time'],
        
        # Training breakdown
        'avg_epoch_time': np.mean(timing_metrics['epoch_times']),
        'total_data_loading_time': sum(timing_metrics['data_loading_times']),
        'total_victim_forward_time': sum(timing_metrics['victim_forward_times']),
        'total_surrogate_forward_time': sum(timing_metrics['forward_pass_times']),
        'total_loss_computation_time': sum(timing_metrics['loss_computation_times']),
        'total_backward_pass_time': sum(timing_metrics['backward_pass_times']),
        
        # Model statistics
        'victim_model_stats': timing_metrics['victim_model_stats'],
        'surrogate_model_stats': timing_metrics['surrogate_model_stats'],
        
        # Computational metrics - KEY DIFFERENCE FROM RDA
        'victim_flops_per_sample': timing_metrics.get('victim_flops_per_sample', 0),
        'surrogate_flops_per_sample': timing_metrics.get('surrogate_flops_per_sample', 0),
        'total_victim_flops': total_victim_flops,
        'total_surrogate_flops': total_surrogate_flops,
        'total_flops': total_flops,
        'total_victim_queries': timing_metrics['total_victim_queries'],
        
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
            'num_epochs': len(timing_metrics['epoch_times']),
            'avg_training_loss': avg_training_loss,
            'final_training_loss': final_training_loss,
            'avg_training_mse': avg_training_mse,
            'final_training_mse': final_training_mse,
            'target_dataset': args.target_dataset,
            'surrogate_dataset': args.surrogate_dataset,
            'victim_architecture': args.victim_arch,
            'surrogate_architecture': args.surrogate_arch,
            'query_limit': args.num_query,
        }
    }
    
    # Print comprehensive summary (SAME FORMAT AS RDA)
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY - CONTRASTIVE STEAL")
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
    print(f"  - Data Loading & Setup: {performance_summary['data_loading_setup_time']:.2f}s")
    print(f"  - Training: {performance_summary['total_training_time']:.2f}s")
    print(f"\nVictim Querying Details:")
    print(f"  - Victim queries: {timing_metrics['total_victim_queries']:,} forward passes (EVERY EPOCH)")
    print(f"  - Victim FLOPs: {total_victim_flops:,} ({total_victim_flops/1e9:.2f} GFLOPs)")
    print(f"\nSurrogate Training Details:")
    print(f"  - Training epochs: {len(timing_metrics['epoch_times'])}")
    print(f"  - Avg epoch time: {performance_summary['avg_epoch_time']:.2f}s")
    print(f"  - Total surrogate FLOPs: {total_surrogate_flops:,} ({total_surrogate_flops/1e9:.2f} GFLOPs)")
    print(f"  - Combined FLOPs (victim + surrogate): {total_flops:,} ({total_flops/1e9:.2f} GFLOPs)")
    print(f"  - Final training loss: {final_training_loss:.6f}")
    print(f"  - Final training MSE: {final_training_mse:.6f}")
    print(f"\nSystem Resources:")
    print(f"  - Average CPU usage: {avg_cpu_usage:.1f}%")
    print(f"  - Peak CPU usage: {max_cpu_usage:.1f}%")
    print(f"  - Average GPU memory: {avg_gpu_allocated:.2f}GB")
    print(f"  - Peak GPU memory: {max_gpu_allocated:.2f}GB")
    
    # Save detailed timing metrics (SAME FORMAT AS RDA)
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
    
    # Save per-epoch timing data (SAME FORMAT AS RDA) - with pandas fallback
    epoch_data = {
        'epoch': range(1, len(timing_metrics['epoch_times']) + 1),
        'total_time': timing_metrics['epoch_times'],
        'data_loading_time': timing_metrics['data_loading_times'],
        'victim_forward_time': timing_metrics['victim_forward_times'],
        'surrogate_forward_time': timing_metrics['forward_pass_times'],
        'loss_computation_time': timing_metrics['loss_computation_times'],
        'backward_pass_time': timing_metrics['backward_pass_times'],
        'training_loss': timing_metrics['training_losses'],
        'training_mse': timing_metrics['training_mse'],
        'victim_queries': timing_metrics['victim_queries_per_epoch'],
        'victim_flops': timing_metrics['victim_flops_per_epoch'],
        'surrogate_flops': timing_metrics['surrogate_flops_per_epoch'],
        'total_flops': [v + s for v, s in zip(timing_metrics['victim_flops_per_epoch'], timing_metrics['surrogate_flops_per_epoch'])]
    }
    
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
    
    # Dump args (SAME AS RDA)
    with open(args.log_results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    
    print(f"\nModel saved to: {args.save_path}/srgt_of_{args.victim_path.split('/')[-1]}")
    print("="*80)


if __name__ == "__main__":
    main()
