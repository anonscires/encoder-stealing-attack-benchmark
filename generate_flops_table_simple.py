"""
Generate FLOPs comparison table for different dataset sizes
Simplified format with three main tables: Victim, Surrogate, and Total
"""

import math

# Model configurations
RESNET34_FLOPS_CIFAR = 1.16e9  # FLOPs per forward pass (32x32)
VICTIM_FLOPS_PER_SAMPLE = RESNET34_FLOPS_CIFAR
SURROGATE_FLOPS_PER_SAMPLE = RESNET34_FLOPS_CIFAR

# RDA specific parameters
RDA_PROTO_PATCH = 10
RDA_QUERY_PATCH = 5

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 200
BACKWARD_MULTIPLIER = 2.0
EMBEDDING_DIM = 512

def calculate_infonce_loss_flops(batch_size, embedding_dim=512):
    """Calculate FLOPs for InfoNCE loss"""
    matmul_flops = batch_size * batch_size * embedding_dim * 2
    other_ops = batch_size * batch_size * 5
    return matmul_flops + other_ops

def calculate_rda_loss_flops(batch_size, embedding_dim=512):
    """Calculate FLOPs for RDA custom loss (contrastive + alignment)"""
    contrast_flops = calculate_infonce_loss_flops(batch_size, embedding_dim) * 1.5
    align_flops = batch_size * embedding_dim * 8
    return contrast_flops + align_flops

def calculate_attack_flops(dataset_size, attack_type):
    """Calculate total FLOPs for a given dataset size and attack type"""
    num_batches = math.ceil(dataset_size / BATCH_SIZE)
    
    if attack_type in ['sslsteal', 'contsteal']:
        # Victim queried every epoch
        victim_forward_passes = dataset_size * NUM_EPOCHS
        victim_forward_flops = victim_forward_passes * VICTIM_FLOPS_PER_SAMPLE
        
        # Surrogate training
        surrogate_forward_passes = dataset_size * NUM_EPOCHS
        surrogate_forward_flops = surrogate_forward_passes * SURROGATE_FLOPS_PER_SAMPLE
        
        # Loss computation
        loss_computation_flops = num_batches * NUM_EPOCHS * calculate_infonce_loss_flops(BATCH_SIZE)
        
        # Backward passes
        surrogate_backward_flops = surrogate_forward_flops * BACKWARD_MULTIPLIER
        
        total_flops = (victim_forward_flops + surrogate_forward_flops + 
                       loss_computation_flops + surrogate_backward_flops)
        
        return {
            'victim_flops': victim_forward_flops,
            'surrogate_flops': surrogate_forward_flops + surrogate_backward_flops,
            'loss_flops': loss_computation_flops,
            'total_flops': total_flops,
            'victim_queries': victim_forward_passes
        }
    
    elif attack_type == 'rda':
        # Phase 1: Prototype extraction (victim queried once)
        victim_forward_passes_proto = dataset_size * RDA_PROTO_PATCH
        victim_forward_flops_proto = victim_forward_passes_proto * VICTIM_FLOPS_PER_SAMPLE
        averaging_flops = dataset_size * EMBEDDING_DIM * RDA_PROTO_PATCH
        
        # Phase 2: Surrogate training (victim not queried)
        surrogate_forward_passes_total = dataset_size * RDA_QUERY_PATCH * NUM_EPOCHS
        surrogate_forward_flops = surrogate_forward_passes_total * SURROGATE_FLOPS_PER_SAMPLE
        
        loss_computations_per_epoch = num_batches * RDA_QUERY_PATCH
        total_loss_computations = loss_computations_per_epoch * NUM_EPOCHS
        loss_computation_flops = total_loss_computations * calculate_rda_loss_flops(BATCH_SIZE)
        
        surrogate_backward_flops = surrogate_forward_flops * BACKWARD_MULTIPLIER
        
        total_flops = (victim_forward_flops_proto + averaging_flops + 
                       surrogate_forward_flops + loss_computation_flops + 
                       surrogate_backward_flops)
        
        return {
            'victim_flops': victim_forward_flops_proto,
            'surrogate_flops': surrogate_forward_flops + surrogate_backward_flops,
            'loss_flops': loss_computation_flops + averaging_flops,
            'total_flops': total_flops,
            'victim_queries': victim_forward_passes_proto
        }

def generate_tables(dataset_sizes):
    """Generate three main tables: Victim, Surrogate, and Total TFLOPs"""
    
    print("\n" + "="*120)
    print("FLOPS ANALYSIS - COMPARISON ACROSS DATASET SIZES")
    print(f"Configuration: Batch Size={BATCH_SIZE}, Epochs={NUM_EPOCHS}, Model=ResNet34")
    print("="*120)
    print()
    
    # Calculate for all attacks and dataset sizes
    results = {}
    for size in dataset_sizes:
        results[size] = {
            'sslsteal': calculate_attack_flops(size, 'sslsteal'),
            'contsteal': calculate_attack_flops(size, 'contsteal'),
            'rda': calculate_attack_flops(size, 'rda')
        }
    
    # ========================================================================
    # TABLE 1: VICTIM QUERYING COST (TFLOPs)
    # ========================================================================
    print("TABLE 1: VICTIM QUERYING COST (TFLOPs)")
    print("-" * 120)
    print(f"{'Dataset Size':<15} {'SSL Stealing':<25} {'Contrastive Steal':<25} {'RDA':<25} {'RDA Reduction':<15}")
    print("-" * 120)
    
    for size in dataset_sizes:
        ssl_victim = results[size]['sslsteal']['victim_flops'] / 1e12
        const_victim = results[size]['contsteal']['victim_flops'] / 1e12
        rda_victim = results[size]['rda']['victim_flops'] / 1e12
        reduction = (1 - rda_victim / ssl_victim) * 100 if ssl_victim > 0 else 0
        
        print(f"{size:<15,} {ssl_victim:<25.2f} {const_victim:<25.2f} {rda_victim:<25.2f} {reduction:<15.1f}%")
    
    print("-" * 120)
    print()
    
    # ========================================================================
    # TABLE 2: SURROGATE TRAINING COST (TFLOPs)
    # ========================================================================
    print("TABLE 2: SURROGATE TRAINING COST (TFLOPs)")
    print("-" * 120)
    print(f"{'Dataset Size':<15} {'SSL Stealing':<25} {'Contrastive Steal':<25} {'RDA':<25} {'RDA/SSL Ratio':<15}")
    print("-" * 120)
    
    for size in dataset_sizes:
        ssl_surrogate = results[size]['sslsteal']['surrogate_flops'] / 1e12
        const_surrogate = results[size]['contsteal']['surrogate_flops'] / 1e12
        rda_surrogate = results[size]['rda']['surrogate_flops'] / 1e12
        ratio = rda_surrogate / ssl_surrogate if ssl_surrogate > 0 else 0
        
        print(f"{size:<15,} {ssl_surrogate:<25.2f} {const_surrogate:<25.2f} {rda_surrogate:<25.2f} {ratio:<15.2f}x")
    
    print("-" * 120)
    print()
    
    # ========================================================================
    # TABLE 3: TOTAL COMPUTATIONAL COST (TFLOPs)
    # ========================================================================
    print("TABLE 3: TOTAL COMPUTATIONAL COST (TFLOPs)")
    print("-" * 120)
    print(f"{'Dataset Size':<15} {'SSL Stealing':<25} {'Contrastive Steal':<25} {'RDA':<25} {'RDA/SSL Ratio':<15}")
    print("-" * 120)
    
    for size in dataset_sizes:
        ssl_total = results[size]['sslsteal']['total_flops'] / 1e12
        const_total = results[size]['contsteal']['total_flops'] / 1e12
        rda_total = results[size]['rda']['total_flops'] / 1e12
        ratio = rda_total / ssl_total if ssl_total > 0 else 0
        
        print(f"{size:<15,} {ssl_total:<25.2f} {const_total:<25.2f} {rda_total:<25.2f} {ratio:<15.2f}x")
    
    print("-" * 120)
    print()
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    print()
    
    print("Victim Querying Efficiency:")
    print(f"  - RDA reduces victim queries by ~95% compared to SSL/ContSteal")
    print(f"  - SSL/ContSteal query the victim {NUM_EPOCHS}x more (every epoch)")
    print()
    
    print("Computational Cost Trade-offs:")
    print(f"  - RDA has ~3.76x higher total cost due to {RDA_QUERY_PATCH}x augmentations")
    print(f"  - But RDA victim cost is only ~5% of SSL/ContSteal victim cost")
    print(f"  - RDA surrogate training is {RDA_QUERY_PATCH}x more expensive")
    print()
    
    print("Key Insights:")
    print(f"  1. SSL Stealing and ContSteal have IDENTICAL costs")
    print(f"  2. RDA trades higher surrogate training cost for dramatically lower victim queries")
    print(f"  3. All costs scale linearly with dataset size")
    print(f"  4. RDA is ideal when victim access is limited/monitored")
    print(f"  5. SSL/ContSteal are more efficient when victim access is unrestricted")
    print()
    print("="*120)

if __name__ == "__main__":
    # Dataset sizes to analyze
    dataset_sizes = [256, 512, 1536, 2500, 5750, 7375, 9000]
    
    print("\n")
    print("╔" + "="*118 + "╗")
    print("║" + " "*118 + "║")
    print("║" + " "*30 + "SSL MODEL STEALING ATTACKS - FLOPS COMPARISON" + " "*43 + "║")
    print("║" + " "*118 + "║")
    print("╚" + "="*118 + "╝")
    
    generate_tables(dataset_sizes)
    
    print("\nANALYSIS COMPLETE\n")
