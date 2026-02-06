"""
Generate FLOPs comparison table for different dataset sizes
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

def generate_table(dataset_sizes):
    """Generate comprehensive comparison table"""
    
    print("\n" + "="*150)
    print("FLOPS ANALYSIS - COMPARISON ACROSS DATASET SIZES")
    print(f"Configuration: Batch Size={BATCH_SIZE}, Epochs={NUM_EPOCHS}, Model=ResNet34")
    print("="*150)
    print()
    
    # Calculate for all attacks and dataset sizes
    results = {}
    for size in dataset_sizes:
        results[size] = {
            'sslsteal': calculate_attack_flops(size, 'sslsteal'),
            'contsteal': calculate_attack_flops(size, 'contsteal'),
            'rda': calculate_attack_flops(size, 'rda')
        }
    
    # Table 1: Victim Querying Cost
    print("TABLE 1: VICTIM QUERYING COST (TFLOPs)")
    print("-" * 170)
    print(f"{'Dataset Size':<15} {'SSL Stealing':<25} {'Contrastive Steal':<25} {'RDA':<25} {'SSL Queries':<20} {'RDA Queries':<20} {'Reduction':<15}")
    print("-" * 170)
    
    for size in dataset_sizes:
        ssl_victim = results[size]['sslsteal']['victim_flops'] / 1e12
        const_victim = results[size]['contsteal']['victim_flops'] / 1e12
        rda_victim = results[size]['rda']['victim_flops'] / 1e12
        ssl_queries = results[size]['sslsteal']['victim_queries']
        rda_queries = results[size]['rda']['victim_queries']
        reduction = (1 - rda_queries / ssl_queries) * 100 if ssl_queries > 0 else 0
        
        print(f"{size:<15,} {ssl_victim:<25,.2f} {const_victim:<25,.2f} {rda_victim:<25,.2f} {ssl_queries:<20,} {rda_queries:<20,} {reduction:<15.1f}%")
    
    print("-" * 170)
    print()
    
    # Table 2: Total FLOPs
    print("TABLE 2: TOTAL COMPUTATIONAL COST (TFLOPs)")
    print("-" * 150)
    print(f"{'Dataset Size':<15} {'SSL Stealing':<25} {'Contrastive Steal':<25} {'RDA':<25} {'RDA/SSL Ratio':<15}")
    print("-" * 150)
    
    for size in dataset_sizes:
        ssl_flops = results[size]['sslsteal']['total_flops'] / 1e12
        const_flops = results[size]['contsteal']['total_flops'] / 1e12
        rda_flops = results[size]['rda']['total_flops'] / 1e12
        ratio = rda_flops / ssl_flops if ssl_flops > 0 else 0
        
        print(f"{size:<15,} {ssl_flops:<25,.2f} {const_flops:<25,.2f} {rda_flops:<25,.2f} {ratio:<15.2f}x")
    
    print("-" * 150)
    print()
    
    
    # Table 3: Surrogate Training Cost
    print("TABLE 3: SURROGATE TRAINING COST (TFLOPs)")
    print("-" * 150)
    print(f"{'Dataset Size':<15} {'SSL Stealing':<25} {'Contrastive Steal':<25} {'RDA':<25} {'RDA/SSL Ratio':<15}")
    print("-" * 150)
    
    for size in dataset_sizes:
        ssl_surrogate = results[size]['sslsteal']['surrogate_flops'] / 1e12
        const_surrogate = results[size]['contsteal']['surrogate_flops'] / 1e12
        rda_surrogate = results[size]['rda']['surrogate_flops'] / 1e12
        ratio = rda_surrogate / ssl_surrogate if ssl_surrogate > 0 else 0
        
        print(f"{size:<15,} {ssl_surrogate:<25,.2f} {const_surrogate:<25,.2f} {rda_surrogate:<25,.2f} {ratio:<15.2f}x")
    
    print("-" * 150)
    print()
    
    # Table 4: Breakdown by Component
    print("TABLE 4: DETAILED BREAKDOWN FOR EACH DATASET SIZE")
    print("="*150)
    
    for size in dataset_sizes:
        print(f"\nDataset Size: {size:,} samples")
        print("-" * 150)
        print(f"{'Attack':<20} {'Victim FLOPs':<20} {'Surrogate FLOPs':<20} {'Loss FLOPs':<20} {'Total FLOPs':<20}")
        print("-" * 150)
        
        for attack in ['sslsteal', 'contsteal', 'rda']:
            attack_name = {'sslsteal': 'SSL Stealing', 'contsteal': 'ContSteal', 'rda': 'RDA'}[attack]
            victim = results[size][attack]['victim_flops'] / 1e12
            surrogate = results[size][attack]['surrogate_flops'] / 1e12
            loss = results[size][attack]['loss_flops'] / 1e12
            total = results[size][attack]['total_flops'] / 1e12
            
            print(f"{attack_name:<20} {victim:<20.2f} {surrogate:<20.2f} {loss:<20.2f} {total:<20.2f}")
        
        print("-" * 150)
    
    print()
    print("="*150)
    
    # Summary insights
    print("\nKEY INSIGHTS:")
    print("-" * 150)
    print(f"1. SSL Stealing and ContSteal have IDENTICAL costs (both query victim every epoch)")
    print(f"2. RDA victim queries are CONSTANT ({RDA_PROTO_PATCH}x dataset size) regardless of epochs")
    print(f"3. RDA reduces victim queries by ~{(1 - RDA_PROTO_PATCH/NUM_EPOCHS)*100:.1f}% compared to SSL/ContSteal")
    print(f"4. RDA's higher total cost comes from {RDA_QUERY_PATCH}x augmentations during surrogate training")
    print(f"5. As dataset size increases, victim querying cost scales linearly for all attacks")
    print(f"6. RDA's relative efficiency (total FLOPs ratio) remains ~constant across dataset sizes")
    print("-" * 150)
    print()

if __name__ == "__main__":
    # Dataset sizes to analyze
    dataset_sizes = [256, 512, 1536, 2500, 5750, 7375, 9000]
    
    print("\n")
    print("╔" + "="*148 + "╗")
    print("║" + " "*148 + "║")
    print("║" + " "*35 + "SSL MODEL STEALING ATTACKS - FLOPS COMPARISON TABLE" + " "*61 + "║")
    print("║" + " "*148 + "║")
    print("╚" + "="*148 + "╝")
    
    generate_table(dataset_sizes)
    
    print("\n" + "="*150)
    print("ANALYSIS COMPLETE")
    print("="*150)
    print()
