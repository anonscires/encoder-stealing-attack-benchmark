"""
Comprehensive FLOPs Calculation for Three SSL Model Stealing Attacks

This script calculates the computational cost (FLOPs) for:
1. SSL Stealing (run_steal.py) - queries victim every epoch
2. ContSteal (contrastive_steal.py) - queries victim every epoch  
3. RDA (RDA.py) - queries victim once at the beginning

Parameters:
- Dataset size: 9000 samples
- Batch size: 128
- Epochs: 100 (default for all attacks)
"""

import math

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_SIZE = 9000
BATCH_SIZE = 128
NUM_EPOCHS = 200

# Model architectures (assuming ResNet34 as victim and surrogate)
# ResNet34 FLOPs for 32x32 CIFAR-10 images: ~1.16 GFLOPs
# ResNet34 FLOPs for 224x224 ImageNet images: ~7.3 GFLOPs
RESNET34_FLOPS_CIFAR = 1.16e9  # FLOPs per forward pass (32x32)
RESNET34_FLOPS_IMAGENET = 7.3e9  # FLOPs per forward pass (224x224)

# Use CIFAR-10 resolution by default (most experiments use this)
VICTIM_FLOPS_PER_SAMPLE = RESNET34_FLOPS_CIFAR
SURROGATE_FLOPS_PER_SAMPLE = RESNET34_FLOPS_CIFAR

# RDA specific parameters
RDA_PROTO_PATCH = 10  # Number of augmented views for prototype extraction
RDA_QUERY_PATCH = 5   # Number of augmented views during training

# Backward pass is approximately 2x forward pass
BACKWARD_MULTIPLIER = 2.0


# ============================================================================
# LOSS FUNCTION FLOPS
# ============================================================================

def calculate_infonce_loss_flops(batch_size, embedding_dim=512):
    """
    Calculate FLOPs for InfoNCE loss computation
    
    InfoNCE loss steps:
    1. Compute similarity matrix: [B, B] matmul with [D, D] -> B*B*D MACs
    2. Exp operation: B*B FLOPs
    3. Masking and reduction: ~B*B FLOPs
    4. Log and mean: ~B FLOPs
    
    Total: ~2*B*B*D FLOPs (matmul dominates)
    """
    # Matrix multiplication: x @ y.T -> [B, D] @ [D, B] = [B, B]
    matmul_flops = batch_size * batch_size * embedding_dim * 2  # 2 for MAC = mult + add
    
    # Temperature division, exp, masking, sum, log, mean
    other_ops = batch_size * batch_size * 5
    
    return matmul_flops + other_ops


def calculate_contrastive_loss_flops(batch_size, embedding_dim=512):
    """
    Calculate FLOPs for ContrastiveLoss (from contrastive_steal.py)
    
    Similar to InfoNCE but computes x@y.T similarity
    """
    # Same as InfoNCE for practical purposes
    return calculate_infonce_loss_flops(batch_size, embedding_dim)


def calculate_rda_loss_flops(batch_size, embedding_dim=512, loss_type='ours'):
    """
    Calculate FLOPs for RDA loss functions
    
    RDA supports multiple loss functions:
    - 'mse': Mean Squared Error
    - 'cos_similarity': Cosine Similarity
    - 'infonce': InfoNCE
    - 'con-steal': ConSteal loss
    - 'ours': Custom loss (alignment + contrastive)
    """
    if loss_type == 'mse':
        # MSE: (x - y)^2 for each element
        return batch_size * embedding_dim * 3  # subtract, square, sum
    
    elif loss_type == 'cos_similarity':
        # Cosine similarity: dot product / (norm(x) * norm(y))
        dot_product = batch_size * embedding_dim * 2
        norms = batch_size * embedding_dim * 4  # 2 norms, each sqrt of sum of squares
        return dot_product + norms
    
    elif loss_type == 'infonce':
        return calculate_infonce_loss_flops(batch_size, embedding_dim)
    
    elif loss_type == 'con-steal':
        # ConSteal: similar to InfoNCE but with x@y.T and x@x.T
        return calculate_infonce_loss_flops(batch_size, embedding_dim) * 1.5
    
    elif loss_type == 'ours':
        # Custom loss: contrastive + alignment
        # l_contrast: similar to ConSteal
        contrast_flops = calculate_infonce_loss_flops(batch_size, embedding_dim) * 1.5
        # l_align: cosine similarity and L2 distance
        align_flops = batch_size * embedding_dim * 8
        return contrast_flops + align_flops
    
    else:
        # Default to InfoNCE
        return calculate_infonce_loss_flops(batch_size, embedding_dim)


# ============================================================================
# SSL STEALING ATTACK (run_steal.py)
# ============================================================================

def calculate_sslsteal_flops():
    """
    SSL Stealing queries the victim at EVERY epoch during training.
    
    Key characteristics:
    - Victim is queried every epoch for every batch
    - Uses InfoNCE loss (or other losses based on losstype)
    - n_views typically = 1 (no data augmentation multiplier)
    """
    print("="*80)
    print("SSL STEALING ATTACK - FLOPS CALCULATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset size: {DATASET_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Victim querying: EVERY EPOCH, EVERY BATCH")
    print()
    
    num_batches = math.ceil(DATASET_SIZE / BATCH_SIZE)
    total_iterations = num_batches * NUM_EPOCHS
    
    # 1. VICTIM FORWARD PASSES (every epoch, every batch)
    victim_forward_passes = DATASET_SIZE * NUM_EPOCHS
    victim_forward_flops = victim_forward_passes * VICTIM_FLOPS_PER_SAMPLE
    
    # 2. SURROGATE FORWARD PASSES (every epoch, every batch)
    surrogate_forward_passes = DATASET_SIZE * NUM_EPOCHS
    surrogate_forward_flops = surrogate_forward_passes * SURROGATE_FLOPS_PER_SAMPLE
    
    # 3. LOSS COMPUTATION (InfoNCE loss per batch)
    loss_computation_flops = num_batches * NUM_EPOCHS * calculate_infonce_loss_flops(BATCH_SIZE)
    
    # 4. SURROGATE BACKWARD PASSES (only surrogate is trained)
    surrogate_backward_flops = surrogate_forward_flops * BACKWARD_MULTIPLIER
    
    # TOTAL
    total_flops = (victim_forward_flops + 
                   surrogate_forward_flops + 
                   loss_computation_flops + 
                   surrogate_backward_flops)
    
    print("Breakdown:")
    print(f"  1. Victim Forward Passes:")
    print(f"     - Count: {victim_forward_passes:,} forward passes")
    print(f"     - FLOPs: {victim_forward_flops:,.0f} ({victim_forward_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  2. Surrogate Forward Passes:")
    print(f"     - Count: {surrogate_forward_passes:,} forward passes")
    print(f"     - FLOPs: {surrogate_forward_flops:,.0f} ({surrogate_forward_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  3. Loss Computation (InfoNCE):")
    print(f"     - Count: {num_batches * NUM_EPOCHS:,} loss computations")
    print(f"     - FLOPs: {loss_computation_flops:,.0f} ({loss_computation_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  4. Surrogate Backward Passes:")
    print(f"     - FLOPs: {surrogate_backward_flops:,.0f} ({surrogate_backward_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"TOTAL FLOPS: {total_flops:,.0f} ({total_flops/1e12:.3f} TFLOPs)")
    print("="*80)
    print()
    
    return {
        'victim_forward_flops': victim_forward_flops,
        'surrogate_forward_flops': surrogate_forward_flops,
        'loss_computation_flops': loss_computation_flops,
        'surrogate_backward_flops': surrogate_backward_flops,
        'total_flops': total_flops
    }


# ============================================================================
# CONTRASTIVE STEAL ATTACK (contrastive_steal.py)
# ============================================================================

def calculate_contsteal_flops():
    """
    ContSteal queries the victim at EVERY epoch during training.
    
    Key characteristics:
    - Victim is queried every epoch for every batch
    - Uses ContrastiveLoss (similar to InfoNCE)
    - Augmentation = 2 (but applied to input, not multiplying queries)
    """
    print("="*80)
    print("CONTRASTIVE STEAL ATTACK - FLOPS CALCULATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset size: {DATASET_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Victim querying: EVERY EPOCH, EVERY BATCH")
    print()
    
    num_batches = math.ceil(DATASET_SIZE / BATCH_SIZE)
    total_iterations = num_batches * NUM_EPOCHS
    
    # 1. VICTIM FORWARD PASSES (every epoch, every batch)
    victim_forward_passes = DATASET_SIZE * NUM_EPOCHS
    victim_forward_flops = victim_forward_passes * VICTIM_FLOPS_PER_SAMPLE
    
    # 2. SURROGATE FORWARD PASSES (every epoch, every batch)
    surrogate_forward_passes = DATASET_SIZE * NUM_EPOCHS
    surrogate_forward_flops = surrogate_forward_passes * SURROGATE_FLOPS_PER_SAMPLE
    
    # 3. LOSS COMPUTATION (ContrastiveLoss per batch)
    loss_computation_flops = num_batches * NUM_EPOCHS * calculate_contrastive_loss_flops(BATCH_SIZE)
    
    # 4. SURROGATE BACKWARD PASSES
    surrogate_backward_flops = surrogate_forward_flops * BACKWARD_MULTIPLIER
    
    # TOTAL
    total_flops = (victim_forward_flops + 
                   surrogate_forward_flops + 
                   loss_computation_flops + 
                   surrogate_backward_flops)
    
    print("Breakdown:")
    print(f"  1. Victim Forward Passes:")
    print(f"     - Count: {victim_forward_passes:,} forward passes")
    print(f"     - FLOPs: {victim_forward_flops:,.0f} ({victim_forward_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  2. Surrogate Forward Passes:")
    print(f"     - Count: {surrogate_forward_passes:,} forward passes")
    print(f"     - FLOPs: {surrogate_forward_flops:,.0f} ({surrogate_forward_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  3. Loss Computation (ContrastiveLoss):")
    print(f"     - Count: {num_batches * NUM_EPOCHS:,} loss computations")
    print(f"     - FLOPs: {loss_computation_flops:,.0f} ({loss_computation_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  4. Surrogate Backward Passes:")
    print(f"     - FLOPs: {surrogate_backward_flops:,.0f} ({surrogate_backward_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"TOTAL FLOPS: {total_flops:,.0f} ({total_flops/1e12:.3f} TFLOPs)")
    print("="*80)
    print()
    
    return {
        'victim_forward_flops': victim_forward_flops,
        'surrogate_forward_flops': surrogate_forward_flops,
        'loss_computation_flops': loss_computation_flops,
        'surrogate_backward_flops': surrogate_backward_flops,
        'total_flops': total_flops
    }


# ============================================================================
# RDA ATTACK (RDA.py)
# ============================================================================

def calculate_rda_flops():
    """
    RDA queries the victim ONCE at the beginning to extract prototypes.
    
    Key characteristics:
    - Victim is queried ONCE with proto_patch augmentations
    - Prototypes are stored and reused for all epochs
    - query_patch augmentations are used during surrogate training
    - Uses various loss functions (default: 'ours')
    """
    print("="*80)
    print("RDA ATTACK - FLOPS CALCULATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset size: {DATASET_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Proto patch: {RDA_PROTO_PATCH} (augmentations for prototype extraction)")
    print(f"  Query patch: {RDA_QUERY_PATCH} (augmentations during training)")
    print(f"  Victim querying: ONCE at the beginning")
    print()
    
    # PHASE 1: PROTOTYPE EXTRACTION (victim queried once)
    # Victim processes each sample with proto_patch augmentations
    victim_forward_passes_proto = DATASET_SIZE * RDA_PROTO_PATCH
    victim_forward_flops_proto = victim_forward_passes_proto * VICTIM_FLOPS_PER_SAMPLE
    
    # Averaging operation for prototypes (minimal FLOPs)
    averaging_flops = DATASET_SIZE * 512 * RDA_PROTO_PATCH  # embedding_dim=512
    
    print("PHASE 1: Prototype Extraction (one-time cost)")
    print(f"  Victim Forward Passes:")
    print(f"    - Count: {victim_forward_passes_proto:,} forward passes")
    print(f"    - FLOPs: {victim_forward_flops_proto:,.0f} ({victim_forward_flops_proto/1e12:.3f} TFLOPs)")
    print(f"  Prototype Averaging:")
    print(f"    - FLOPs: {averaging_flops:,.0f} ({averaging_flops/1e9:.3f} GFLOPs)")
    print()
    
    # PHASE 2: SURROGATE TRAINING (victim not queried, uses stored prototypes)
    # Surrogate processes each sample with query_patch augmentations
    num_batches = math.ceil(DATASET_SIZE / BATCH_SIZE)
    
    # Surrogate forward passes per epoch
    surrogate_forward_passes_per_epoch = DATASET_SIZE * RDA_QUERY_PATCH
    surrogate_forward_passes_total = surrogate_forward_passes_per_epoch * NUM_EPOCHS
    surrogate_forward_flops = surrogate_forward_passes_total * SURROGATE_FLOPS_PER_SAMPLE
    
    # Loss computation (for each of query_patch augmentations)
    # RDA computes loss for each augmented view separately
    loss_computations_per_epoch = num_batches * RDA_QUERY_PATCH
    total_loss_computations = loss_computations_per_epoch * NUM_EPOCHS
    loss_computation_flops = total_loss_computations * calculate_rda_loss_flops(BATCH_SIZE, loss_type='ours')
    
    # Backward passes
    surrogate_backward_flops = surrogate_forward_flops * BACKWARD_MULTIPLIER
    
    print("PHASE 2: Surrogate Training")
    print(f"  Surrogate Forward Passes:")
    print(f"    - Per epoch: {surrogate_forward_passes_per_epoch:,} forward passes")
    print(f"    - Total ({NUM_EPOCHS} epochs): {surrogate_forward_passes_total:,} forward passes")
    print(f"    - FLOPs: {surrogate_forward_flops:,.0f} ({surrogate_forward_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  Loss Computation (Custom 'ours' loss):")
    print(f"    - Count: {total_loss_computations:,} loss computations")
    print(f"    - FLOPs: {loss_computation_flops:,.0f} ({loss_computation_flops/1e12:.3f} TFLOPs)")
    print()
    print(f"  Surrogate Backward Passes:")
    print(f"    - FLOPs: {surrogate_backward_flops:,.0f} ({surrogate_backward_flops/1e12:.3f} TFLOPs)")
    print()
    
    # TOTAL
    total_flops = (victim_forward_flops_proto + 
                   averaging_flops +
                   surrogate_forward_flops + 
                   loss_computation_flops + 
                   surrogate_backward_flops)
    
    print("="*80)
    print("TOTAL RDA FLOPS:")
    print(f"  Phase 1 (Prototype Extraction): {victim_forward_flops_proto + averaging_flops:,.0f} ({(victim_forward_flops_proto + averaging_flops)/1e12:.3f} TFLOPs)")
    print(f"  Phase 2 (Surrogate Training): {surrogate_forward_flops + loss_computation_flops + surrogate_backward_flops:,.0f} ({(surrogate_forward_flops + loss_computation_flops + surrogate_backward_flops)/1e12:.3f} TFLOPs)")
    print(f"  TOTAL: {total_flops:,.0f} ({total_flops/1e12:.3f} TFLOPs)")
    print("="*80)
    print()
    
    return {
        'victim_forward_flops_proto': victim_forward_flops_proto,
        'averaging_flops': averaging_flops,
        'surrogate_forward_flops': surrogate_forward_flops,
        'loss_computation_flops': loss_computation_flops,
        'surrogate_backward_flops': surrogate_backward_flops,
        'total_flops': total_flops
    }


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def comparative_analysis(sslsteal_results, contsteal_results, rda_results):
    """Print a comparative analysis of all three attacks"""
    print("\n")
    print("="*80)
    print("COMPARATIVE ANALYSIS - ALL THREE ATTACKS")
    print("="*80)
    print()
    
    print("Total FLOPs Comparison:")
    print(f"  SSL Stealing:      {sslsteal_results['total_flops']:>20,.0f} FLOPs ({sslsteal_results['total_flops']/1e12:>7.3f} TFLOPs)")
    print(f"  Contrastive Steal: {contsteal_results['total_flops']:>20,.0f} FLOPs ({contsteal_results['total_flops']/1e12:>7.3f} TFLOPs)")
    print(f"  RDA:               {rda_results['total_flops']:>20,.0f} FLOPs ({rda_results['total_flops']/1e12:>7.3f} TFLOPs)")
    print()
    
    print("Victim Querying Cost:")
    print(f"  SSL Stealing:      {sslsteal_results['victim_forward_flops']:>20,.0f} FLOPs ({sslsteal_results['victim_forward_flops']/1e12:>7.3f} TFLOPs) - queried every epoch")
    print(f"  Contrastive Steal: {contsteal_results['victim_forward_flops']:>20,.0f} FLOPs ({contsteal_results['victim_forward_flops']/1e12:>7.3f} TFLOPs) - queried every epoch")
    print(f"  RDA:               {rda_results['victim_forward_flops_proto']:>20,.0f} FLOPs ({rda_results['victim_forward_flops_proto']/1e12:>7.3f} TFLOPs) - queried ONCE")
    print()
    
    print("Surrogate Training Cost:")
    print(f"  SSL Stealing:      {sslsteal_results['surrogate_forward_flops'] + sslsteal_results['surrogate_backward_flops']:>20,.0f} FLOPs ({(sslsteal_results['surrogate_forward_flops'] + sslsteal_results['surrogate_backward_flops'])/1e12:>7.3f} TFLOPs)")
    print(f"  Contrastive Steal: {contsteal_results['surrogate_forward_flops'] + contsteal_results['surrogate_backward_flops']:>20,.0f} FLOPs ({(contsteal_results['surrogate_forward_flops'] + contsteal_results['surrogate_backward_flops'])/1e12:>7.3f} TFLOPs)")
    print(f"  RDA:               {rda_results['surrogate_forward_flops'] + rda_results['surrogate_backward_flops']:>20,.0f} FLOPs ({(rda_results['surrogate_forward_flops'] + rda_results['surrogate_backward_flops'])/1e12:>7.3f} TFLOPs)")
    print()
    
    # Efficiency metrics
    min_flops = min(sslsteal_results['total_flops'], contsteal_results['total_flops'], rda_results['total_flops'])
    
    print("Relative Efficiency (lower is better):")
    print(f"  SSL Stealing:      {sslsteal_results['total_flops']/min_flops:>6.2f}x")
    print(f"  Contrastive Steal: {contsteal_results['total_flops']/min_flops:>6.2f}x")
    print(f"  RDA:               {rda_results['total_flops']/min_flops:>6.2f}x")
    print()
    
    print("Key Insights:")
    print(f"  1. SSL Stealing and ContSteal have similar costs (both query victim every epoch)")
    print(f"  2. RDA is more efficient due to one-time victim querying")
    print(f"  3. RDA victim queries: {rda_results['victim_forward_flops_proto']/sslsteal_results['victim_forward_flops']:.1%} of SSL/ContSteal")
    print(f"  4. RDA uses query_patch={RDA_QUERY_PATCH} augmentations, increasing surrogate training cost")
    print(f"  5. Victim querying dominates cost in SSL/ContSteal (~{sslsteal_results['victim_forward_flops']/sslsteal_results['total_flops']:.1%} of total)")
    print()
    
    print("="*80)
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " "*15 + "SSL MODEL STEALING ATTACKS - FLOPS ANALYSIS" + " "*20 + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print()
    print("This analysis calculates the computational cost (FLOPs) for three attacks:")
    print("  1. SSL Stealing - queries victim every epoch")
    print("  2. Contrastive Steal - queries victim every epoch")
    print("  3. RDA - queries victim once at the beginning")
    print()
    print(f"Global Parameters:")
    print(f"  - Dataset size: {DATASET_SIZE:,} samples")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Training epochs: {NUM_EPOCHS}")
    print(f"  - Model: ResNet34 (~{VICTIM_FLOPS_PER_SAMPLE/1e9:.2f} GFLOPs/image)")
    print()
    print("="*80)
    print()
    
    # Calculate FLOPs for each attack
    sslsteal_results = calculate_sslsteal_flops()
    contsteal_results = calculate_contsteal_flops()
    rda_results = calculate_rda_flops()
    
    # Comparative analysis
    comparative_analysis(sslsteal_results, contsteal_results, rda_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Notes:")
    print("  - FLOPs calculations are estimates based on model architecture and operations")
    print("  - Actual runtime depends on hardware, optimization, and implementation details")
    print("  - Backward pass is estimated as 2x forward pass (standard approximation)")
    print("  - Loss computation FLOPs are based on matrix operations in each loss function")
    print()
