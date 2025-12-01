import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

# Ensure paths are correct
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import CageDataset, create_split
from src.model import ShapeTransformationNetwork
from src.loss import CageDeformationLoss
import src.utils.grid_utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description="Train Cage Deformation Model")
    parser.add_argument('--config-path', type=str, default='configs/train_config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (continue same config)')
    parser.add_argument('--resume-new', type=str, default=None,
                        help='Path to checkpoint to load weights from, but use NEW config '
                             '(new wandb run, new output dir, new hyperparameters)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    return parser.parse_args()

def save_checkpoint(epoch, model, optimizer, scheduler, cfg, output_dir, best_val_loss, is_best=False, wandb_id=None):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': OmegaConf.to_container(cfg),
        'wandb_id': wandb_id
    }
    
    # Always save latest
    latest_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(state, latest_path)
    
    # Save best
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)
        print(f"  💾 Saved best model to {best_path}")

    if epoch % 30 == 0:
        epoch30_path = os.path.join(output_dir, 'epoch30.pth')
        torch.save(state, epoch30_path)
        print(f"   💾 Saved epoch 30 model to {epoch30_path}")


    if epoch % 100 == 0:
        epoch100_path = os.path.join(output_dir, 'epoch100.pth')
        torch.save(state, epoch100_path)
        print(f"   💾 Saved epoch 100 model to {epoch100_path}")

def visualize_single_sample(input_img, template_mask, cage_deformed, boundary_pts, interior_pts):
    """
    Create a single 2x2 visualization for one sample.
    Returns a numpy array image (H, W, 3).
    """
    import cv2
    
    # Convert to numpy
    img = input_img.permute(1, 2, 0).cpu().numpy()  # (H, W, 4) [RGB + template mask]
    cage = cage_deformed.detach().cpu().numpy()
    b_pts = boundary_pts.detach().cpu().numpy()
    i_pts = interior_pts.detach().cpu().numpy()
    
    # Extract components
    gt_img = img[..., :3]  # RGB (ground truth)
    template = template_mask.squeeze().cpu().numpy()  # Template mask
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Convert [-1, 1] to pixel coords [0, 256]
    px_b = (b_pts[:, 0] + 1) / 2 * 255
    py_b = (b_pts[:, 1] + 1) / 2 * 255
    px_i = (i_pts[:, 0] + 1) / 2 * 255
    py_i = (i_pts[:, 1] + 1) / 2 * 255
    cage_px = (cage[:, 0] + 1) / 2 * 255
    cage_py = (cage[:, 1] + 1) / 2 * 255
    
    # 1. Target Shape
    axes[0, 0].imshow(gt_img)
    axes[0, 0].set_title("Target", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Template (original)
    axes[0, 1].imshow(template, cmap='gray')
    axes[0, 1].set_title("Template", fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Deformed + Cage
    axes[1, 0].set_facecolor('lightgray')
    axes[1, 0].scatter(px_i, py_i, s=0.3, c='white', alpha=0.5)
    axes[1, 0].scatter(px_b, py_b, s=1, c='blue', alpha=0.8)
    axes[1, 0].plot(np.append(cage_px, cage_px[0]), np.append(cage_py, cage_py[0]), 
                    'r-', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_xlim(0, 256)
    axes[1, 0].set_ylim(256, 0)
    axes[1, 0].set_title("Deformed + Cage", fontsize=10, fontweight='bold')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].axis('off')
    
    # 4. Overlay: Deformed on Target (KEY)
    axes[1, 1].imshow(gt_img)
    axes[1, 1].scatter(px_b, py_b, s=2, c='lime', alpha=0.9)
    axes[1, 1].plot(np.append(cage_px, cage_px[0]), np.append(cage_py, cage_py[0]), 
                    'r-', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlim(0, 256)
    axes[1, 1].set_ylim(256, 0)
    axes[1, 1].set_title("Overlay (green=deformed)", fontsize=10, fontweight='bold', color='green')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() which is the modern API
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]  # Drop alpha channel, keep RGB
    plt.close(fig)
    
    return img_array


def create_combined_visualization(samples_data, save_path=None):
    """
    Combine 4 sample visualizations (2 train + 2 val) into a single 2x2 grid image.
    
    Args:
        samples_data: List of dicts with keys: input_img, template_mask, cage_deformed, 
                      boundary_pts, interior_pts, label
        save_path: Optional path to save the combined image
    
    Returns:
        Combined image as numpy array (H, W, 3)
    """
    import cv2
    
    # Generate individual visualizations
    viz_images = []
    labels = []
    for data in samples_data:
        img = visualize_single_sample(
            data['input_img'], 
            data['template_mask'],
            data['cage_deformed'],
            data['boundary_pts'],
            data['interior_pts']
        )
        # Make a writable copy
        viz_images.append(img.copy())
        labels.append(data.get('label', ''))
    
    # Ensure we have exactly 4 images (pad with blank if needed)
    while len(viz_images) < 4:
        # Create blank image
        viz_images.append(np.ones_like(viz_images[0]) * 255)
        labels.append('')
    
    # Resize all to same size
    target_h, target_w = viz_images[0].shape[:2]
    for i in range(len(viz_images)):
        if viz_images[i].shape[:2] != (target_h, target_w):
            viz_images[i] = cv2.resize(viz_images[i], (target_w, target_h))
    
    # Add labels to each image
    for i, (img, label) in enumerate(zip(viz_images, labels)):
        if label:
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Arrange in 2x2 grid
    row1 = np.concatenate([viz_images[0], viz_images[1]], axis=1)
    row2 = np.concatenate([viz_images[2], viz_images[3]], axis=1)
    combined = np.concatenate([row1, row2], axis=0)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    return combined


def visualize_results(input_img, template_mask, cage_deformed, boundary_pts, interior_pts, save_path):
    """Legacy function for backward compatibility - saves single sample visualization."""
    img = visualize_single_sample(input_img, template_mask, cage_deformed, boundary_pts, interior_pts)
    import cv2
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def train_one_epoch(model, loader, optimizer, loss_func, rest_cage, device, epoch, cfg):
    model.train()
    
    # Metrics tracker
    loss_meter = defaultdict(float)
    num_batches = 0
    
    # === Curriculum Learning: Extended Warmup (Per README) ===
    # Phase I (0-5): Affine only
    # Phase II (5-50): Affine + Cage (no residual)
    # Phase III (50+): Full (Affine + Cage + Residual)
    warmup_affine_epochs = getattr(cfg.train, 'warmup_affine_epochs', 5)
    warmup_cage_epochs = getattr(cfg.train, 'warmup_cage_epochs', 50)
    
    freeze_cage = epoch < warmup_affine_epochs
    freeze_residual = epoch < warmup_cage_epochs
    
    # Freeze/Unfreeze parameters accordingly
    for param in model.cage_net.parameters():
        param.requires_grad = not freeze_cage
    for param in model.residual_head.parameters():
        param.requires_grad = not freeze_residual

    # Status description
    if freeze_cage:
        phase_str = "Phase I: Affine Only"
    elif freeze_residual:
        phase_str = "Phase II: Cage"
    else:
        phase_str = "Phase III: Full"
    
    status_desc = f"Epoch {epoch} [{phase_str}]"
    pbar = tqdm(loader, desc=status_desc)
    
    for batch in pbar:
        # Move data to device
        inputs = batch['input'].to(device)  # [RGB, Mask]
        pts_b = batch['pts_boundary'].to(device)
        pts_i = batch['pts_interior'].to(device)
        tgt_sdf_map = batch['target_sdf'].to(device)  # (B, 1, H, W)
        
        mvc_w_b = batch['weights_boundary'].to(device)
        mvc_w_i = batch['weights_interior'].to(device)
        
        # [SUCTION] Target boundary points for reverse Chamfer
        pts_target_b = batch['pts_target_boundary'].to(device)
        # [SLIDING] Whether boundary points are ordered
        boundary_is_ordered = batch['boundary_is_ordered'][0].item()  # Same for all in batch
        
        # Prepare Rest Cage (Expand to Batch)
        B = inputs.shape[0]
        curr_rest_cage = rest_cage.unsqueeze(0).expand(B, -1, -1).to(device)
        
        optimizer.zero_grad()
        
        # --- Forward Pass ---
        output = model(inputs, curr_rest_cage)
        affine_mat = output['affine_matrix']
        cage_offsets = output['cage_offsets']
        latent_z = output['latent_z']
        
        if freeze_cage:
            # Force offsets to zero during Phase I
            cage_offsets = cage_offsets * 0.0
            
        # 1. Deform Cage
        cage_deformed = utils.apply_affine(curr_rest_cage, affine_mat) + cage_offsets
        
        # 2. Deform Points (MVC)
        deformed_b = torch.bmm(mvc_w_b, cage_deformed)
        deformed_i = torch.bmm(mvc_w_i, cage_deformed)
        
        # 3. Apply Residual Flow (Phase III only)
        residual_b = None
        if not freeze_residual:
            residual_b = model.compute_residual(latent_z, deformed_b, output['spatial_features'])
            deformed_b = deformed_b + residual_b
        
        # Helper for SDF Sampling
        def sample_sdf(points):
            # grid_sample expects (B, 1, N, 2)
            return F.grid_sample(tgt_sdf_map, points.unsqueeze(2), align_corners=True).squeeze(2)
        
        # --- Loss Calculation ---
        loss_dict = loss_func(
            deformed_b, deformed_i, 
            cage_deformed, curr_rest_cage,
            sample_sdf,
            mvc_w_b, mvc_w_i, 
            epoch,
            residual_b=residual_b,
            target_sdf_map=tgt_sdf_map,  # For coverage loss (prevent collapse)
            pts_target_boundary=pts_target_b,  # For Suction loss
            boundary_is_ordered=boundary_is_ordered  # For Sliding loss
        )
        
        loss = loss_dict['total']
        
        # Backward
        loss.backward()
        if hasattr(cfg.train.scheduler, 'clip_grad_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.scheduler.clip_grad_norm)
        optimizer.step()
        
        # --- Logging ---
        num_batches += 1
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_meter[k] += v.item()
            else:
                loss_meter[k] += v
                
        # Update progress bar with key metrics
        pbar.set_postfix({
            'loss': loss.item(), 
            'sdf_b': loss_dict['sdf_b'].item(),
            'sdf_i': loss_dict['sdf_i'].item(),
            'res': loss_dict['res_mag'].item() if not freeze_residual else 0.0
        })
        
    # Return average losses
    return {k: v / num_batches for k, v in loss_meter.items()}

def validate(model, loader, loss_func, rest_cage, device, epoch):
    model.eval()
    loss_meter = defaultdict(float)
    num_batches = 0
    
    # Determine if residual is active
    warmup_cage_epochs = getattr(loss_func.cfg.train, 'warmup_cage_epochs', 50)
    use_residual = epoch >= warmup_cage_epochs
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            pts_b = batch['pts_boundary'].to(device)
            pts_i = batch['pts_interior'].to(device)
            tgt_sdf_map = batch['target_sdf'].to(device)
            mvc_w_b = batch['weights_boundary'].to(device)
            mvc_w_i = batch['weights_interior'].to(device)
            
            # [SUCTION/SLIDING] New fields
            pts_target_b = batch['pts_target_boundary'].to(device)
            boundary_is_ordered = batch['boundary_is_ordered'][0].item()
            
            B = inputs.shape[0]
            curr_rest_cage = rest_cage.unsqueeze(0).expand(B, -1, -1).to(device)
            
            # Forward
            output = model(inputs, curr_rest_cage)
            affine_mat = output['affine_matrix']
            cage_offsets = output['cage_offsets']
            latent_z = output['latent_z']
            
            cage_deformed = utils.apply_affine(curr_rest_cage, affine_mat) + cage_offsets
            
            deformed_b = torch.bmm(mvc_w_b, cage_deformed)
            deformed_i = torch.bmm(mvc_w_i, cage_deformed)
            
            # Apply residual if in Phase III
            residual_b = None
            if use_residual:
                residual_b = model.compute_residual(latent_z, deformed_b, output['spatial_features'])
                deformed_b = deformed_b + residual_b
            
            def sample_sdf(points):
                return F.grid_sample(tgt_sdf_map, points.unsqueeze(2), align_corners=True).squeeze(2)
                
            loss_dict = loss_func(
                deformed_b, deformed_i, 
                cage_deformed, curr_rest_cage,
                sample_sdf, 
                mvc_w_b, mvc_w_i, 
                epoch,
                residual_b=residual_b,
                target_sdf_map=tgt_sdf_map,
                pts_target_boundary=pts_target_b,
                boundary_is_ordered=boundary_is_ordered
            )
            
            num_batches += 1
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    loss_meter[k] += v.item()
                else:
                    loss_meter[k] += v
                
    return {k: v / num_batches for k, v in loss_meter.items()}

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    
    # --- Setup WandB ---
    wandb_id = None
    
    # [新增邏輯] 如果是 resume 模式，先偷看 Checkpoint 裡的 wandb_id
    if args.resume and os.path.isfile(args.resume):
        try:
            # 只加載到 CPU 讀取 ID，避免占用 GPU 記憶體
            checkpoint = torch.load(args.resume, map_location='cpu')
            wandb_id = checkpoint.get('wandb_id', None)
            if wandb_id:
                print(f"🔄 Found previous WandB run ID: {wandb_id}")
        except Exception as e:
            print(f"⚠️ Could not extract wandb_id from checkpoint: {e}")


    if not args.no_wandb:
        run = wandb.init(
                entity=cfg.experiment.entity,
                project=cfg.experiment.project_name,
                name=cfg.experiment.run_name,
                id=wandb_id,
                resume="allow" if wandb_id else None,  # Only allow resume if we have an ID
                config=OmegaConf.to_container(cfg)
            )
        
        if wandb_id:
            print(f"🔄 Resumed WandB run: {wandb.run.id}")
        else:
            print(f"📊 Started new WandB run: {wandb.run.id}")

        code_artifact = wandb.Artifact(
            name="code_snapshot",
            type="code"
        )

        code_artifact.add_dir("src")
        code_artifact.add_file("configs/train_config.yaml")
        run.log_artifact(code_artifact)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on {device}")
    
    # Dataset - with new parameters for Suction/Sliding losses
    num_target_boundary = getattr(cfg.data, 'num_target_boundary', 512)
    ordered_boundary = getattr(cfg.data, 'ordered_boundary', True)
    
    train_dataset = CageDataset(
        split_file=cfg.data.split_file,
        dataset_dir=cfg.data.dataset_dir,
        template_dir=cfg.data.template_dir,
        template_names=cfg.data.template_names,
        split='train',
        num_points=cfg.data.num_points,
        cage_num_vertices=cfg.model.cage_num_vertices,
        cage_radius=cfg.model.cage_radius,
        num_target_boundary=num_target_boundary,
        ordered_boundary=ordered_boundary
    )
    val_dataset = CageDataset(
        split_file=cfg.data.split_file,
        dataset_dir=cfg.data.dataset_dir,
        template_dir=cfg.data.template_dir,
        template_names=cfg.data.template_names,
        split='val',
        num_points=cfg.data.num_points,
        cage_num_vertices=cfg.model.cage_num_vertices,
        cage_radius=cfg.model.cage_radius,
        num_target_boundary=num_target_boundary,
        ordered_boundary=ordered_boundary
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None
    )
    
    # Model
    # Model with v9 Multi-Scale Fourier features
    use_multiscale_ff = getattr(cfg.model, 'use_multiscale_ff', True)
    multiscale_ff_scales = getattr(cfg.model, 'multiscale_ff_scales', [1.0, 5.0, 15.0])
    
    model = ShapeTransformationNetwork(
        input_channels=cfg.model.input_channels,
        cage_num_vertices=cfg.model.cage_num_vertices,
        latent_dim=getattr(cfg.model, 'latent_dim', 256),
        use_gaussian_ff=getattr(cfg.model, 'use_gaussian_ff', True),
        gaussian_ff_features=getattr(cfg.model, 'gaussian_ff_features', 96),
        gaussian_ff_scale=getattr(cfg.model, 'gaussian_ff_scale', 10.0),
        use_multiscale_ff=use_multiscale_ff,
        multiscale_ff_scales=list(multiscale_ff_scales)  # Convert from OmegaConf ListConfig
    ).to(device)
    
    print(f"📊 Gaussian Fourier: {cfg.model.use_gaussian_ff}")
    print(f"📊 Multi-Scale Fourier: {use_multiscale_ff}, Scales: {multiscale_ff_scales}")
    
    # Loss module
    loss_module = CageDeformationLoss(cfg).to(device)
    
    # ================================================================
    # Optimizer with Parameter Groups for Phase III LR Separation
    # ================================================================
    # In Phase III, we want:
    #   - Affine + Cage: lower LR (already well-trained, protect latent z)
    #   - Residual: higher LR (newly activated, needs to learn quickly)
    #
    # Parameter groups:
    #   Group 0: backbone + latent_encoder + fc_affine + cage_net
    #   Group 1: residual_head
    # ================================================================
    
    # Get learning rates from config
    base_lr = cfg.train.optimizer.lr
    residual_lr = getattr(cfg.train.optimizer, 'residual_lr', base_lr)  # Default: same as base
    
    # Separate parameters into groups
    backbone_affine_cage_params = []
    residual_params = []
    
    for name, param in model.named_parameters():
        if 'residual_head' in name:
            residual_params.append(param)
        else:
            backbone_affine_cage_params.append(param)
    
    print(f"📊 Parameter groups:")
    print(f"   Backbone+Affine+Cage: {sum(p.numel() for p in backbone_affine_cage_params):,} params, LR={base_lr}")
    print(f"   Residual Head: {sum(p.numel() for p in residual_params):,} params, LR={residual_lr}")
    
    optimizer = optim.Adam([
        {'params': backbone_affine_cage_params, 'lr': base_lr, 'name': 'backbone_affine_cage'},
        {'params': residual_params, 'lr': residual_lr, 'name': 'residual'}
    ], weight_decay=cfg.train.optimizer.weight_decay)
    
    # Scheduler - applies to all parameter groups
    # Each group will decay from its initial LR to min_lr
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cfg.train.scheduler.T_0,
        T_mult=cfg.train.scheduler.T_mult,
        eta_min=cfg.train.scheduler.min_lr
    )
    
    # Generate Standard Rest Cage
    rest_cage = utils.generate_circular_cage(
        cfg.model.cage_num_vertices, 
        radius=cfg.model.cage_radius,
        device=device
    )
    
    # Resume Checkpoint - Two modes:
    # 1. --resume: Continue training with SAME config (same wandb run, same output dir)
    # 2. --resume-new: Load weights only, use NEW config (new wandb, new output, new hyperparams)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        # Mode 1: Full resume - continue same training
        if os.path.isfile(args.resume):
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"   ✓ Restored scheduler state")
            else:
                # Fallback: manually step scheduler to correct position
                print(f"   ⚠️ No scheduler state in checkpoint, stepping manually...")
                for _ in range(checkpoint['epoch'] + 1):
                    scheduler.step()
            
            # Restore best validation loss
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                print(f"   ✓ Restored best_val_loss: {best_val_loss:.4f}")
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"   Resuming from epoch {start_epoch}")
        else:
            print(f"⚠️ Checkpoint not found: {args.resume}")
    
    elif args.resume_new:
        # Mode 2: Load weights only, fresh training state with NEW config
        if os.path.isfile(args.resume_new):
            print(f"🔄 Loading model weights from: {args.resume_new}")
            print(f"   Using NEW config: {args.config_path}")
            checkpoint = torch.load(args.resume_new, map_location=device)
            
            # Load model weights (may have mismatched keys if architecture changed)
            model_state = checkpoint['model_state_dict']
            current_state = model.state_dict()
            
            # Only load matching keys
            matched_keys = []
            for k in model_state.keys():
                if k in current_state and model_state[k].shape == current_state[k].shape:
                    current_state[k] = model_state[k]
                    matched_keys.append(k)
            
            model.load_state_dict(current_state)
            print(f"   Loaded {len(matched_keys)}/{len(model_state)} weight tensors")
            
            # DO NOT load optimizer state - fresh start with new hyperparameters
            # DO NOT restore epoch - start from 0
            # Config is already loaded from args.config_path (new config)
            start_epoch = 0
            print(f"   Starting fresh training from epoch 0 with new config")
            print(f"   New output dir: {cfg.experiment.output_dir}")
            print(f"   New wandb run: {cfg.experiment.run_name}")
        else:
            print(f"⚠️ Checkpoint not found: {args.resume_new}")

    # === Main Loop ===
    for epoch in range(start_epoch, cfg.train.epochs):
        # 1. Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_module, rest_cage, device, epoch, cfg)
        
        # 2. Validation
        val_metrics = validate(model, val_loader, loss_module, rest_cage, device, epoch)
        
        # 3. Scheduler Step
        scheduler.step()
        
        # Get current LRs for both parameter groups
        lr_backbone_affine_cage = optimizer.param_groups[0]['lr']
        lr_residual = optimizer.param_groups[1]['lr']
        
        # 4. Print Summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  LR (Backbone+Affine+Cage): {lr_backbone_affine_cage:.2e}")
        print(f"  LR (Residual): {lr_residual:.2e}")
        print(f"  Train Total Loss: {train_metrics['total']:.4f}")
        print(f"  Val Total Loss:   {val_metrics['total']:.4f}")
        
        # 5. WandB Logging - include both LRs
        if not args.no_wandb:
            log_dict = {f"Train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"Val/{k}": v for k, v in val_metrics.items()})
            # Log both learning rates
            log_dict["Params/LR_Backbone_Affine_Cage"] = lr_backbone_affine_cage
            log_dict["Params/LR_Residual"] = lr_residual
            log_dict["Params/Epoch"] = epoch
            wandb.log(log_dict, step=epoch)
            
        # 6. Save Checkpoint
        is_best = val_metrics['total'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total']
            print(f"  ⭐ New best val loss: {best_val_loss:.4f}")
            
        current_wandb_id = wandb.run.id if not args.no_wandb and wandb.run else None
        
        # Save checkpoint with scheduler state and best_val_loss
        save_checkpoint(epoch, model, optimizer, scheduler, cfg, cfg.experiment.output_dir, 
                       best_val_loss, is_best, wandb_id=current_wandb_id)
        
        viz_interval = cfg.train.get('viz_interval', 5)
        if epoch % viz_interval == 0:
            print("  📸 Generating combined visualization (2 train + 2 val)...")
            model.eval()
            viz_dir = os.path.join(cfg.experiment.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # [新增 1] 判斷現在是否該加上 Residual (Phase III)
            warmup_cage_epochs = getattr(cfg.train, 'warmup_cage_epochs', 50)
            use_residual = epoch >= warmup_cage_epochs
            
            with torch.no_grad():
                samples_data = []
                
                # --- Get 2 training samples ---
                train_iter = iter(train_loader)
                for i in range(2):
                    try:
                        batch = next(train_iter)
                        inputs = batch['input'].to(device)[:1]
                        mvc_w_b = batch['weights_boundary'].to(device)[:1]
                        mvc_w_i = batch['weights_interior'].to(device)[:1]
                        curr_rest_cage = rest_cage.unsqueeze(0).to(device)
                        
                        # Forward pass
                        output = model(inputs, curr_rest_cage)
                        aff = output['affine_matrix']
                        off = output['cage_offsets']
                        latent_z = output['latent_z']
                        
                        cage_def = utils.apply_affine(curr_rest_cage, aff) + off
                        def_b = torch.bmm(mvc_w_b, cage_def)
                        def_i = torch.bmm(mvc_w_i, cage_def)
                        
                        # [新增 2] 加上 Residual (如果有啟用的話)
                        # 這是您原本漏掉的關鍵步驟！
                        if use_residual:
                            res_b = model.compute_residual(latent_z, def_b, output['spatial_features'])
                            def_b = def_b + res_b
                        
                        samples_data.append({
                            'input_img': inputs[0],
                            'template_mask': inputs[0, 3:4],
                            'cage_deformed': cage_def[0],
                            'boundary_pts': def_b[0],
                            'interior_pts': def_i[0],
                            'label': f'Train #{i+1}'
                        })
                    except StopIteration:
                        break
                
                # --- Get 2 validation samples ---
                val_iter = iter(val_loader)
                for i in range(2):
                    try:
                        batch = next(val_iter)
                        inputs = batch['input'].to(device)[:1]
                        mvc_w_b = batch['weights_boundary'].to(device)[:1]
                        mvc_w_i = batch['weights_interior'].to(device)[:1]
                        curr_rest_cage = rest_cage.unsqueeze(0).to(device)
                        
                        # Forward pass
                        output = model(inputs, curr_rest_cage)
                        aff = output['affine_matrix']
                        off = output['cage_offsets']
                        latent_z = output['latent_z']
                        
                        cage_def = utils.apply_affine(curr_rest_cage, aff) + off
                        def_b = torch.bmm(mvc_w_b, cage_def)
                        def_i = torch.bmm(mvc_w_i, cage_def)
                        
                        # [新增 3] 加上 Residual (如果有啟用的話)
                        if use_residual:
                            res_b = model.compute_residual(latent_z, def_b, output['spatial_features'])
                            def_b = def_b + res_b
                        
                        samples_data.append({
                            'input_img': inputs[0],
                            'template_mask': inputs[0, 3:4],
                            'cage_deformed': cage_def[0],
                            'boundary_pts': def_b[0],
                            'interior_pts': def_i[0],
                            'label': f'Val #{i+1}'
                        })
                    except StopIteration:
                        break
                
                # Create combined 2x2 visualization
                combined_path = os.path.join(viz_dir, f"epoch{epoch}_combined.png")
                combined_img = create_combined_visualization(samples_data, combined_path)
                print(f"    Saved: {combined_path}")
                
                if not args.no_wandb:
                    wandb.log({
                        "Visualization/Combined_Samples": wandb.Image(
                            combined_img, 
                            caption=f"Epoch {epoch}: Train(top) + Val(bottom)"
                        )
                    }, step=epoch)

if __name__ == "__main__":
    main()