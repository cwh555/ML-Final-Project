import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Delaunay
import numpy as np
from tqdm import tqdm
import argparse
import random
import time
import math
import glob
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import CageDataset, create_split
from src.model import ShapeTransformationNetwork
import src.utils.grid_utils as utils

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def set_requires_grad(module, status):
    """設置模塊的梯度開關 (凍結/解凍)"""
    for param in module.parameters():
        param.requires_grad = status

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs/train_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()

def init_weights_zero(model):
    print("🔧 Initializing output heads to ZERO/IDENTITY...")
    if hasattr(model, 'fc_global'):
        torch.nn.init.zeros_(model.fc_global[-1].weight)
        model.fc_global[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    if hasattr(model, 'fc_local'):
        torch.nn.init.zeros_(model.fc_local[-1].weight)
        torch.nn.init.zeros_(model.fc_local[-1].bias)

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_metric, cfg, is_best, output_dir):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_metric': best_val_metric,
        'wandb_run_id': wandb.run.id if wandb.run else None,
        'config': OmegaConf.to_container(cfg)
    }
    torch.save(state, os.path.join(output_dir, "checkpoint_latest.pth"))
    if wandb.run: wandb.save(os.path.join(output_dir, "checkpoint_latest.pth"), base_path=output_dir)
    if is_best:
        torch.save(state, os.path.join(output_dir, "best_model.pth"))
        if wandb.run: wandb.save(os.path.join(output_dir, "best_model.pth"), base_path=output_dir)
    if epoch % cfg.train.save_interval == 0:
        torch.save(state, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth"))

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    if not os.path.exists(checkpoint_path): 
        return 0, float('inf'), None
    
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    state = torch.load(checkpoint_path)
    
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    
    if scheduler and state.get('scheduler_state_dict'):
        scheduler.load_state_dict(state['scheduler_state_dict'])
        print(f"✅ Scheduler state restored")
    
    start_epoch = state['epoch'] + 1
    best_val_metric = state.get('best_val_metric', float('inf'))
    wandb_run_id = state.get('wandb_run_id', None)
    
    print(f"✅ Resumed from epoch {state['epoch']}, best_val={best_val_metric:.4f}")
    if wandb_run_id:
        print(f"✅ WandB run_id: {wandb_run_id}")
    
    return start_epoch, best_val_metric, wandb_run_id

def get_curriculum_params(cfg, epoch, total_epochs):
    """
    完全基於配置文件的 Curriculum Learning
    三階段策略：Phase 1 (全域對齊) → Phase 2 (局部變形) → Phase 3 (精細化)
    """
    ramp_end_ratio = cfg.train.curriculum.ramp_end_ratio
    raw_progress = epoch / total_epochs
    progress = min(1.0, raw_progress / ramp_end_ratio)
    
    # === Phase 1: Global Alignment (0-20%) ===
    if progress < 0.2:
        p1 = cfg.train.curriculum.phase1
        w_align = cfg.train.alpha_align * p1.w_align_scale
        w_boundary_sdf = p1.w_boundary_sdf
        w_repul = p1.w_repulsion
        w_phys = cfg.train.alpha_physics * p1.w_physics_scale
        w_attn_guide = p1.w_attn_guide
        w_flow = p1.w_flow
        mu = p1.mu
        lam = p1.lam
    
    # === Phase 2: Local Deformation (20-50%) ===
    elif progress < 0.5:
        stage2_progress = (progress - 0.2) / 0.3
        p2 = cfg.train.curriculum.phase2
        
        w_align = cfg.train.alpha_align * p2.w_align_scale
        w_boundary_sdf = cfg.train.alpha_boundary_sdf * p2.w_boundary_sdf_scale * stage2_progress
        w_repul = p2.w_repulsion
        
        # 物理權重線性插值
        w_phys_scale = p2.w_physics_scale_start + (p2.w_physics_scale_end - p2.w_physics_scale_start) * stage2_progress
        w_phys = cfg.train.alpha_physics * w_phys_scale
        
        # Attention 引導逐漸增強
        w_attn_guide = p2.w_attn_guide_max * stage2_progress
        
        # Flow 逐漸減弱
        w_flow = p2.w_flow_start + (p2.w_flow_end - p2.w_flow_start) * stage2_progress
        
        # 彈性參數指數衰減（材料逐漸軟化）
        mu = p2.mu_start * ((p2.mu_end / p2.mu_start) ** stage2_progress)
        lam = p2.lam_start + (p2.lam_end - p2.lam_start) * stage2_progress
    
    # === Phase 3: Refinement (50-100%) ===
    else:
        stage3_progress = (progress - 0.5) / 0.5
        p3 = cfg.train.curriculum.phase3
        
        w_align = cfg.train.alpha_align * p3.w_align_scale
        w_boundary_sdf = cfg.train.alpha_boundary_sdf * p3.w_boundary_sdf_scale
        w_repul = cfg.train.alpha_repulsion * p3.w_repulsion_scale * stage3_progress
        
        w_phys_scale = p3.w_physics_scale_start + (p3.w_physics_scale_end - p3.w_physics_scale_start) * stage3_progress
        w_phys = cfg.train.alpha_physics * w_phys_scale
        
        # Attention 引導逐漸減弱
        w_attn_guide = p3.w_attn_guide_start + (p3.w_attn_guide_end - p3.w_attn_guide_start) * stage3_progress
        
        w_flow = p3.w_flow_start + (p3.w_flow_end - p3.w_flow_start) * stage3_progress
        
        # 材料進一步軟化以支持極端變形
        mu = p3.mu_start * ((p3.mu_end / p3.mu_start) ** stage3_progress)
        lam = p3.lam_start + (p3.lam_end - p3.lam_start) * stage3_progress
    
    # === Topology Constraint Weight (拓撲保持約束) ===
    if progress < 0.2:
        w_interior = 0.0  # Phase 1: 不啟動，讓 Affine 先對齊
    elif progress < 0.5:
        stage2_progress = (progress - 0.2) / 0.3
        w_interior = cfg.train.get('alpha_topology', 150.0) * stage2_progress  # Phase 2: 逐漸增強
    else:
        w_interior = cfg.train.get('alpha_topology', 150.0)  # Phase 3: 全力執行
    
    return w_align, w_boundary_sdf, w_repul, w_phys, mu, lam, w_attn_guide, w_flow, w_interior

def create_viz_mesh_data(mask_tensor, num_samples=2500):
    mask = mask_tensor.squeeze().cpu().numpy()
    if mask.max() <= 1.0: mask = (mask * 255).astype(np.uint8)
    else: mask = mask.astype(np.uint8)
    h, w = mask.shape
    ys, xs = np.where(mask > 127)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    if points.shape[0] > num_samples:
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        points = points[idx]
    if points.shape[0] < 3: return None
    tri = Delaunay(points)
    faces = tri.simplices
    centroids = points[faces].mean(axis=1).astype(int)
    centroids[:, 0] = np.clip(centroids[:, 0], 0, w-1); centroids[:, 1] = np.clip(centroids[:, 1], 0, h-1)
    valid = mask[centroids[:, 1], centroids[:, 0]] > 127
    faces = faces[valid]
    points_norm = (points / np.array([w-1, h-1])) * 2 - 1
    v0 = points[faces[:, 0]]; v1 = points[faces[:, 1]]; v2 = points[faces[:, 2]]
    vec1 = v1 - v0; vec2 = v2 - v0
    ref_areas = 0.5 * np.abs(vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])
    return {'vertices': torch.tensor(points_norm, dtype=torch.float32), 'faces': faces, 'ref_areas': ref_areas}

def visualize_results(model, train_data, val_data, epoch, cfg, device):
    if not wandb.run: return
    cdict = {'red': ((0.0, 0.5, 0.5), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)), 'green': ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.8, 0.8)), 'blue': ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)), 'alpha': ((0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.3, 0.3))}
    density_cmap = LinearSegmentedColormap('Density', cdict)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=100)
    fig.suptitle(f"Epoch {epoch} - Spatial Attention Corrected", fontsize=14)
    datasets = [('Train', train_data), ('Val', val_data)]
    grid_rest = utils.generate_regular_grid(cfg.model.cage_resolution).to(device)
    VIZ_SCALE = 256.0
    
    for row_idx, (label, data) in enumerate(datasets):
        if data is None: continue
        inputs, tgt_pts, _, sdfs, _, spatial_weights = data
        col = 0
        img_input = inputs[col].unsqueeze(0)
        target_points_norm = tgt_pts[col].cpu().numpy()
        target_points_viz = (target_points_norm + 1) * (VIZ_SCALE / 2.0)
        template_mask = inputs[col, 3:4]
        mesh_data = create_viz_mesh_data(template_mask)
        if mesh_data is None: continue
        mesh_verts = mesh_data['vertices'].to(device)
        mesh_faces = mesh_data['faces']
        ref_areas = mesh_data['ref_areas']
        idx, w = utils.precompute_bilinear_weights(mesh_verts, cfg.model.cage_resolution)
        
        with torch.no_grad():
            affine, offsets, attn_weights, _ = model(img_input)
            B = 1
            grid_rest_batch = grid_rest.unsqueeze(0).expand(B, -1, -1)
            grid_global = utils.apply_affine_transform(grid_rest_batch, affine)
            grid_deformed = grid_global + offsets
            deformed_verts_norm = utils.deform_points_with_grid(
                grid_deformed, idx.unsqueeze(0), w.unsqueeze(0)
            ).squeeze(0).cpu().numpy()
        
        viz_verts = (deformed_verts_norm + 1) * (VIZ_SCALE / 2.0)
        new_areas = 0.5 * np.abs((viz_verts[mesh_faces[:, 1]] - viz_verts[mesh_faces[:, 0]])[:, 0] * (viz_verts[mesh_faces[:, 2]] - viz_verts[mesh_faces[:, 0]])[:, 1] - (viz_verts[mesh_faces[:, 1]] - viz_verts[mesh_faces[:, 0]])[:, 1] * (viz_verts[mesh_faces[:, 2]] - viz_verts[mesh_faces[:, 0]])[:, 0])
        area_ratios = new_areas / (ref_areas + 1e-6)
        viz_ratios = np.log2(area_ratios + 1e-6)
        viz_colors = np.clip(viz_ratios / 4.0 + 0.5, 0.0, 1.0)
        
        axes[row_idx, 0].imshow(np.clip(inputs[col, :3].cpu().permute(1, 2, 0).numpy(), 0, 1))
        axes[row_idx, 0].plot(target_points_viz[:, 0], target_points_viz[:, 1], '.', c='lime', ms=2, alpha=0.5)
        axes[row_idx, 0].axis('off'); axes[row_idx, 0].set_title("Input + GT")
        
        axes[row_idx, 1].tripcolor(viz_verts[:, 0], viz_verts[:, 1], mesh_faces, facecolors=viz_colors, cmap=density_cmap, edgecolors='none', shading='flat', vmin=0, vmax=1)
        axes[row_idx, 1].set_xlim(0, VIZ_SCALE); axes[row_idx, 1].set_ylim(VIZ_SCALE, 0); axes[row_idx, 1].axis('off'); axes[row_idx, 1].set_title("Density")
        
        axes[row_idx, 2].triplot(viz_verts[:, 0], viz_verts[:, 1], mesh_faces, color='red', linewidth=0.5, alpha=0.6)
        axes[row_idx, 2].plot(target_points_viz[:, 0], target_points_viz[:, 1], '.', c='lime', ms=2, alpha=0.7)
        axes[row_idx, 2].set_xlim(0, VIZ_SCALE); axes[row_idx, 2].set_ylim(VIZ_SCALE, 0); axes[row_idx, 2].axis('off'); axes[row_idx, 2].set_title("Alignment")
        
        attn_viz = attn_weights.squeeze(0).squeeze(1).cpu().numpy()
        attn_2d = attn_viz.reshape(cfg.model.cage_resolution, cfg.model.cage_resolution)
        im = axes[row_idx, 3].imshow(attn_2d, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
        axes[row_idx, 3].axis('off'); axes[row_idx, 3].set_title("Attention")
        plt.colorbar(im, ax=axes[row_idx, 3], fraction=0.046)

    plt.tight_layout()
    wandb.log({"Viz/Results": wandb.Image(fig)}, step=epoch)
    plt.close(fig)

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    set_seed()
    device = torch.device(cfg.train.device)
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    if not os.path.exists(cfg.data.split_file):
        create_split(cfg.data.dataset_dir, cfg.data.split_file, cfg.data.val_split_ratio)
    
    train_ds = CageDataset(cfg.data.split_file, cfg.data.dataset_dir, cfg.data.template_dir,
                           cfg.data.template_names, split='train', num_points=cfg.data.num_points)
    val_ds = CageDataset(cfg.data.split_file, cfg.data.dataset_dir, cfg.data.template_dir,
                         cfg.data.template_names, split='val', num_points=cfg.data.num_points)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)
    num_boundary = train_ds.num_boundary
    
    model = ShapeTransformationNetwork(
        cfg.model.input_channels, 
        cfg.model.cage_resolution,
        coarse_grid_res=cfg.model.get('coarse_grid_res', 8)
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.optimizer.lr,
                            weight_decay=cfg.train.optimizer.weight_decay)
    
    # Create scheduler
    scheduler = None
    if cfg.train.scheduler.type == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train.scheduler.T_0,
            T_mult=cfg.train.scheduler.T_mult,
            eta_min=cfg.train.scheduler.min_lr
        )
    
    # Resume or initialize
    wandb_run_id = None
    if args.resume is None:
        init_weights_zero(model)
        start_epoch, best_val_metric = 0, float('inf')
    else:
        start_epoch, best_val_metric, wandb_run_id = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    grid_rest = utils.generate_regular_grid(cfg.model.cage_resolution).to(device)
    template_data = []
    for t in train_ds.templates:
        pts = t['points'].to(device)
        idx, w = utils.precompute_bilinear_weights(pts, cfg.model.cage_resolution)
        template_data.append({'idx': idx, 'w': w})
    
    # WandB initialization with resume support
    if cfg.wandb.enable:
        if wandb_run_id:
            # Resume existing run
            wandb.init(
                entity=cfg.experiment.entity,
                project=cfg.experiment.project_name,
                id=wandb_run_id,
                resume="must",
                config=OmegaConf.to_container(cfg)
            )
            print(f"🔄 Resumed WandB run: {wandb_run_id}")
        else:
            # Start new run
            wandb.init(
                entity=cfg.experiment.entity,
                project=cfg.experiment.project_name,
                name=cfg.experiment.run_name,
                config=OmegaConf.to_container(cfg)
            )
            print(f"🚀 Started new WandB run: {wandb.run.id}")
        wandb.save("src/*.py", policy="now"); wandb.save("src/utils/*.py", policy="now"); wandb.save("configs/*.yaml", policy="now")

    print(f"\n✅ Start Training: {cfg.experiment.run_name} (Attention Fix)")
    
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        losses = {'total': 0, 'align': 0, 'sdf': 0, 'repul': 0, 'phys': 0, 'attn': 0, 'interior': 0}
        
        w_align, w_boundary_sdf, w_repul, w_phys, mu, lam, w_attn_guide, w_flow, w_interior = \
            get_curriculum_params(cfg, epoch, cfg.train.epochs)
        
        # [核心策略] Phase 1: 凍結 Fine Grid，強迫優化器只能動 Coarse Grid
        # 這樣 Coarse 網格才會學到 U 型大彎曲
        phase1_end = int(cfg.train.epochs * 0.2)
        if epoch < phase1_end:
            set_requires_grad(model.fc_local, False)  # 凍結 Fine
            set_requires_grad(model.fc_coarse, True)  # 解凍 Coarse
            if epoch == 0:
                print("🧊 Phase 1: Fine Grid 凍結，只訓練 Coarse Grid (大尺度彎曲)")
        else:
            set_requires_grad(model.fc_local, True)   # 解凍 Fine
            set_requires_grad(model.fc_coarse, True)  # 解凍 Coarse
            if epoch == phase1_end:
                print("🔥 Phase 2/3: Fine Grid 解凍，同時訓練 Coarse + Fine (細節修正)")
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        viz_train_data = None
        flip_ratios = []
        
        for batch in pbar:
            inputs = batch['input'].to(device)
            target_pts = batch['target_points'].to(device)
            target_sdfs = batch['target_sdf'].to(device)
            t_indices = batch['template_idx']
            
            optimizer.zero_grad()
            
            # Forward (新增 coarse_deformation 返回值)
            affine, offsets, spatial_weights, coarse_deformation = model(inputs)
            
            B = inputs.shape[0]
            grid_rest_batch = grid_rest.unsqueeze(0).expand(B, -1, -1)
            # [關鍵] 應用 Affine 得到 Global Grid
            grid_global = utils.apply_affine_transform(grid_rest_batch, affine)
            grid_deformed = grid_global + offsets
            
            deformed_pts_list = []
            for b in range(inputs.shape[0]):
                t_id = t_indices[b].item()
                d_pts = utils.deform_points_with_grid(
                    grid_deformed[b:b+1],
                    template_data[t_id]['idx'].unsqueeze(0),
                    template_data[t_id]['w'].unsqueeze(0)
                )
                deformed_pts_list.append(d_pts.squeeze(0))
            deformed_stack = torch.stack(deformed_pts_list)
            
            deformed_boundary = deformed_stack[:, :num_boundary]
            deformed_interior = deformed_stack[:, num_boundary:]
            
            l_align = utils.loss_chamfer(deformed_boundary, target_pts)
            l_sdf_boundary = utils.loss_sdf_boundary_focused(deformed_boundary, target_sdfs)
            
            if w_repul > 0:
                l_repul = utils.loss_interior_repulsion_conservative(deformed_interior, target_sdfs, threshold=0.02)
            else:
                l_repul = torch.tensor(0.0, device=device)
            
            l_phys, flip_ratio = utils.loss_physics_with_flow_consistency(
                grid_deformed, cfg.model.cage_resolution,
                mu=mu, lam=lam, barrier_strength=1e4, flow_weight=w_flow
            )
            
            # 這能確保 8x8 網格保持平滑，不會出現奇怪的突起
            l_coarse_phys, _ = utils.loss_physics_with_flow_consistency(
                coarse_deformation, cfg.model.cage_resolution, # 這裡傳入插值後的 coarse grid
                mu=mu * 2.0,  # 給予更高的剛性
                lam=lam, 
                flow_weight=w_flow
            )
            
            if w_attn_guide > 0:
                # [關鍵] 傳入 grid_global 進行引導
                l_attn_guide = utils.loss_spatial_attention_guidance(
                    spatial_weights, target_pts, grid_global, cfg.model.cage_resolution
                )
            else:
                l_attn_guide = torch.tensor(0.0, device=device)
            
            # [修正] Topology Preservation: 空洞对齐空洞
            if w_interior > 0:
                template_masks = inputs[:, 3:4, :, :]  # 提取 Template Mask (B, 1, H, W)
                l_interior, penetration_ratio = utils.loss_topology_preservation(
                    template_masks, grid_deformed, target_sdfs, cfg.model.cage_resolution, num_samples=1024
                )
            else:
                l_interior = torch.tensor(0.0, device=device)
                penetration_ratio = torch.tensor(0.0, device=device)
            
            loss = (
                w_align * l_align +
                w_boundary_sdf * l_sdf_boundary +
                w_repul * l_repul +
                w_phys * l_phys +
                0.5 * w_phys * l_coarse_phys +
                w_attn_guide * l_attn_guide +
                w_interior * l_interior
            )
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                          cfg.train.scheduler.get('clip_grad_norm', 0.5))
            loss.backward()
            optimizer.step()
            
            losses['total'] += loss.item()
            losses['align'] += l_align.item()
            losses['sdf'] += l_sdf_boundary.item()
            losses['repul'] += l_repul.item()
            losses['phys'] += l_phys.item()
            losses['attn'] += l_attn_guide.item()
            losses['interior'] += l_interior.item()
            flip_ratios.append(flip_ratio.item())
            
            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'Int': f"{l_interior.item():.3f}",
                'Pen': f"{penetration_ratio.item():.2%}"
            })
            
            if viz_train_data is None:
                viz_train_data = (inputs, target_pts, deformed_stack, target_sdfs,
                                 grid_deformed, spatial_weights)

        avg_flip_ratio = sum(flip_ratios) / len(flip_ratios)
        if cfg.wandb.enable:
            wandb.log({
                "Train/Total": losses['total'] / len(train_loader),
                "Train/Align": losses['align'] / len(train_loader),
                "Train/SDF": losses['sdf'] / len(train_loader),
                "Train/Repul": losses['repul'] / len(train_loader),
                "Train/Physics": losses['phys'] / len(train_loader),
                "Train/Attention": losses['attn'] / len(train_loader),
                "Train/Interior": losses['interior'] / len(train_loader),
                "Monitor/Flip_Ratio": avg_flip_ratio,
                "Params/Mu": mu,
                "Weights/AttentionGuide": w_attn_guide,
                "Weights/Interior": w_interior
            }, step=epoch)
        
        model.eval()
        val_align_total = 0
        viz_val_data = None
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                tgt_pts = batch['target_points'].to(device)
                target_sdfs = batch['target_sdf'].to(device)
                t_indices = batch['template_idx']
                
                affine, offsets, spatial_weights, coarse_deformation = model(inputs)
                B = inputs.shape[0]
                grid_rest_batch = grid_rest.unsqueeze(0).expand(B, -1, -1)
                grid_global = utils.apply_affine_transform(grid_rest_batch, affine)
                grid_deformed = grid_global + offsets
                
                deformed_pts_list = []
                for b in range(inputs.shape[0]):
                    t_id = t_indices[b].item()
                    d_pts = utils.deform_points_with_grid(
                        grid_deformed[b:b+1],
                        template_data[t_id]['idx'].unsqueeze(0),
                        template_data[t_id]['w'].unsqueeze(0)
                    )
                    deformed_pts_list.append(d_pts.squeeze(0))
                deformed_stack = torch.stack(deformed_pts_list)
                deformed_boundary = deformed_stack[:, :num_boundary]
                l_align = utils.loss_chamfer(deformed_boundary, tgt_pts)
                val_align_total += l_align.item()
                
                if viz_val_data is None:
                    viz_val_data = (inputs, tgt_pts, deformed_stack, target_sdfs,
                                   grid_deformed, spatial_weights)

        avg_val_align = val_align_total / len(val_loader)
        if cfg.wandb.enable:
            wandb.log({"Val/Align_Chamfer": avg_val_align}, step=epoch)
        
        if epoch % cfg.train.viz_interval == 0:
            visualize_results(model, viz_train_data, viz_val_data, epoch, cfg, device)
        
        print(f"      Train Total: {losses['total']/len(train_loader):.4f} | Val Align: {avg_val_align:.4f}")
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if cfg.wandb.enable:
                wandb.log({"Params/LearningRate": current_lr}, step=epoch)
        
        is_best = avg_val_align < best_val_metric
        if is_best: best_val_metric = avg_val_align
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_metric, cfg, is_best, cfg.experiment.output_dir)

if __name__ == "__main__":
    main()

