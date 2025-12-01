"""
Model Visualization for Cage-based Shape Deformation - Fixed
=============================================================
Uses model's COMPLETE prediction pipeline: Affine + Cage + Residual
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from omegaconf import OmegaConf
import argparse
import os
import sys
from tqdm import tqdm
import time
import cv2
import pickle
import torchvision
from torchvision import transforms
import json
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.dataset import CageDataset
from src.model import ShapeTransformationNetwork
import src.utils.grid_utils as utils

VIZ_SCALE = 256.0 
VIZ_DPI = 150 


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize cage deformation model')
    parser.add_argument('--config-path', type=str, default='configs/train_config.yaml')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='images_eval')
    parser.add_argument('--specific-templates', nargs='+', default=None)
    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--omniglot', action='store_true', help='Use Omniglot dataset')
    parser.add_argument('--real', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show-cage', action='store_true')
    parser.add_argument('--no-residual', action='store_true',
                        help='Skip residual flow (only show cage deformation)')
    parser.add_argument('--evaluate-metrics', action='store_true',
                        help='Run comprehensive metric evaluation on all validation data')
    parser.add_argument('--num-eval-samples', type=int, default=None,
                        help='Number of samples for metric evaluation (default: all)')
    return parser.parse_args()


def load_model(cfg, checkpoint_path, device):
    """Load model from checkpoint."""
    model = ShapeTransformationNetwork(
        input_channels=cfg.model.input_channels,
        cage_num_vertices=cfg.model.get('cage_num_vertices', 96),
        latent_dim=cfg.model.get('latent_dim', 256),
        use_gaussian_ff=cfg.model.get('use_gaussian_ff', True),
        gaussian_ff_features=cfg.model.get('gaussian_ff_features', 64),
        gaussian_ff_scale=cfg.model.get('gaussian_ff_scale', 10.0),
        use_multiscale_ff=cfg.model.get('use_multiscale_ff', True),
        multiscale_ff_scales=cfg.model.get('multiscale_ff_scales', [1.0, 5.0, 15.0])
    ).to(device)
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in state: 
        model.load_state_dict(state['model_state_dict'])
        print(f"✅ Loaded from epoch {state.get('epoch', 'N/A')}")
    else: 
        model.load_state_dict(state)
    
    model.eval()
    return model


def create_template_mesh(template_dir, template_name, device, rest_cage, num_samples=3000):
    """Load template and compute MVC weights for visualization."""
    path = os.path.join(template_dir, f"{template_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {path}")
        
    data = np.load(path, allow_pickle=True)
    mask = data['mask'] 
    
    if mask.max() <= 1.0: 
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_float = mask.astype(np.float32)
    else: 
        mask_uint8 = mask.astype(np.uint8)
        mask_float = mask.astype(np.float32) / 255.0

    h, w = mask_uint8.shape
    mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)
    
    # Sample points inside template for mesh visualization
    ys, xs = np.where(mask_uint8 > 127)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    
    if points.shape[0] > num_samples:
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        points = points[idx]
    
    if points.shape[0] < 3:
        return None
        
    # Create Delaunay triangulation for visualization
    tri = Delaunay(points)
    centroids = points[tri.simplices].mean(axis=1).astype(int)
    centroids[:, 0] = np.clip(centroids[:, 0], 0, w-1)
    centroids[:, 1] = np.clip(centroids[:, 1], 0, h-1)
    
    valid = mask_uint8[centroids[:, 1], centroids[:, 0]] > 127
    faces = tri.simplices[valid]
    
    # Reference areas for quality check
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    vec1, vec2 = v1 - v0, v2 - v0
    ref_areas = 0.5 * np.abs(vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])
    
    # Normalize to [-1, 1]
    points_norm = (points / np.array([w-1, h-1])) * 2 - 1
    vertices_tensor = torch.tensor(points_norm, dtype=torch.float32).to(device)
    
    # Pre-compute MVC weights for cage deformation
    cage_batch = rest_cage.unsqueeze(0).to(device)
    pts_batch = vertices_tensor.unsqueeze(0)
    mvc_weights = utils.compute_mvc_weights(pts_batch, cage_batch)
    
    return {
        'vertices_norm': vertices_tensor,
        'mvc_weights': mvc_weights.squeeze(0),
        'faces': faces,
        'ref_areas': ref_areas,
        'mask_tensor': mask_tensor,
        'shape': (h, w)
    }


def check_mesh_quality(deformed_verts, faces, ref_areas):
    """Check mesh quality after deformation."""
    if len(faces) == 0:
        return {'flip_ratio': 0, 'collapsed': False, 'area_ratios': np.array([1.0]),
                'mean_ratio': 1.0}
    
    v0 = deformed_verts[faces[:, 0]]
    v1 = deformed_verts[faces[:, 1]]
    v2 = deformed_verts[faces[:, 2]]
    vec1, vec2 = v1 - v0, v2 - v0
    
    signed_areas = 0.5 * (vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])
    new_areas = np.abs(signed_areas)
    
    flip_ratio = float(np.sum(signed_areas < 0)) / len(faces)
    area_ratios = new_areas / (ref_areas + 1e-8)
    
    return {
        'flip_ratio': flip_ratio,
        'area_ratios': area_ratios,
        'mean_ratio': float(np.mean(area_ratios)),
        'collapsed': flip_ratio > 0.05
    }


# ============================================================================
# 2D EVALUATION METRICS (CVPR/ECCV/ICCV & MICCAI Standard)
# ============================================================================
# 
# 【核心指標：證明撕裂與拓撲正確 (Topology & Tearing)】
# 1. Hausdorff Distance (HD/HD95) - MICCAI standard, 輪廓一致性的黃金標準
# 2. Boundary F-Score (BF Score) - Gated-SCNN (CVPR 2019), 邊緣細節專用
#
# 【基礎指標：證明整體形狀吻合 (Global Shape)】
# 3. Mask IoU - Mask R-CNN (CVPR 2017), DeepLab, 實體面積重疊度
# 4. Contour Chamfer Distance - RPM-Net, 2D 輪廓點平均最近距離
# ============================================================================

def compute_all_2d_contour_metrics(contour_pred, contour_target, bf_threshold=2.0, target_tree=None):
    """
    Compute all 2D contour-based metrics in ONE pass (OPTIMIZED).
    Builds KDTree only once instead of rebuilding for each metric.
    
    【專為 2D Deformation + Tearing 設計】
    
    Args:
        contour_pred: (N, 2) predicted contour points (pixel coords)
        contour_target: (M, 2) target contour points (pixel coords)
        bf_threshold: BF Score 的像素閾值 (default: 2 pixels)
        target_tree: Pre-built cKDTree for target (optional, for caching)
    
    Returns: dict with HD, HD95, BF Score, Contour CD, etc.
    """
    results = {}
    
    if len(contour_pred) == 0 or len(contour_target) == 0:
        results['hausdorff'] = float('inf')
        results['hausdorff_95'] = float('inf')
        results['bf_score'] = 0.0
        results['bf_precision'] = 0.0
        results['bf_recall'] = 0.0
        results['contour_chamfer'] = float('inf')
        return results
    
    # Build KDTrees - use pre-built target_tree if provided
    tree_target = target_tree if target_tree is not None else cKDTree(contour_target)
    tree_pred = cKDTree(contour_pred)
    
    # Query distances ONCE (reused for all metrics)
    dist_p2t, _ = tree_target.query(contour_pred, k=1)
    dist_t2p, _ = tree_pred.query(contour_target, k=1)
    
    # 1. Hausdorff Distance (HD) - 最不合群的點的距離
    results['hausdorff'] = max(np.max(dist_p2t), np.max(dist_t2p))
    
    # 2. HD95 - 第95百分位，排除極端噪聲，更穩定
    results['hausdorff_95'] = max(np.percentile(dist_p2t, 95), np.percentile(dist_t2p, 95))
    
    # 3. Boundary F-Score (BF Score) - 邊緣精準度
    precision = np.mean(dist_p2t <= bf_threshold)
    recall = np.mean(dist_t2p <= bf_threshold)
    bf_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    results['bf_score'] = bf_score
    results['bf_precision'] = precision
    results['bf_recall'] = recall
    
    # 4. Contour Chamfer Distance (L1)
    results['contour_chamfer'] = (np.mean(dist_p2t) + np.mean(dist_t2p)) / 2.0
    
    return results


def compute_mask_iou(mask_pred, mask_target):
    """
    Compute bidirectional Chamfer Distance.
    Reference: Fan et al. "A Point Set Generation Network" (CVPR 2017)
    """
    if len(points_pred) == 0 or len(points_target) == 0:
        return {'chamfer_l1': float('inf'), 'chamfer_l2': float('inf'),
                'accuracy': float('inf'), 'completeness': float('inf')}
    
    tree_target = cKDTree(points_target)
    tree_pred = cKDTree(points_pred)
    
    dist_p2t, _ = tree_target.query(points_pred, k=1)
    accuracy_l1, accuracy_l2 = np.mean(dist_p2t), np.mean(dist_p2t ** 2)
    
    dist_t2p, _ = tree_pred.query(points_target, k=1)
    completeness_l1, completeness_l2 = np.mean(dist_t2p), np.mean(dist_t2p ** 2)
    
    return {
        'chamfer_l1': (accuracy_l1 + completeness_l1) / 2.0,
        'chamfer_l2': (accuracy_l2 + completeness_l2) / 2.0,
        'accuracy': accuracy_l1, 'completeness': completeness_l1
    }


def compute_hausdorff_distance(points_pred, points_target):
    """
    Compute bidirectional Hausdorff Distance (max error).
    Detects "false bridging" when target has holes but prediction connects them.
    """
    if len(points_pred) == 0 or len(points_target) == 0:
        return float('inf')
    
    tree_target = cKDTree(points_target)
    tree_pred = cKDTree(points_pred)
    
    dist_p2t, _ = tree_target.query(points_pred, k=1)
    dist_t2p, _ = tree_pred.query(points_target, k=1)
    
    return max(np.max(dist_p2t), np.max(dist_t2p))


def compute_fscore(points_pred, points_target, threshold=0.01):
    """
    Compute F-Score at given distance threshold.
    Reference: Tatarchenko et al. "What3D" (CVPR 2019)
    """
    if len(points_pred) == 0 or len(points_target) == 0:
        return {'f_score': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    tree_target = cKDTree(points_target)
    tree_pred = cKDTree(points_pred)
    
    dist_p2t, _ = tree_target.query(points_pred, k=1)
    precision = np.mean(dist_p2t < threshold)
    
    dist_t2p, _ = tree_pred.query(points_target, k=1)
    recall = np.mean(dist_t2p < threshold)
    
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'f_score': f_score, 'precision': precision, 'recall': recall}


def compute_mask_iou(mask_pred, mask_target):
    """
    Compute Mask IoU (Intersection over Union).
    
    Reference: Mask R-CNN (CVPR 2017), DeepLab series
    
    【基礎指標】衡量「實體面積」的重疊度
    邏輯: 如果沒撕開，分子(Intersection)不變，但分母(Union)變大
         （因為你多蓋住了洞），IoU 會下降。
    """
    mask_pred = (mask_pred > 0.5).astype(np.float32) if mask_pred.max() <= 1.0 else (mask_pred > 127).astype(np.float32)
    mask_target = (mask_target > 0.5).astype(np.float32) if mask_target.max() <= 1.0 else (mask_target > 127).astype(np.float32)
    
    intersection = np.sum(mask_pred * mask_target)
    union = np.sum(np.clip(mask_pred + mask_target, 0, 1))
    
    iou = intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
    return iou


def render_deformed_mesh_to_mask(deformed_points, faces, resolution=256):
    """Render deformed mesh to binary occupancy mask (OPTIMIZED)."""
    pts_pixel = ((deformed_points + 1) * 0.5 * (resolution - 1)).astype(np.int32)
    pts_pixel = np.clip(pts_pixel, 0, resolution - 1)
    
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    
    # Vectorized: prepare all triangles at once
    if len(faces) > 0:
        triangles = [pts_pixel[face].reshape(-1, 1, 2) for face in faces]
        cv2.fillPoly(mask, triangles, 255)
    
    return mask


def extract_all_boundary_points_pixel(mask, num_points=1024):
    """
    Extract boundary points in PIXEL coordinates (for 2D contour metrics).
    Includes all contours: outer boundary + hole boundaries.
    
    Args:
        mask: (H, W) binary mask
        num_points: max number of points to return
    
    Returns:
        (N, 2) boundary points in pixel coordinates
    """
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    # RETR_TREE gets all contours including holes (critical for tearing)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros((0, 2))
    
    all_pts = []
    for cnt in contours:
        pts = cnt.squeeze().reshape(-1, 2)
        if pts.ndim == 2 and pts.shape[0] > 0:
            all_pts.append(pts)
    
    if not all_pts:
        return np.zeros((0, 2))
    
    boundary_pts = np.vstack(all_pts).astype(np.float32)
    if boundary_pts.shape[0] > num_points:
        idx = np.random.choice(boundary_pts.shape[0], num_points, replace=False)
        boundary_pts = boundary_pts[idx]
    
    return boundary_pts  # Return in pixel coordinates for 2D metrics


def run_metric_evaluation(args, cfg, device, model, template_meshes, rest_cage):
    """
    Run comprehensive 2D metric evaluation on validation set.
    
    【核心指標：證明撕裂與拓撲正確 (Topology & Tearing)】
    - Hausdorff Distance (HD / HD95) - MICCAI standard, 輪廓一致性黃金標準
    - Boundary F-Score (BF Score) - Gated-SCNN (CVPR 2019), 邊緣細節專用
    
    【基礎指標：證明整體形狀吻合 (Global Shape)】
    - Mask IoU - Mask R-CNN (CVPR 2017), 實體面積重疊度
    - Contour Chamfer Distance - RPM-Net, 輪廓點平均最近距離
    """
    print("\n" + "="*70)
    print("🔬 2D DEFORMATION METRIC EVALUATION")
    print("="*70)
    print("\n【核心指標：證明撕裂與拓撲正確 (Topology & Tearing)】")
    print("  [TEARING] Hausdorff Distance (HD)  - MICCAI standard")
    print("  [TEARING] HD95                     - 95th percentile, more robust")
    print("  [TEARING] Boundary F-Score (BF)    - Gated-SCNN (CVPR 2019)")
    print("\n【基礎指標：證明整體形狀吻合 (Global Shape)】")
    print("  [SHAPE]   Mask IoU                 - Mask R-CNN (CVPR 2017)")
    print("  [SHAPE]   Contour Chamfer (CD)     - RPM-Net, 輪廓平均距離")
    print("="*70 + "\n")
    
    cage_num_vertices = cfg.model.get('cage_num_vertices', 96)
    cage_radius = cfg.model.get('cage_radius', 1.2)
    
    # Load validation dataset
    print("📊 Loading validation dataset...")
    val_dataset = CageDataset(
        cfg.data.split_file, cfg.data.dataset_dir, cfg.data.template_dir,
        cfg.data.template_names, split='val', num_points=cfg.data.num_points,
        cage_num_vertices=cage_num_vertices, cage_radius=cage_radius
    )
    
    num_samples = args.num_eval_samples or len(val_dataset)
    num_samples = min(num_samples, len(val_dataset))
    print(f"   Evaluating {num_samples} samples × {len(template_meshes)} templates")
    
    # Initialize metrics storage
    all_metrics = defaultdict(list)
    per_template_metrics = {t: defaultdict(list) for t in template_meshes.keys()}
    
    resolution = 256
    contour_points = 1024  # Points to sample from contours
    bf_threshold = 2.0     # BF Score pixel threshold (standard: 2 pixels)
    
    # Pre-compute rest_cage_batch once
    rest_cage_batch = rest_cage.unsqueeze(0)
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating"):
            sample = val_dataset[i]
            
            # Get target mask
            target_mask_np = sample['input'][0].numpy()
            if target_mask_np.max() <= 1.0:
                target_mask_np = (target_mask_np * 255).astype(np.uint8)
            
            # Resize target mask to evaluation resolution
            target_mask_resized = cv2.resize(
                target_mask_np.astype(np.float32),
                (resolution, resolution),
                interpolation=cv2.INTER_NEAREST
            )
            
            # OPTIMIZATION: Extract target contour ONCE per sample (shared across templates)
            target_contour = extract_all_boundary_points_pixel(target_mask_resized, contour_points)
            # Pre-build KDTree for target contour (used by all templates)
            target_contour_tree = cKDTree(target_contour) if len(target_contour) > 0 else None
            
            source_img = sample['input'][:3].to(device)
            
            for t_name, t_data in template_meshes.items():
                # Prepare model input
                template_mask = t_data['mask_tensor'].to(device)
                if template_mask.shape[-2:] != source_img.shape[-2:]:
                    template_mask = F.interpolate(
                        template_mask.unsqueeze(0),
                        size=source_img.shape[-2:],
                        mode='nearest'
                    ).squeeze(0)
                
                model_input = torch.cat([source_img, template_mask], dim=0).unsqueeze(0)
                
                # Forward pass
                output = model(model_input, rest_cage_batch)
                
                # Apply deformation
                cage_after_affine = utils.apply_affine(rest_cage_batch, output['affine_matrix'])
                cage_deformed = cage_after_affine + output['cage_offsets']
                
                mvc_weights = t_data['mvc_weights']
                deformed_verts = torch.mm(mvc_weights, cage_deformed.squeeze(0))
                
                if not args.no_residual:
                    # Apply residual
                    residual = model.compute_residual(
                        output['latent_z'],
                        deformed_verts.unsqueeze(0),
                        output['spatial_features']
                    )
                    deformed_verts = deformed_verts + residual.squeeze(0)
                deformed_verts_np = deformed_verts.cpu().numpy()
                
                # =========================================================
                # [新增] FILTERING STEP: 過濾掉過度拉伸的三角形 (Blue Debris)
                # =========================================================
                # 1. 計算每個三角形的拉伸率 (利用現有的 helper function)
                #    注意: t_data['ref_areas'] 是 template 原始每個三角形的面積
                quality = check_mesh_quality(deformed_verts_np, t_data['faces'], t_data['ref_areas'])
                area_ratios = quality['area_ratios']  # 這是 (N_faces,) 的陣列

                # 2. 設定閾值 (Threshold)
                #    邏輯: 如果面積變大超過 N 倍 (stretch_threshold)，視為撕裂/殘渣
                stretch_threshold = 2.0 

                # 3. 篩選 Valid Faces
                valid_mask = area_ratios < stretch_threshold
                filtered_faces = t_data['faces'][valid_mask]

                # 4. 使用篩選後的 Faces 進行渲染
                #    (這樣產生的 pred_mask 中間就會有乾淨的洞，不會被藍色填滿)
                pred_mask = render_deformed_mesh_to_mask(deformed_verts_np, filtered_faces, resolution)
                # =========================================================                

                # ==========================================
                # 1. Mask IoU (基礎指標：整體形狀)
                # ==========================================
                iou = compute_mask_iou(pred_mask, target_mask_resized)
                all_metrics['mask_iou'].append(iou)
                per_template_metrics[t_name]['mask_iou'].append(iou)
                
                # ==========================================
                # 2. Extract predicted contour for contour-based metrics
                # ==========================================
                pred_contour = extract_all_boundary_points_pixel(pred_mask, contour_points)
                
                # ==========================================
                # 3. OPTIMIZED: Compute all 2D contour metrics in single pass
                #    (reuse target KDTree)
                # ==========================================
                contour_metrics = compute_all_2d_contour_metrics(
                    pred_contour, target_contour, bf_threshold,
                    target_tree=target_contour_tree
                )
                
                # 核心指標：Hausdorff (HD & HD95)
                all_metrics['hausdorff'].append(contour_metrics['hausdorff'])
                all_metrics['hausdorff_95'].append(contour_metrics['hausdorff_95'])
                per_template_metrics[t_name]['hausdorff'].append(contour_metrics['hausdorff'])
                per_template_metrics[t_name]['hausdorff_95'].append(contour_metrics['hausdorff_95'])
                
                # 核心指標：Boundary F-Score
                all_metrics['bf_score'].append(contour_metrics['bf_score'])
                all_metrics['bf_precision'].append(contour_metrics['bf_precision'])
                all_metrics['bf_recall'].append(contour_metrics['bf_recall'])
                per_template_metrics[t_name]['bf_score'].append(contour_metrics['bf_score'])
                
                # 基礎指標：Contour Chamfer
                all_metrics['contour_chamfer'].append(contour_metrics['contour_chamfer'])
                per_template_metrics[t_name]['contour_chamfer'].append(contour_metrics['contour_chamfer'])
    
    # Print results
    print("\n" + "="*70)
    print("📊 2D DEFORMATION EVALUATION RESULTS")
    print("="*70)
    
    # 核心指標 - Tearing/Topology
    print("\n┌" + "─"*68 + "┐")
    print("│ 【核心指標】撕裂與拓撲正確性 (Topology & Tearing)" + " "*19 + "│")
    print("├" + "─"*68 + "┤")
    
    summary = {}
    core_metrics = ['hausdorff', 'hausdorff_95', 'bf_score', 'bf_precision', 'bf_recall']
    
    for metric in core_metrics:
        if metric in all_metrics:
            values = np.array(all_metrics[metric])
            values = values[np.isfinite(values)]
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                summary[metric] = {'mean': mean_val, 'std': std_val}
                
                if 'score' in metric or 'precision' in metric or 'recall' in metric:
                    print(f"│  {metric:28s}:  {mean_val:.4f} ± {std_val:.4f}" + " "*20 + "│")
                else:
                    print(f"│  {metric:28s}:  {mean_val:.2f} ± {std_val:.2f} px" + " "*17 + "│")
    
    print("└" + "─"*68 + "┘")
    
    # 基礎指標 - Global Shape
    print("\n┌" + "─"*68 + "┐")
    print("│ 【基礎指標】整體形狀吻合 (Global Shape)" + " "*28 + "│")
    print("├" + "─"*68 + "┤")
    
    base_metrics = ['mask_iou', 'contour_chamfer']
    
    for metric in base_metrics:
        if metric in all_metrics:
            values = np.array(all_metrics[metric])
            values = values[np.isfinite(values)]
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                summary[metric] = {'mean': mean_val, 'std': std_val}
                
                if 'iou' in metric:
                    print(f"│  {metric:28s}:  {mean_val:.4f} ± {std_val:.4f}" + " "*20 + "│")
                else:
                    print(f"│  {metric:28s}:  {mean_val:.2f} ± {std_val:.2f} px" + " "*17 + "│")
    
    print("└" + "─"*68 + "┘")
    
    # Per-template breakdown
    print("\n┌" + "─"*68 + "┐")
    print("│ PER-TEMPLATE BREAKDOWN" + " "*45 + "│")
    print("├" + "─"*68 + "┤")
    
    template_summary = {}
    for t_name in template_meshes.keys():
        print(f"│  [{t_name}]" + " "*(62-len(t_name)) + "│")
        template_summary[t_name] = {}
        for metric in ['mask_iou', 'hausdorff_95', 'bf_score', 'contour_chamfer']:
            if metric in per_template_metrics[t_name]:
                values = np.array(per_template_metrics[t_name][metric])
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    mean_val = np.mean(values)
                    template_summary[t_name][metric] = mean_val
                    if 'iou' in metric or 'score' in metric:
                        print(f"│    {metric:26s}:  {mean_val:.4f}" + " "*26 + "│")
                    else:
                        print(f"│    {metric:26s}:  {mean_val:.2f} px" + " "*22 + "│")
    
    print("└" + "─"*68 + "┘")
    
    # Save results
    results_path = os.path.join(args.output_dir, f"eval_metrics_{int(time.time())}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'overall': {k: {'mean': float(v['mean']), 'std': float(v['std'])} 
                       for k, v in summary.items()},
            'per_template': {t: {m: float(v) for m, v in metrics.items()} 
                            for t, metrics in template_summary.items()},
            'config': {
                'checkpoint': args.checkpoint_path,
                'num_samples': num_samples,
                'templates': list(template_meshes.keys())
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Citation guidance
    print("\n" + "="*70)
    print("📚 2D METRIC CITATIONS FOR YOUR PAPER")
    print("="*70)
    print("""
When reporting these metrics, cite:

【核心指標 - 撕裂與拓撲】

1. **Hausdorff Distance (HD / HD95)**:
   Standard metric in MICCAI for medical image segmentation.
   Also used in CVPR/ECCV shape analysis papers.
   
   針對 Tearing 任務的意義:
   - 如果 Target 有洞但 Template 沒撕開 → HD 暴增
   - 如果 Template 正確撕開 → HD 驟降

2. **Boundary F-Score (BF Score)**:
   Takikawa et al. "Gated-SCNN: Gated Shape CNNs for 
   Semantic Segmentation" CVPR 2019
   
   專門證明「撕裂路徑」的精準度，不受面積影響。

【基礎指標 - 整體形狀】

3. **Mask IoU**:
   He et al. "Mask R-CNN" ICCV 2017
   Chen et al. "DeepLab" CVPR/ECCV series
   
   衡量實體面積重疊度。

4. **Contour Chamfer Distance**:
   Adapted from RPM-Net, Deep Closest Point for 2D.
   衡量輪廓點的平均最近距離。

Rationale for 2D Deformation + Tearing:
- HD95: 直接衡量「最不合群的點」，撕裂錯誤會被放大
- BF Score: 只看邊緣，小洞也能被準確評估
- Mask IoU: 整體形狀的面積吻合度
- Contour CD: 輪廓的整體貼合度
""")
    
    return summary


def extract_gt_contour(input_tensor, num_points=1024):
    """Extract ground truth contour from input image."""
    if input_tensor.dim() == 3:
        img = input_tensor[0].cpu().numpy()
    else:
        img = input_tensor.cpu().numpy()
    
    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    _, thresh = cv2.threshold(img_uint8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros((0, 2))
    
    all_pts = []
    for cnt in contours:
        pts = cnt.squeeze().reshape(-1, 2)
        if pts.ndim == 2 and pts.shape[0] > 0:
            all_pts.append(pts)
    
    if not all_pts:
        return np.zeros((0, 2))
    
    target_pts = np.vstack(all_pts).astype(np.float32)
    
    if target_pts.shape[0] > num_points:
        idx = np.random.choice(target_pts.shape[0], num_points, replace=False)
        target_pts = target_pts[idx]
    
    h, w = img.shape[:2]
    return (target_pts / np.array([w-1, h-1])) * 2 - 1


def get_real_samples(pkl_path, N, num_points=1024):
    """Load real data from pickle file."""
    print(f"🥒 Loading Real Data from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"   Found {len(data)} samples")
    
    samples_list = []
    
    for idx in range(min(N, len(data))):
        item = data[idx]
        img_np = item[0]
        mask_np = item[1]
        label_str = str(item[2])
        
        # Clean background using mask
        mask_binary = (mask_np > 127).astype(np.uint8)
        mask_3ch = np.stack([mask_binary]*3, axis=-1)
        cleaned_img = img_np * mask_3ch
        
        input_tensor = torch.from_numpy(cleaned_img).float().div(255.0).permute(2, 0, 1)
        
        # Extract contour for GT visualization
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        all_pts = []
        for cnt in contours:
            pts = cnt.squeeze().reshape(-1, 2)
            if pts.ndim == 2 and pts.shape[0] > 0:
                all_pts.append(pts)
        
        if len(all_pts) > 0:
            target_pts = np.vstack(all_pts).astype(np.float32)
            if target_pts.shape[0] > num_points:
                idx_pts = np.random.choice(target_pts.shape[0], num_points, replace=False)
                target_pts = target_pts[idx_pts]
            target_pts = (target_pts / VIZ_SCALE) * 2 - 1
        else:
            target_pts = np.zeros((0, 2))
            
        samples_list.append({
            'input': input_tensor,
            'target_points': torch.from_numpy(target_pts),
            'label': label_str
        })
        
    return samples_list


def get_mnist_samples(num_points=1024):
    """Load MNIST samples for testing."""
    print("📥 Loading MNIST dataset...")
    tf = transforms.Compose([
        transforms.Resize((int(VIZ_SCALE), int(VIZ_SCALE))),
        transforms.ToTensor(), 
    ])
    dataset = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=tf)
    samples_dict = {}
    kernel = np.ones((15, 15), np.uint8)
    
    for img, label in dataset:
        if label not in samples_dict:
            mask_np = (img.squeeze().numpy() * 255).astype(np.uint8)
            mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
            mask_dilated = mask_np
            input_tensor = torch.from_numpy(mask_dilated).float().div(255.0).unsqueeze(0)
            
            _, thresh = cv2.threshold(mask_dilated, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            all_pts = []
            for cnt in contours:
                pts = cnt.squeeze().reshape(-1, 2)
                if pts.ndim == 2 and pts.shape[0] > 0:
                    all_pts.append(pts)
            
            if len(all_pts) > 0:
                target_pts = np.vstack(all_pts).astype(np.float32)
                if target_pts.shape[0] > num_points:
                    idx = np.random.choice(target_pts.shape[0], num_points, replace=False)
                    target_pts = target_pts[idx]
                target_pts = (target_pts / VIZ_SCALE) * 2 - 1
            else:
                target_pts = np.zeros((0, 2))
            
            samples_dict[label] = {
                'input': input_tensor, 
                'target_points': torch.from_numpy(target_pts),
                'label': str(label)
            }
        if len(samples_dict) == 10:
            break   
    return [samples_dict[i] for i in range(10)]


def get_omniglot_samples(num_points=1024):
    """Load Omniglot samples, apply inversion and dilation like MNIST."""
    print("📥 Loading Omniglot dataset...")
    tf = transforms.Compose([
        transforms.Resize((int(VIZ_SCALE), int(VIZ_SCALE))),
        transforms.ToTensor(),
        transforms.Grayscale() # Ensure 1 channel
    ])
    # Use 'background' set which has many alphabets
    dataset = torchvision.datasets.Omniglot(root='./data_omniglot', background=True, download=True, transform=tf)
    
    samples_list = []
    seen_labels = set()
    kernel = np.ones((15, 15), np.uint8) # Same thickening kernel as MNIST
    
    # Iterate to find 10 unique characters
    for i in range(len(dataset)):
        img, label = dataset[i]
        
        if label in seen_labels:
            continue
            
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
        
        # Omniglot is typically black stroke on white bg (or vice versa depending on load)
        # We need White stroke on Black background (like MNIST logic)
        # Simple check: if background is white (mean > 127), invert it.
        if np.mean(img_np) > 127:
            img_np = 255 - img_np
            
        # Apply thickening (dilation) same as MNIST
        mask_dilated = cv2.dilate(img_np, kernel, iterations=1)
        
        input_tensor = torch.from_numpy(mask_dilated).float().div(255.0).unsqueeze(0)
        
        _, thresh = cv2.threshold(mask_dilated, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        all_pts = []
        for cnt in contours:
            pts = cnt.squeeze().reshape(-1, 2)
            if pts.ndim == 2 and pts.shape[0] > 0:
                all_pts.append(pts)
        
        if len(all_pts) > 0:
            target_pts = np.vstack(all_pts).astype(np.float32)
            if target_pts.shape[0] > num_points:
                idx = np.random.choice(target_pts.shape[0], num_points, replace=False)
                target_pts = target_pts[idx]
            target_pts = (target_pts / VIZ_SCALE) * 2 - 1
        else:
            target_pts = np.zeros((0, 2))
            
        samples_list.append({
            'input': input_tensor,
            'target_points': torch.from_numpy(target_pts),
            'label': f"Omni_{label}"
        })
        
        seen_labels.add(label)
        if len(samples_list) == 10:
            break
            
    return samples_list


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    cfg = OmegaConf.load(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate rest cage
    cage_num_vertices = cfg.model.get('cage_num_vertices', 96)
    cage_radius = cfg.model.get('cage_radius', 1.2)
    rest_cage = utils.generate_circular_cage(cage_num_vertices, radius=cage_radius, device=device)
    
    # Load Templates
    if args.specific_templates:
        M_templates = args.specific_templates
    else:
        M_templates = cfg.data.template_names
    
    template_meshes = {}
    print("🔧 Loading Templates...")
    for t_name in M_templates:
        try:
            mesh_data = create_template_mesh(cfg.data.template_dir, t_name, device, rest_cage)
            if mesh_data:
                template_meshes[t_name] = mesh_data
                print(f"   ✅ {t_name}: {mesh_data['vertices_norm'].shape[0]} points")
        except Exception as e:
            print(f"   ⚠️  Failed {t_name}: {e}")

    if not template_meshes:
        print("❌ No templates loaded!")
        return

    # Load Model
    model = load_model(cfg, args.checkpoint_path, device)
    
    # === METRIC EVALUATION MODE ===
    if args.evaluate_metrics:
        run_metric_evaluation(args, cfg, device, model, template_meshes, rest_cage)
        return
    
    # === VISUALIZATION MODE ===
    # Prepare Samples
    if args.real:
        viz_samples = get_real_samples(args.real, args.N)
        suffix = "real"
    elif args.mnist:
        viz_samples = get_mnist_samples()
        suffix = "mnist"
    elif args.omniglot:
        viz_samples = get_omniglot_samples()
        suffix = "omniglot"
    else:
        print("📊 Loading Validation Dataset...")
        val_dataset = CageDataset(
            cfg.data.split_file, cfg.data.dataset_dir, cfg.data.template_dir,
            cfg.data.template_names, split='val', num_points=cfg.data.num_points,
            cage_num_vertices=cage_num_vertices, cage_radius=cage_radius
        )
        N = min(args.N, len(val_dataset))
        viz_samples = [val_dataset[i] for i in range(N)]
        suffix = "val"

    N = len(viz_samples)
    if N == 0:
        print("⚠️  No samples found")
        return

    M = len(template_meshes)
    template_names_sorted = list(template_meshes.keys())
    
    print(f"\n🎨 Creating {N}x{M} visualization...")
    if args.no_residual:
        print("   ⚠️  Residual flow DISABLED (cage only)")
    else:
        print("   ✅ Using FULL pipeline (cage + residual)")
    
    # Color map for mesh quality
    cdict = {
        'red':   ((0.0, 0.5, 0.5), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.8, 0.8)),
        'blue':  ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)),
        'alpha': ((0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.3, 0.3))
    }
    density_cmap = LinearSegmentedColormap('Density', cdict)

    fig = plt.figure(figsize=(M * 3, N * 3), facecolor='white')
    gs = fig.add_gridspec(N, M, hspace=0.1, wspace=0.05)
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(viz_samples, desc="Processing")):
            
            # Prepare input
            if args.mnist or args.omniglot:
                source_img = sample['input'].repeat(3, 1, 1).to(device)
                label_str = sample.get('label', str(i))
            else:
                source_img = sample['input'][:3, :, :].to(device)
                label_str = str(sample.get('label', i))
            
            # Get GT points for visualization
            if 'target_points' in sample:
                pts = sample['target_points']
                if isinstance(pts, torch.Tensor):
                    pts = pts.cpu().numpy()
            else:
                pts = extract_gt_contour(sample['input'][:3])
            
            target_points_viz = (pts + 1) * (VIZ_SCALE / 2.0)

            # Process each template
            for j, t_name in enumerate(template_names_sorted):
                t_data = template_meshes[t_name]
                
                # === STEP 1: Prepare Model Input ===
                # Input format: [RGB(3 channels) + Template Mask(1 channel)]
                template_mask = t_data['mask_tensor'].to(device)
                
                if template_mask.shape[-2:] != source_img.shape[-2:]:
                    template_mask = F.interpolate(
                        template_mask.unsqueeze(0), 
                        size=source_img.shape[-2:], 
                        mode='nearest'
                    ).squeeze(0)

                model_input = torch.cat([source_img, template_mask], dim=0).unsqueeze(0)
                
                # === STEP 2: Model Forward (Affine + Cage) ===
                rest_cage_batch = rest_cage.unsqueeze(0)
                output = model(model_input, rest_cage_batch)
                
                affine_mat = output['affine_matrix']      # (1, 2, 3)
                cage_offsets = output['cage_offsets']     # (1, K, 2)
                latent_z = output['latent_z']             # (1, latent_dim)
                
                # === STEP 3: Apply Cage Deformation ===
                cage_after_affine = utils.apply_affine(rest_cage_batch, affine_mat)
                cage_deformed = cage_after_affine + cage_offsets
                
                # Deform template vertices using MVC weights
                mvc_weights = t_data['mvc_weights']
                deformed_verts = torch.mm(mvc_weights, cage_deformed.squeeze(0))  # (N, 2)
                
                # === STEP 4: Apply Residual Flow (FULL MODEL) ===
                if not args.no_residual:
                    residual = model.compute_residual(latent_z, deformed_verts.unsqueeze(0), output['spatial_features'])
                    deformed_verts = deformed_verts + residual.squeeze(0)
                
                # === STEP 5: Visualization ===
                deformed_verts_np = deformed_verts.cpu().numpy()
                viz_verts = (deformed_verts_np + 1) * (VIZ_SCALE / 2.0)
                
                # Check mesh quality
                metrics = check_mesh_quality(viz_verts, t_data['faces'], t_data['ref_areas'])
                face_colors = np.clip(np.log2(metrics['area_ratios'] + 1e-6) / 4.0 + 0.5, 0, 1)
                
                ax = fig.add_subplot(gs[i, j])
                
                # Draw mesh faces (colored by area ratio)
                if len(t_data['faces']) > 0:
                    ax.tripcolor(viz_verts[:, 0], viz_verts[:, 1], t_data['faces'], 
                                facecolors=face_colors, cmap=density_cmap, 
                                shading='flat', vmin=0, vmax=1, alpha=0.8)
                    # Draw mesh edges
                    ax.triplot(viz_verts[:, 0], viz_verts[:, 1], t_data['faces'], 
                              'k-', linewidth=0.3, alpha=0.4)
                
                # Draw GT contour (green dots)
                if len(target_points_viz) > 0:
                    ax.plot(target_points_viz[:, 0], target_points_viz[:, 1], 
                           '.', c='lime', ms=2.0, alpha=0.6, label='GT')
                
                # Draw cage if requested
                if args.show_cage:
                    cage_viz = (cage_deformed.squeeze(0).cpu().numpy() + 1) * (VIZ_SCALE / 2.0)
                    cage_closed = np.vstack([cage_viz, cage_viz[0:1]])
                    ax.plot(cage_closed[:, 0], cage_closed[:, 1], 'r-', lw=1.5, alpha=0.8, label='Cage')
                    ax.plot(cage_viz[:, 0], cage_viz[:, 1], 'ro', ms=2, alpha=0.8)
                
                ax.set_xlim(0, VIZ_SCALE)
                ax.set_ylim(VIZ_SCALE, 0)
                ax.axis('off')
                
                # Labels
                if i == 0:
                    ax.set_title(t_name, fontsize=10, fontweight='bold')
                if j == 0:
                    ax.text(-20, VIZ_SCALE/2, label_str, 
                           rotation=90, va='center', fontsize=12, fontweight='bold')

    out_path = os.path.join(args.output_dir, f"viz_{suffix}_{int(time.time())}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=VIZ_DPI, facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {out_path}")

if __name__ == '__main__':
    main()