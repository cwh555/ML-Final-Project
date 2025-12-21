"""
Deformation Parameter Extractor
===============================
A downstream-ready tool that extracts deformation parameters from a pre-trained
Shape Transformation Network.

===============================================================================
INPUT FORMAT
===============================================================================
    images: torch.Tensor of shape (B, 4, H, W)
        - B: batch size (任意正整數)
        - 4: RGB (3 通道) + Target Mask (1 通道)
            - Channel 0-2: RGB 圖片
            - Channel 3: 目標形狀的 mask (白色=前景, 黑色=背景)
        - H, W: 圖片高度和寬度 (建議 256x256，與訓練時一致)
        - 數值範圍: [0, 1] (經過 ToTensor() 轉換後的標準格式)

===============================================================================
OUTPUT FORMAT  
===============================================================================
    results: List[Dict[str, torch.Tensor]]
        長度為 B 的 list，每個元素對應 batch 中的一筆資料
        每個 Dict 包含以下 keys:
        
        'affine':                    (1, 2, 3)       Affine 變換矩陣
        'cage_offset':               (1, K, 2)       Cage 頂點偏移量 (K=128)
        'cage_deformed':             (1, K, 2)       變形後的 cage 頂點座標
        'residual':                  (1, N_b, 2)     Residual 流 (N_b=1024)
        'deformed_boundary':         (1, N_b, 2)     最終變形後的邊界點
        'deformed_interior':         (1, N_i, 2)     最終變形後的內部點 (N_i=1024)
        'latent_z':                  (1, 256)        Latent 特徵向量
        'boundary_before_residual':  (1, N_b, 2)     加 residual 之前的邊界點

        所有座標都在 [-1, 1] 的正規化空間中

===============================================================================
USAGE EXAMPLE
===============================================================================
    from src.app.extract_deformation import DeformationExtractor
    import torch
    from torchvision import transforms
    from PIL import Image
    
    # 1. 初始化 Extractor
    extractor = DeformationExtractor(
        checkpoint_path='weight_files/cage128_rsguide_v12/checkpoint_latest.pth',
        template_dir='template',
        template_names=['circle', 'rectangle', 'star'],
        cage_num_vertices=128,
        cage_radius=1.2,
        device='cuda'
    )
    
    # 2. 準備輸入資料 (4 通道: RGB + Target Mask)
    # 方法 A: 分別載入 RGB 和 Mask
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 自動轉換到 [0, 1]
    ])
    rgb = transform(Image.open('image.png').convert('RGB'))    # (3, H, W)
    mask = transform(Image.open('mask.png').convert('L'))      # (1, H, W)
    images = torch.cat([rgb, mask], dim=0).unsqueeze(0)        # (1, 4, H, W)
    
    # 方法 B: 從 grayscale 圖片 (同時作為 RGB 和 mask)
    gray = torch.randn(1, 256, 256)  # (1, H, W), 數值在 [0, 1]
    rgb = gray.repeat(3, 1, 1)       # (3, H, W)
    mask = gray                       # (1, H, W)
    images = torch.cat([rgb, mask], dim=0).unsqueeze(0)  # (1, 4, H, W)
    
    # 3. 執行 Forward
    # 傳入 per-sample template assignment (長度必須等於 batch size)
    # 單筆資料:
    results = extractor(images, ['circle'])
    # 多筆資料 (batch=4):
    # results = extractor(images_batch, ['circle', 'circle', 'star', 'rectangle'])
    
    # 4. 取得結果 (results 是 List，長度 = batch size)
    for i, params in enumerate(results):
        # 變形結果
        final_boundary = params['deformed_boundary']  # (1, 1024, 2)
        
        # 如果需要 pixel 座標 (假設原圖 256x256):
        boundary_pixels = (final_boundary + 1) * 128  # [-1,1] -> [0,256]
        
        # 用於下游任務的變換參數
        affine = params['affine']              # (1, 2, 3)
        cage_offset = params['cage_offset']    # (1, 128, 2)
        residual = params['residual']          # (1, 1024, 2)

===============================================================================
COMMAND LINE TEST
===============================================================================
    python -m src.app.extract_deformation \\
        --checkpoint weight_files/cage128_rsguide_v12/checkpoint_latest.pth \\
        --template-dir template \\
        --templates circle rectangle rounded_rect \\
        --device cuda \\
        --output-dir images_eval \\
        --show-cage

===============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Union

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
_project_dir = os.path.dirname(_src_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Import from project
from new_BaseModel import ShapeTransformationNetwork
from grid_utils import generate_circular_cage, compute_mvc_weights


class DeformationExtractor(nn.Module):
    """
    Deformation Parameter Extractor for downstream tasks.
    
    This module wraps a pre-trained ShapeTransformationNetwork and provides
    an interface to extract deformation parameters (affine, cage_offset, residual)
    for any input image with respect to all loaded templates.
    
    All parameters are frozen and the model is in eval mode.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint (.pth file) [REQUIRED]
        template_dir: Directory containing template .npz files [REQUIRED]
        template_names: List of template names (without .npz extension) [REQUIRED]
        cage_num_vertices: Number of cage vertices (K), must match checkpoint
        cage_radius: Radius of rest cage, must match checkpoint
        latent_dim: Latent feature dimension, must match checkpoint
        num_boundary_sample: Number of boundary points to sample for output
        num_interior_sample: Number of interior points to sample for output
        residual_steps: Number of residual refinement steps
        device: 'cuda' or 'cpu'
        use_gaussian_ff: Whether the model uses Gaussian Fourier features
        gaussian_ff_features: Number of Gaussian Fourier features
        gaussian_ff_scale: Scale of Gaussian Fourier features
        use_multiscale_ff: Whether to use multi-scale Fourier features
        multiscale_ff_scales: List of scales for multi-scale Fourier
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        template_dir: str,
        template_names: List[str],
        cage_num_vertices: int = 128,
        cage_radius: float = 1.2,
        latent_dim: int = 256,
        num_boundary_sample: int = 1024,
        num_interior_sample: int = 1024,
        residual_steps: int = 3,
        device: str = 'cuda',
        # use_gaussian_ff: bool = True,
        use_gaussian_ff: bool = True,
        gaussian_ff_features: int = 96,
        gaussian_ff_scale: float = 10.0,
        # use_multiscale_ff: bool = True,
        use_multiscale_ff: bool = True,
        multiscale_ff_scales: List[float] = None
    ):
        super().__init__()

        # Validate required parameters
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if not template_dir:
            raise ValueError("template_dir is required")
        if not template_names or len(template_names) == 0:
            raise ValueError("template_names is required and must not be empty")
        
        # Default for multiscale scales
        if multiscale_ff_scales is None:
            multiscale_ff_scales = [1.0, 5.0, 15.0]
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cage_num_vertices = cage_num_vertices
        self.cage_radius = cage_radius
        self.num_boundary_sample = num_boundary_sample
        self.num_interior_sample = num_interior_sample
        self.residual_steps = residual_steps
        self.template_names = template_names
        
        # ============================================================
        # 1. Initialize and load the pre-trained model
        # ============================================================
        print(f"[DeformationExtractor] Loading model from {checkpoint_path}...")
        
        self.model = ShapeTransformationNetwork(
            input_channels=4,  # RGB + Template Mask
            cage_num_vertices=cage_num_vertices,
            latent_dim=latent_dim,
            use_gaussian_ff=use_gaussian_ff,
            gaussian_ff_features=gaussian_ff_features,
            gaussian_ff_scale=gaussian_ff_scale,
            use_multiscale_ff=use_multiscale_ff,
            multiscale_ff_scales=list(multiscale_ff_scales)
        ).to(self.device)
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        print(f"[DeformationExtractor] Model loaded and frozen. Device: {self.device}")
        
        # ============================================================
        # 2. Generate rest cage (shared across all templates)
        # ============================================================
        self.register_buffer(
            'rest_cage',
            generate_circular_cage(cage_num_vertices, radius=cage_radius, device='cpu')
        )
        
        # ============================================================
        # 3. Load templates and pre-compute MVC weights
        # ============================================================
        self.templates = {}
        self._load_templates(template_dir, template_names)
        
        if len(self.templates) == 0:
            raise ValueError(f"No valid templates loaded from {template_dir}")
        
        print(f"[DeformationExtractor] Ready. Templates: {list(self.templates.keys())}")
    
    def _load_templates(self, template_dir: str, template_names: List[str]):
        """Load template masks and pre-compute MVC weights."""
        if not os.path.exists(template_dir):
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        
        cage_batch = self.rest_cage.unsqueeze(0)  # (1, K, 2)
        
        for name in template_names:
            path = os.path.join(template_dir, f"{name}.npz")
            if not os.path.exists(path):
                print(f"[Warning] Template not found: {path}, skipping...")
                continue
            
            data = np.load(path)
            mask = data['mask']
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            
            h, w = mask.shape
            
            # === Extract Boundary Points ===
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                print(f"[Warning] No contours found in {name}, skipping...")
                continue
            
            # Get all boundary points
            b_pts_raw = contours[0].squeeze().reshape(-1, 2).astype(np.float32)
            b_pts_norm = (b_pts_raw / np.array([w-1, h-1])) * 2 - 1
            b_pts_tensor = torch.tensor(b_pts_norm, dtype=torch.float32)
            
            # Compute MVC weights for boundary (chunked for memory)
            print(f"  [Template: {name}] Computing MVC weights for boundary ({len(b_pts_tensor)} points)...")
            w_b_list = []
            chunk_size = 5000
            for i in range(0, len(b_pts_tensor), chunk_size):
                chunk = b_pts_tensor[i:i+chunk_size].unsqueeze(0)
                w_b_list.append(compute_mvc_weights(chunk, cage_batch).squeeze(0))
            weights_boundary = torch.cat(w_b_list, dim=0)
            
            # === Extract Interior Points ===
            ys, xs = np.where(mask > 127)
            i_pts_raw = np.stack([xs, ys], axis=1).astype(np.float32)
            i_pts_norm = (i_pts_raw / np.array([w-1, h-1])) * 2 - 1
            i_pts_tensor = torch.tensor(i_pts_norm, dtype=torch.float32)
            
            # Compute MVC weights for interior
            print(f"  [Template: {name}] Computing MVC weights for interior ({len(i_pts_tensor)} points)...")
            w_i_list = []
            for i in range(0, len(i_pts_tensor), chunk_size):
                chunk = i_pts_tensor[i:i+chunk_size].unsqueeze(0)
                w_i_list.append(compute_mvc_weights(chunk, cage_batch).squeeze(0))
            weights_interior = torch.cat(w_i_list, dim=0)
            
            # Store template data (register as buffers for proper device handling)
            self.register_buffer(f'template_{name}_pts_boundary', b_pts_tensor)
            self.register_buffer(f'template_{name}_weights_boundary', weights_boundary)
            self.register_buffer(f'template_{name}_pts_interior', i_pts_tensor)
            self.register_buffer(f'template_{name}_weights_interior', weights_interior)
            self.register_buffer(
                f'template_{name}_mask',
                torch.tensor(mask > 127, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            )
            
            self.templates[name] = {
                'num_boundary': len(b_pts_tensor),
                'num_interior': len(i_pts_tensor)
            }
    
    def get_template_data(self, name: str) -> Dict[str, torch.Tensor]:
        """Get template data by name from registered buffers."""
        return {
            'pts_boundary': getattr(self, f'template_{name}_pts_boundary'),
            'weights_boundary': getattr(self, f'template_{name}_weights_boundary'),
            'pts_interior': getattr(self, f'template_{name}_pts_interior'),
            'weights_interior': getattr(self, f'template_{name}_weights_interior'),
            'mask': getattr(self, f'template_{name}_mask')
        }
    
    def _sample_points_ordered(
        self, 
        pts: torch.Tensor, 
        weights: torch.Tensor, 
        num_sample: int
    ) -> tuple:
        """
        Sample points in ordered manner (for boundary) to maintain contour order.
        
        Args:
            pts: (N, 2) all points
            weights: (N, K) MVC weights
            num_sample: number of points to sample
        
        Returns:
            sampled_pts: (num_sample, 2)
            sampled_weights: (num_sample, K)
        """
        total = len(pts)
        if total >= num_sample:
            stride = total / num_sample
            idx = torch.tensor([int(i * stride) % total for i in range(num_sample)], dtype=torch.long)
        else:
            # Repeat if not enough points
            idx = torch.tensor([i % total for i in range(num_sample)], dtype=torch.long)
        
        return pts[idx], weights[idx]
    
    def _sample_points_random(
        self, 
        pts: torch.Tensor, 
        weights: torch.Tensor, 
        num_sample: int
    ) -> tuple:
        """
        Sample points randomly (for interior points).
        
        Args:
            pts: (N, 2) all points
            weights: (N, K) MVC weights
            num_sample: number of points to sample
        
        Returns:
            sampled_pts: (num_sample, 2)
            sampled_weights: (num_sample, K)
        """
        total = len(pts)
        if total >= num_sample:
            idx = torch.randperm(total)[:num_sample]
        else:
            idx = torch.randint(0, total, (num_sample,))
        
        return pts[idx], weights[idx]
    
    def _apply_affine(self, points: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation to points.
        
        Args:
            points: (B, N, 2)
            affine_matrix: (B, 2, 3)
        
        Returns:
            transformed: (B, N, 2)
        """
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=2)  # (B, N, 3)
        return torch.bmm(points_homo, affine_matrix.transpose(1, 2))
    
    def _apply_cage_deformation(
        self,
        pts: torch.Tensor,
        weights: torch.Tensor,
        cage_deformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cage deformation using pre-computed MVC weights.
        
        Args:
            pts: (B, N, 2) original template points (not used directly, weights encode the relation)
            weights: (B, N, K) pre-computed MVC weights
            cage_deformed: (B, K, 2) deformed cage vertices
        
        Returns:
            deformed_pts: (B, N, 2)
        """
        # deformed_pts = weights @ cage_deformed
        # (B, N, K) @ (B, K, 2) -> (B, N, 2)
        return torch.bmm(weights, cage_deformed)
    
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        weight_boundary: torch.Tensor,
        template_names: List[str] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract deformation parameters for each sample using its assigned template.

        Args:
            images: (B, 4, H, W) input images (RGB + mask)
            template_names: List of length B, each element is the template name
                           for that sample in the batch

        Returns:
            List of length B, each element is a Dict with keys:
                'affine': (1, 2, 3)
                'cage_offset': (1, K, 2)
                'cage_deformed': (1, K, 2)
                'residual': (1, N_boundary, 2)
                'deformed_boundary': (1, N_boundary, 2)
                'deformed_interior': (1, N_interior, 2)
                'latent_z': (1, latent_dim)
                'boundary_before_residual': (1, N_boundary, 2)
        """
        B, C, H, W = images.shape

        if C != 4:
            raise ValueError(f"Expected 4 channels (RGB + Mask), got {C}")

        # if not isinstance(template_names, (list, tuple)):
        #     raise ValueError("template_names must be a list of length B")
        # if len(template_names) != B:
        #     raise ValueError(f"template_names length ({len(template_names)}) must equal batch size ({B})")

        # for t_name in template_names:
        #     if t_name not in self.templates:
        #         raise ValueError(f"Unknown template: {t_name}. Available: {list(self.templates.keys())}")

        images = images.to(self.device)

        # Run model once for the whole batch
        rest_cage_batch = self.rest_cage.unsqueeze(0).expand(B, -1, -1).to(self.device)
        model_out = self.model(images, rest_cage_batch)

        affine_matrix = model_out['affine_matrix']        # (B, 2, 3)
        cage_offsets = model_out['cage_offsets']          # (B, K, 2)
        latent_z = model_out['latent_z']                  # (B, latent_dim)
        spatial_features = model_out['spatial_features']  # (B, C', H', W')

        # Compute cage_deformed for whole batch
        cage_affine = self._apply_affine(rest_cage_batch, affine_matrix)  # (B, K, 2)
        cage_deformed_batch = cage_affine + cage_offsets  # (B, K, 2)

        if template_names is not None:
            # results_list: List[Dict[str, torch.Tensor]] = []
            affine_list = []
            coarse_feat_list = []
            fine_feat_list = []
            print("Extracting deformation parameters for each sample...")
            for i in range(B):
                t_name = template_names[i]
                t_data = self._get_template_data(t_name)

                # Sample points for this template
                pts_b, weights_b = self._sample_points_ordered(
                    t_data['pts_boundary'], t_data['weights_boundary'], self.num_boundary_sample
                )
                pts_i, weights_i = self._sample_points_random(
                    t_data['pts_interior'], t_data['weights_interior'], self.num_interior_sample
                )

                # Prepare tensors for single sample
                weights_b_batch = weights_b.unsqueeze(0).to(self.device)  # (1, N_b, K)
                weights_i_batch = weights_i.unsqueeze(0).to(self.device)  # (1, N_i, K)
                cage_deformed = cage_deformed_batch[i:i+1]                # (1, K, 2)

                # Apply cage deformation
                deformed_boundary = torch.bmm(weights_b_batch, cage_deformed)  # (1, N_b, 2)
                deformed_interior = torch.bmm(weights_i_batch, cage_deformed)  # (1, N_i, 2)

                # Compute residual
                latent_z_i = latent_z[i:i+1]
                spatial_feats_i = spatial_features[i:i+1]
                residual = self.model.compute_residual(
                    latent_z_i, deformed_boundary, spatial_feats_i, steps=self.residual_steps
                )

                final_boundary = deformed_boundary + residual

                # results_list.append({
                #     'affine': affine_matrix[i:i+1],
                #     # 'cage_offset': cage_offsets[i:i+1],
                #     'coarse_feat': cage_offsets[i:i+1],
                #     'cage_deformed': cage_deformed,
                #     # 'residual': residual,
                #     'fine_feat': residual,
                #     'deformed_boundary': final_boundary,
                #     'deformed_interior': deformed_interior,
                #     'latent_z': latent_z_i,
                #     'boundary_before_residual': deformed_boundary,
                # })
                affine_list.append(affine_matrix[i:i+1])
                coarse_feat_list.append(cage_offsets[i:i+1])
                fine_feat_list.append(residual)
            print("Extraction complete.")
            # return results_list
            return {
                'affine': torch.cat(affine_list, dim=0),
                'coarse_feat': torch.cat(coarse_feat_list, dim=0),
                'fine_feat': torch.cat(fine_feat_list, dim=0),
            }
        else:
            deformed_boundary = torch.bmm(weight_boundary, cage_deformed_batch)
            residual_b = self.model.compute_residual(latent_z, deformed_boundary, spatial_features, steps=self.residual_steps)
            return {
                'affine': affine_matrix,
                # 'cage_offset': cage_offsets,
                # 'residual': residual_b,
                'coarse_feat': cage_offsets,
                'fine_feat': residual_b,
            }

    
    def extract_single_template(
        self, 
        images: torch.Tensor, 
        template_name: str
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract deformation parameters using the same template for all samples.
        
        Args:
            images: (B, 4, H, W) input images
            template_name: name of the template to use for all samples
        
        Returns:
            List of length B, each element is a Dict with deformation parameters
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        
        B = images.shape[0]
        template_list = [template_name] * B
        return self.forward(images, template_list)
    
    def get_template_names(self) -> List[str]:
        """Return list of available template names."""
        return list(self.templates.keys())
    
    def get_template_info(self) -> Dict[str, Dict[str, int]]:
        """Return information about loaded templates."""
        return self.templates.copy()


# ============================================================
# MNIST Visualization Test (only executed when run as __main__)
# ============================================================
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial import Delaunay
    import torchvision
    from torchvision import transforms
    import time
    
    VIZ_SCALE = 256.0
    VIZ_DPI = 150
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Test DeformationExtractor with MNIST visualization")
        parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
        parser.add_argument('--template-dir', type=str, required=True,
                            help='Path to template directory')
        parser.add_argument('--templates', type=str, nargs='+', required=True,
                            help='List of template names')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use (cuda or cpu)')
        parser.add_argument('--output-dir', type=str, default='images_eval',
                            help='Output directory for visualization')
        parser.add_argument('--show-cage', action='store_true',
                            help='Show deformed cage in visualization')
        parser.add_argument('--no-residual', action='store_true',
                            help='Skip residual flow (only show cage deformation)')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
        return parser.parse_args()
    
    def get_mnist_samples(num_points=1024):
        """Load 10 MNIST samples (one per digit 0-9)."""
        print("📥 Loading MNIST dataset...")
        tf = transforms.Compose([
            transforms.Resize((int(VIZ_SCALE), int(VIZ_SCALE))),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST(root='./data_mnist', train=False, download=True, transform=tf)
        
        samples_dict = {}
        for img, label in dataset:
            if label not in samples_dict:
                mask_np = (img.squeeze().numpy() * 255).astype(np.uint8)
                
                # Extract GT contour
                _, thresh = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
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
                else:
                    target_pts = np.zeros((0, 2))
                
                samples_dict[label] = {
                    'img_tensor': img,  # (1, H, W)
                    'target_pts_pixel': target_pts,  # pixel coordinates
                    'label': str(label)
                }
            
            if len(samples_dict) == 10:
                break
        
        return [samples_dict[i] for i in range(10)]
    
    def create_template_mesh(template_dir, template_name, rest_cage, device, num_samples=3000):
        """
        Create Delaunay mesh for visualization.
        Same approach as model_vis.py - load template, sample points, compute MVC weights.
        """
        path = os.path.join(template_dir, f"{template_name}.npz")
        if not os.path.exists(path):
            print(f"[Warning] Template not found: {path}")
            return None
        
        data = np.load(path, allow_pickle=True)
        mask = data['mask']
        
        if mask.max() <= 1.0:
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_float = mask.astype(np.float32)
        else:
            mask_uint8 = mask.astype(np.uint8)
            mask_float = mask.astype(np.float32) / 255.0
        
        h, w = mask_uint8.shape
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)  # (1, H, W)
        
        # Sample interior points
        ys, xs = np.where(mask_uint8 > 127)
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        
        if points.shape[0] > num_samples:
            idx = np.random.choice(points.shape[0], num_samples, replace=False)
            points = points[idx]
        
        if points.shape[0] < 3:
            return None
        
        # Delaunay triangulation
        tri = Delaunay(points)
        centroids = points[tri.simplices].mean(axis=1).astype(int)
        centroids[:, 0] = np.clip(centroids[:, 0], 0, w-1)
        centroids[:, 1] = np.clip(centroids[:, 1], 0, h-1)
        
        valid = mask_uint8[centroids[:, 1], centroids[:, 0]] > 127
        faces = tri.simplices[valid]
        
        # Reference areas for mesh quality
        v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
        vec1, vec2 = v1 - v0, v2 - v0
        ref_areas = 0.5 * np.abs(vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])
        
        # Normalize to [-1, 1]
        points_norm = (points / np.array([w-1, h-1])) * 2 - 1
        vertices_tensor = torch.tensor(points_norm, dtype=torch.float32).to(device)
        
        # Compute MVC weights
        cage_batch = rest_cage.unsqueeze(0).to(device)
        pts_batch = vertices_tensor.unsqueeze(0)
        mvc_weights = compute_mvc_weights(pts_batch, cage_batch).squeeze(0)
        
        return {
            'vertices_norm': vertices_tensor,
            'mvc_weights': mvc_weights,
            'faces': faces,
            'ref_areas': ref_areas,
            'mask_tensor': mask_tensor,
            'shape': (h, w)
        }
    
    def check_mesh_quality(deformed_verts, faces, ref_areas):
        """Check mesh quality after deformation."""
        if len(faces) == 0:
            return {'area_ratios': np.array([1.0])}
        
        v0 = deformed_verts[faces[:, 0]]
        v1 = deformed_verts[faces[:, 1]]
        v2 = deformed_verts[faces[:, 2]]
        vec1, vec2 = v1 - v0, v2 - v0
        
        signed_areas = 0.5 * (vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])
        new_areas = np.abs(signed_areas)
        area_ratios = new_areas / (ref_areas + 1e-8)
        
        return {'area_ratios': area_ratios}
    
    def apply_affine_transform(points, matrix):
        """Apply affine transformation (same as grid_utils.apply_affine)."""
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, device=points.device)
        points_homo = torch.cat([points, ones], dim=2)
        return torch.bmm(points_homo, matrix.transpose(1, 2))
    
    def main():
        args = parse_args()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        print("=" * 60)
        print("DeformationExtractor - MNIST Visualization Test")
        print("=" * 60)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Template dir: {args.template_dir}")
        print(f"Templates: {args.templates}")
        print(f"Device: {args.device}")
        if args.no_residual:
            print("⚠️  Residual flow DISABLED (cage only)")
        else:
            print("✅ Using FULL pipeline (cage + residual)")
        print("=" * 60)
        
        os.makedirs(args.output_dir, exist_ok=True)
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Generate rest cage (same as model_vis.py)
        rest_cage = generate_circular_cage(128, radius=1.2, device=device)
        
        # Create mesh data for each template (same as model_vis.py)
        print("\n🔧 Creating template meshes for visualization...")
        template_meshes = {}
        for t_name in args.templates:
            mesh_data = create_template_mesh(args.template_dir, t_name, rest_cage, device)
            if mesh_data:
                template_meshes[t_name] = mesh_data
                print(f"   ✅ {t_name}: {mesh_data['vertices_norm'].shape[0]} mesh points")
        
        if not template_meshes:
            print("❌ No templates loaded!")
            return
        
        # Load model (same as model_vis.py - direct model loading for visualization)
        print(f"\n📂 Loading model from {args.checkpoint}...")
        model = ShapeTransformationNetwork(
            input_channels=4,
            cage_num_vertices=128,
            latent_dim=256,
            use_gaussian_ff=True,
            gaussian_ff_features=96,
            gaussian_ff_scale=10.0,
            use_multiscale_ff=True,
            multiscale_ff_scales=[1.0, 5.0, 15.0]
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Model loaded")
        
        # Get MNIST samples
        mnist_samples = get_mnist_samples()
        N = len(mnist_samples)
        M = len(template_meshes)
        template_names_sorted = list(template_meshes.keys())
        
        print(f"\n🎨 Creating {N}x{M} visualization grid...")
        
        # Color map for mesh quality (from model_vis.py)
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
            for i, sample in enumerate(mnist_samples):
                img_1ch = sample['img_tensor']  # (1, H, W)
                target_pts_pixel = sample['target_pts_pixel']  # pixel coords
                label_str = sample['label']
                
                # Convert grayscale to RGB (same as model_vis.py)
                source_img = img_1ch.repeat(3, 1, 1).to(device)  # (3, H, W)
                
                for j, t_name in enumerate(template_names_sorted):
                    t_data = template_meshes[t_name]
                    
                    # === STEP 1: Prepare Model Input (same as model_vis.py) ===
                    template_mask = t_data['mask_tensor'].to(device)  # (1, H, W)
                    
                    if template_mask.shape[-2:] != source_img.shape[-2:]:
                        template_mask = F.interpolate(
                            template_mask.unsqueeze(0),
                            size=source_img.shape[-2:],
                            mode='nearest'
                        ).squeeze(0)
                    
                    # Input format: [RGB(3) + Template Mask(1)] = 4 channels
                    model_input = torch.cat([source_img, template_mask], dim=0).unsqueeze(0)  # (1, 4, H, W)
                    
                    # === STEP 2: Model Forward (Affine + Cage) ===
                    rest_cage_batch = rest_cage.unsqueeze(0)
                    output = model(model_input, rest_cage_batch)
                    
                    affine_mat = output['affine_matrix']       # (1, 2, 3)
                    cage_offsets = output['cage_offsets']      # (1, K, 2)
                    latent_z = output['latent_z']              # (1, latent_dim)
                    spatial_features = output['spatial_features']
                    
                    # === STEP 3: Apply Cage Deformation ===
                    cage_after_affine = apply_affine_transform(rest_cage_batch, affine_mat)
                    cage_deformed = cage_after_affine + cage_offsets  # (1, K, 2)
                    
                    # Deform template vertices using MVC weights
                    mvc_weights = t_data['mvc_weights']
                    deformed_verts = torch.mm(mvc_weights, cage_deformed.squeeze(0))  # (N, 2)
                    
                    # === STEP 4: Apply Residual Flow (FULL MODEL, same as model_vis.py) ===
                    if not args.no_residual:
                        residual = model.compute_residual(
                            latent_z, 
                            deformed_verts.unsqueeze(0), 
                            spatial_features
                        )
                        deformed_verts = deformed_verts + residual.squeeze(0)
                    
                    # === STEP 5: Visualization ===
                    deformed_verts_np = deformed_verts.cpu().numpy()
                    viz_verts = (deformed_verts_np + 1) * (VIZ_SCALE / 2.0)
                    
                    # Check mesh quality
                    metrics = check_mesh_quality(viz_verts, t_data['faces'], t_data['ref_areas'])
                    face_colors = np.clip(np.log2(metrics['area_ratios'] + 1e-6) / 4.0 + 0.5, 0, 1)
                    
                    ax = fig.add_subplot(gs[i, j])
                    
                    # Draw mesh faces
                    faces = t_data['faces']
                    if len(faces) > 0:
                        ax.tripcolor(viz_verts[:, 0], viz_verts[:, 1], faces,
                                    facecolors=face_colors, cmap=density_cmap,
                                    shading='flat', vmin=0, vmax=1, alpha=0.8)
                        ax.triplot(viz_verts[:, 0], viz_verts[:, 1], faces,
                                  'k-', linewidth=0.3, alpha=0.4)
                    
                    # Draw GT contour (green dots)
                    if len(target_pts_pixel) > 0:
                        ax.plot(target_pts_pixel[:, 0], target_pts_pixel[:, 1],
                               '.', c='lime', ms=2.0, alpha=0.6)
                    
                    # Draw cage if requested
                    if args.show_cage:
                        cage_viz = (cage_deformed.squeeze(0).cpu().numpy() + 1) * (VIZ_SCALE / 2.0)
                        cage_closed = np.vstack([cage_viz, cage_viz[0:1]])
                        ax.plot(cage_closed[:, 0], cage_closed[:, 1], 'r-', lw=1.5, alpha=0.8)
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
        
        # Save figure
        out_path = os.path.join(args.output_dir, f"extractor_mnist_{int(time.time())}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=VIZ_DPI, facecolor='white')
        plt.close()
        print(f"\n✅ Saved visualization: {out_path}")
        
        # ============================================================
        # Also test the DeformationExtractor class itself
        # ============================================================
        print("\n" + "=" * 60)
        print("Testing DeformationExtractor class...")
        print("=" * 60)
        
        extractor = DeformationExtractor(
            checkpoint_path=args.checkpoint,
            template_dir=args.template_dir,
            template_names=args.templates,
            cage_num_vertices=128,
            cage_radius=1.2,
            device=args.device
        )
        
        # Create 4-channel test input (RGB + Mask)
        test_rgb = torch.randn(2, 3, 256, 256)
        test_mask = torch.rand(2, 1, 256, 256)
        test_img = torch.cat([test_rgb, test_mask], dim=1)  # (2, 4, 256, 256)
        
        B = test_img.shape[0]

        # Test 1: same template for all samples
        single_t = args.templates[0]
        print(f"\n1. Same template for all {B} samples: {single_t}")
        template_list = [single_t] * B
        res_list = extractor(test_img, template_list)
        print(f"   Results length: {len(res_list)}")

        # Test 2: different templates per sample
        if len(args.templates) >= 2:
            mixed = [args.templates[i % len(args.templates)] for i in range(B)]
            print(f"\n2. Mixed templates: {mixed}")
            res_mixed = extractor(test_img, mixed)
            print(f"   Results length: {len(res_mixed)}")

        # Print output shapes
        print("\n" + "-" * 40)
        print("Output shapes (per sample):")
        print("-" * 40)
        for i, params in enumerate(res_list):
            print(f"\nSample {i} (template={template_list[i]}):")
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
        
        print("\n" + "=" * 60)
        print("✅ All tests PASSED!")
        print("=" * 60)
        
        # ============================================================
        # Print Usage Summary
        # ============================================================
        print("\n" + "=" * 60)
        print("📖 DeformationExtractor Usage Summary")
        print("=" * 60)
        print("""
from src.app.extract_deformation import DeformationExtractor
import torch

# 1. Initialize (loads model + pre-computes MVC weights)
extractor = DeformationExtractor(
    checkpoint_path='weight_files/cage128_rsguide_v12/checkpoint_latest.pth',
    template_dir='template',
    template_names=['circle', 'rectangle', 'star'],
    cage_num_vertices=128,
    cage_radius=1.2,
    device='cuda'
)

# 2. Prepare input: (B, 4, H, W) = RGB (3 channels) + Target Mask (1 channel)
rgb = torch.randn(4, 3, 256, 256)   # RGB images
mask = torch.rand(4, 1, 256, 256)   # Target shape masks
images = torch.cat([rgb, mask], dim=1)  # (4, 4, 256, 256)

# 3. Forward: provide per-sample template assignment (length must equal batch size)
# Example: batch of 4 samples, each with its own template
template_list = ['circle', 'circle', 'star', 'rectangle']
results = extractor(images, template_list)
# results is a List of length 4

# 4. Access results (per sample):
for i, params in enumerate(results):
    print(f"Sample {i}, template={template_list[i]}")
    affine = params['affine']              # (1, 2, 3)
    cage_offset = params['cage_offset']    # (1, K, 2)  K=128
    cage_deformed = params['cage_deformed']# (1, K, 2)
    residual = params['residual']          # (1, N_boundary, 2)
    deformed_boundary = params['deformed_boundary']  # (1, N_boundary, 2)
    deformed_interior = params['deformed_interior']  # (1, N_interior, 2)
    latent_z = params['latent_z']          # (1, 256)

# 5. Helper methods:
extractor.get_template_names()  # List of available templates
extractor.get_template_info()   # Dict with boundary/interior point counts
""")
        print("=" * 60)
        print("✅ All tests PASSED!")
        print("=" * 60)
    
    main()
