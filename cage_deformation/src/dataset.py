"""
Cage Deformation Dataset
Handles:
- Template boundary/interior point sampling
- Pre-compute MVC weights for efficiency
- Signed SDF computation for target shapes
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from scipy.ndimage import distance_transform_edt
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import src.utils.grid_utils as utils


def create_split(dataset_dir, split_file, val_ratio=0.1):
    """Create train/val split file."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {dataset_dir}")
    
    df = pd.DataFrame({'filename': files})
    val_size = int(len(df) * val_ratio)
    train_size = len(df) - val_size
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.loc[:train_size-1, 'split'] = 'train'
    df.loc[train_size:, 'split'] = 'val'
    
    df.to_csv(split_file, index=False)
    print(f"✅ Created split file: {split_file} (Train: {train_size}, Val: {val_size})")


class CageDataset(Dataset):
    def __init__(self, split_file, dataset_dir, template_dir, template_names, 
                 split='train', num_points=4096, cage_num_vertices=96, cage_radius=1.2,
                 num_target_boundary=512, ordered_boundary=True):
        """
        Args:
            split_file: CSV file with train/val split
            dataset_dir: Directory with .npy files (each contains [img, mask])
            template_dir: Directory with template .npz files
            template_names: List of template names (without .npz)
            split: 'train' or 'val'
            num_points: Total number of points to sample (split 50/50 boundary/interior)
            cage_num_vertices: Number of cage vertices (K)
            cage_radius: Radius of rest cage
            num_target_boundary: Number of target boundary points for Suction loss
            ordered_boundary: If True, sample boundary points in contour order (for Sliding loss)
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.num_boundary = num_points // 2
        self.num_interior = num_points - self.num_boundary
        self.num_target_boundary = num_target_boundary
        self.ordered_boundary = ordered_boundary
        
        # Generate rest cage (shared across all templates)
        self.rest_cage = utils.generate_circular_cage(
            cage_num_vertices, radius=cage_radius, device='cpu'
        )
        cage_batch = self.rest_cage.unsqueeze(0)  # (1, K, 2)
        
        # Load templates and pre-compute MVC weights
        self.templates = []
        print(f"[{split}] Loading templates and pre-computing MVC weights...")
        
        for name in template_names:
            path = os.path.join(template_dir, f"{name}.npz")
            if not os.path.exists(path):
                print(f"⚠️  Template not found: {path}, skipping...")
                continue
            
            data = np.load(path)
            mask = data['mask']
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            
            h, w = mask.shape
            
            # === A. Sample Boundary Points ===
            # CRITICAL: Use CHAIN_APPROX_NONE to get ALL boundary points!
            # CHAIN_APPROX_SIMPLE compresses contours and loses information
            # (e.g., rectangle would only have 4 corner points)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                print(f"⚠️  No contours found in {name}, skipping...")
                continue
            
            # Get all boundary points
            b_pts_raw = cnts[0].squeeze().reshape(-1, 2).astype(np.float32)
            
            # Normalize to [-1, 1]
            b_pts_norm = (b_pts_raw / np.array([w-1, h-1])) * 2 - 1
            b_pts_tensor = torch.tensor(b_pts_norm, dtype=torch.float32)
            
            # Compute MVC weights for boundary (in chunks to save memory)
            print(f"  Computing MVC weights for {name} boundary ({len(b_pts_tensor)} points)...")
            w_b_list = []
            chunk_size = 5000
            for i in range(0, len(b_pts_tensor), chunk_size):
                chunk = b_pts_tensor[i:i+chunk_size].unsqueeze(0)
                w_b_list.append(utils.compute_mvc_weights(chunk, cage_batch).squeeze(0))
            weights_boundary = torch.cat(w_b_list, dim=0)
            
            # === B. Sample Interior Points ===
            ys, xs = np.where(mask > 127)
            i_pts_raw = np.stack([xs, ys], axis=1).astype(np.float32)
            i_pts_norm = (i_pts_raw / np.array([w-1, h-1])) * 2 - 1
            i_pts_tensor = torch.tensor(i_pts_norm, dtype=torch.float32)
            
            # Compute MVC weights for interior
            print(f"  Computing MVC weights for {name} interior ({len(i_pts_tensor)} points)...")
            w_i_list = []
            for i in range(0, len(i_pts_tensor), chunk_size):
                chunk = i_pts_tensor[i:i+chunk_size].unsqueeze(0)
                w_i_list.append(utils.compute_mvc_weights(chunk, cage_batch).squeeze(0))
            weights_interior = torch.cat(w_i_list, dim=0)
            
            self.templates.append({
                'name': name,
                'pts_boundary': b_pts_tensor,
                'weights_boundary': weights_boundary,
                'pts_interior': i_pts_tensor,
                'weights_interior': weights_interior,
                'mask': torch.tensor(mask > 127, dtype=torch.float32).unsqueeze(0)
            })
        
        if not self.templates:
            raise ValueError("No valid templates loaded!")
        
        # Load split
        if not os.path.exists(split_file):
            create_split(dataset_dir, split_file)
        
        df = pd.read_csv(split_file)
        self.file_names = df[df['split'] == split]['filename'].tolist()
        print(f"✅ Loaded {len(self.file_names)} {split} samples")

    def compute_signed_sdf(self, mask):
        """
        Compute signed distance field based on OUTER CONTOUR ONLY.
        
        CRITICAL: If target has holes, we don't want template boundary 
        to fit to the hole edges! So we fill holes before computing SDF.
        
        This ensures SDF=0 only on the outer boundary, not on hole edges.
        
        Positive: outside, Negative: inside
        """
        # Step 1: Extract only the external contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Step 2: Create a mask with holes filled (draw filled contour)
        mask_filled = np.zeros_like(mask)
        if contours:
            cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)
        
        # Step 3: Compute SDF on the hole-filled mask
        # Now SDF=0 will only be on the outer boundary
        dist_out = distance_transform_edt(mask_filled == 0)
        dist_in = distance_transform_edt(mask_filled > 127)
        sdf = (dist_out - dist_in) / 256.0
        return torch.tensor(sdf, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        fpath = os.path.join(self.dataset_dir, fname)
        
        try:
            # Load target shape
            data = np.load(fpath, allow_pickle=True)
            tgt_img = torch.tensor(data[0], dtype=torch.float32)
            if tgt_img.max() > 1.0:
                tgt_img /= 255.0
            
            tgt_mask = data[1]
            if tgt_mask.max() <= 1.0:
                tgt_mask = (tgt_mask * 255).astype(np.uint8)
            else:
                tgt_mask = tgt_mask.astype(np.uint8)
            
            tgt_sdf = self.compute_signed_sdf(tgt_mask)
            
            # ================================================================
            # [SUCTION] Extract Target boundary points for reverse Chamfer
            # Reference: Fan et al. (CVPR 2017) - Bidirectional Chamfer
            # ================================================================
            h, w = tgt_mask.shape
            contours, _ = cv2.findContours(tgt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours and len(contours[0]) > 0:
                tgt_boundary_raw = contours[0].squeeze().reshape(-1, 2).astype(np.float32)
                # Normalize to [-1, 1]
                tgt_boundary_norm = (tgt_boundary_raw / np.array([w-1, h-1])) * 2 - 1
                
                # Sample target boundary points (for Suction loss)
                total_tgt = len(tgt_boundary_norm)
                if total_tgt > self.num_target_boundary:
                    # Random sample from target boundary
                    tgt_idx = np.random.choice(total_tgt, self.num_target_boundary, replace=False)
                    tgt_pts = tgt_boundary_norm[tgt_idx]
                else:
                    # Repeat if not enough points
                    tgt_idx = np.random.choice(total_tgt, self.num_target_boundary, replace=True)
                    tgt_pts = tgt_boundary_norm[tgt_idx]
                pts_target_boundary = torch.tensor(tgt_pts, dtype=torch.float32)
            else:
                # Fallback: zeros if no contour found
                pts_target_boundary = torch.zeros(self.num_target_boundary, 2)
            
            # Random template
            t_idx = np.random.randint(len(self.templates))
            t_data = self.templates[t_idx]
            
            # ================================================================
            # [SLIDING] Sample boundary points - ORDERED for uniform edge loss
            # Reference: Kass et al. (IJCV 1988) - Active Contours (Snakes)
            # ================================================================
            total_b = len(t_data['pts_boundary'])
            if self.ordered_boundary:
                # ORDERED sampling: maintain contour order for edge length computation
                # Random start point, then take consecutive points
                start_idx = np.random.randint(0, total_b)
                
                if total_b >= self.num_boundary:
                    # Uniform stride to cover entire contour
                    stride = total_b / self.num_boundary
                    idx_b = np.array([int((start_idx + i * stride) % total_b) for i in range(self.num_boundary)])
                else:
                    # Not enough points: repeat with interpolation
                    idx_b = np.array([int((start_idx + i) % total_b) for i in range(self.num_boundary)])
                idx_b = torch.tensor(idx_b, dtype=torch.long)
            else:
                # RANDOM sampling (original behavior)
                if total_b > self.num_boundary:
                    idx_b = torch.randperm(total_b)[:self.num_boundary]
                else:
                    idx_b = torch.randint(0, total_b, (self.num_boundary,))
            
            sample_pts_b = t_data['pts_boundary'][idx_b]
            sample_w_b = t_data['weights_boundary'][idx_b]
            
            # Sample interior points (random is fine)
            total_i = len(t_data['pts_interior'])
            if total_i > self.num_interior:
                idx_i = torch.randperm(total_i)[:self.num_interior]
            else:
                idx_i = torch.randint(0, total_i, (self.num_interior,))
            
            sample_pts_i = t_data['pts_interior'][idx_i]
            sample_w_i = t_data['weights_interior'][idx_i]
            
            # Stack input: [RGB, Template Mask]
            input_stack = torch.cat([tgt_img, t_data['mask']], dim=0)
            
            return {
                'input': input_stack,
                'pts_boundary': sample_pts_b,
                'weights_boundary': sample_w_b,
                'pts_interior': sample_pts_i,
                'weights_interior': sample_w_i,
                'target_sdf': tgt_sdf,
                'template_idx': torch.tensor(t_idx),
                'pts_target_boundary': pts_target_boundary,  # For Suction loss
                'boundary_is_ordered': torch.tensor(self.ordered_boundary)  # For Sliding loss
            }
        
        except Exception as e:
            print(f"⚠️  Error loading {fname}: {e}")
            # Return next sample on error
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.file_names)