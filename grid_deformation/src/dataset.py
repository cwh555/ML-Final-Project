import os
import torch
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from typing import List, Dict

class CageDataset(Dataset):
    def __init__(self, split_file, dataset_dir, template_dir, template_names: List[str], split='train', num_points=1024):
        self.dataset_dir = dataset_dir
        self.num_points = num_points
        # Split points: 50% Boundary (for Alignment), 50% Interior (for SDF pushing)
        self.num_boundary = num_points // 2
        self.num_interior = num_points - self.num_boundary
        
        self.templates: List[Dict] = []
        
        # 1. Load Templates
        for name in template_names:
            path = os.path.join(template_dir, f"{name}.npz")
            if not os.path.exists(path): continue
            
            data = np.load(path, allow_pickle=True)
            mask = data['mask'] 
            
            # [Fix] Robust Normalization (0/1 -> 0/255)
            if mask.max() <= 1.0:
                mask_u8 = (mask * 255).astype(np.uint8)
            else:
                mask_u8 = mask.astype(np.uint8)
                
            mask_float = (mask_u8 > 127).astype(np.float32)

            # A. Sample Boundary Points - [修正] 只采样外轮廓
            # 使用 RETR_EXTERNAL 确保与 Target 采样逻辑一致
            # Chamfer Loss 只管外壳对齐，内部孔洞由 Topology Loss 处理
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts: continue
            
            all_cnt_pts = []
            for c in cnts:
                all_cnt_pts.append(c.squeeze().reshape(-1, 2))
            b_pts = np.concatenate(all_cnt_pts, axis=0)
            
            # Normalize Boundary
            h, w = mask.shape
            b_pts_norm = (b_pts / np.array([w-1, h-1])) * 2 - 1 
            
            # Sample Boundary
            if b_pts_norm.shape[0] > 0:
                replace = b_pts_norm.shape[0] < self.num_boundary
                choice = np.random.choice(b_pts_norm.shape[0], self.num_boundary, replace=replace)
                b_pts_final = torch.tensor(b_pts_norm[choice], dtype=torch.float32)
            else:
                b_pts_final = torch.zeros((self.num_boundary, 2), dtype=torch.float32)

            # B. Sample Interior Points (mask > 0)
            ys, xs = np.where(mask_u8 > 127)
            if len(xs) > 0:
                i_pts = np.stack([xs, ys], axis=1) # (N, 2)
                i_pts_norm = (i_pts / np.array([w-1, h-1])) * 2 - 1
                
                replace = i_pts_norm.shape[0] < self.num_interior
                choice = np.random.choice(i_pts_norm.shape[0], self.num_interior, replace=replace)
                i_pts_final = torch.tensor(i_pts_norm[choice], dtype=torch.float32)
            else:
                i_pts_final = torch.zeros((self.num_interior, 2), dtype=torch.float32)

            # C. Combine
            combined_pts = torch.cat([b_pts_final, i_pts_final], dim=0)
            
            self.templates.append({
                'name': name,
                'points': combined_pts, # (1024, 2)
                'mask': torch.tensor(mask_float, dtype=torch.float32).unsqueeze(0) 
            })

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file missing: {split_file}")
        
        df = pd.read_csv(split_file)
        self.file_names = df[df['split'] == split]['filename'].tolist()

    def compute_sdf(self, mask_uint8):
        bg_mask = mask_uint8 < 127 
        if bg_mask.all():
             return torch.ones((1, mask_uint8.shape[0], mask_uint8.shape[1]), dtype=torch.float32)
        dist_map = distance_transform_edt(bg_mask)
        sdf = dist_map / 256.0 
        return torch.tensor(sdf, dtype=torch.float32).unsqueeze(0) 

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        fpath = os.path.join(self.dataset_dir, fname)
        
        try:
            data = np.load(fpath, allow_pickle=True)
            
            tgt_img = torch.tensor(data[0], dtype=torch.float32)
            if tgt_img.max() > 1.0:
                tgt_img = tgt_img / 255.0
                
            tgt_mask_raw = data[1]
            if tgt_mask_raw.max() <= 1.0:
                tgt_mask_u8 = (tgt_mask_raw * 255).astype(np.uint8)
            else:
                tgt_mask_u8 = tgt_mask_raw.astype(np.uint8)
            
            tgt_sdf = self.compute_sdf(tgt_mask_u8)
            
            # Target Points: Only Boundary (for Chamfer)
            # [修正] 均匀采样外轮廓 - 整个 simple closed curve 都是外轮廓，包括U型凹陷
            cnts, _ = cv2.findContours(tgt_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                # RETR_EXTERNAL 只返回最外层轮廓（simple closed curve）
                all_pts = []
                for c in cnts:
                    c_pts = c.squeeze().reshape(-1, 2)
                    if len(c_pts) >= 3:
                        all_pts.append(c_pts)
                
                if all_pts:
                    pts = np.concatenate(all_pts, axis=0)
                    h, w = tgt_mask_u8.shape
                    pts = (pts / np.array([w-1, h-1])) * 2 - 1
                    
                    target_sample_size = self.num_boundary
                    # 均匀随机采样，不使用曲率权重
                    replace = pts.shape[0] < target_sample_size
                    choice = np.random.choice(pts.shape[0], target_sample_size, replace=replace)
                    tgt_pts = torch.tensor(pts[choice], dtype=torch.float32)
                else:
                    tgt_pts = torch.zeros((self.num_boundary, 2), dtype=torch.float32)
            else:
                tgt_pts = torch.zeros((self.num_boundary, 2), dtype=torch.float32)
            
            t_idx = np.random.randint(0, len(self.templates))
            tmp_mask = self.templates[t_idx]['mask'] 
            
            input_stack = torch.cat([tgt_img, tmp_mask], dim=0)

            return {
                'input': input_stack,
                'target_points': tgt_pts,
                'target_sdf': tgt_sdf,
                'template_idx': torch.tensor(t_idx, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            raise e

def create_split(dataset_dir, split_file, val_ratio=0.1):
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
    df = pd.DataFrame({'filename': files})
    train_df = df.sample(frac=1-val_ratio, random_state=42)
    val_df = df.drop(train_df.index)
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    full_df = pd.concat([train_df, val_df])
    full_df.to_csv(split_file, index=False)

