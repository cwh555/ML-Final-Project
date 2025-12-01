import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np

class SpatialAttentionModule(nn.Module):
    def __init__(self, feature_dim=512, cage_resolution=32):
        super().__init__()
        self.cage_res = cage_resolution
        self.num_points = cage_resolution * cage_resolution
        
        self.position_mlp = nn.Sequential(
            nn.Linear(4 + feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.register_buffer('position_encoding', self._create_position_encoding())
        
    def _create_position_encoding(self):
        x = torch.linspace(-1, 1, self.cage_res)
        y = torch.linspace(-1, 1, self.cage_res)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        pos_enc = torch.stack([
            torch.sin(grid_x.flatten() * np.pi),
            torch.cos(grid_x.flatten() * np.pi),
            torch.sin(grid_y.flatten() * np.pi),
            torch.cos(grid_y.flatten() * np.pi)
        ], dim=1)
        return pos_enc
    
    def forward(self, global_features):
        B = global_features.shape[0]
        N = self.num_points
        pos_enc = self.position_encoding.unsqueeze(0).expand(B, -1, -1)
        global_feat_expanded = global_features.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([pos_enc, global_feat_expanded], dim=2)
        attention_logits = self.position_mlp(combined)
        # [Fix] Use Sigmoid for independent gating
        spatial_weights = torch.sigmoid(attention_logits)
        return spatial_weights

class ShapeTransformationNetwork(nn.Module):
    def __init__(self, input_channels=4, cage_resolution=32, coarse_grid_res=8):
        super(ShapeTransformationNetwork, self).__init__()
        self.cage_resolution = cage_resolution
        self.num_control_points = cage_resolution * cage_resolution
        self.coarse_grid_res = coarse_grid_res
        
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.spatial_attention = SpatialAttentionModule(feature_dim=self.feature_dim, cage_resolution=cage_resolution)
        
        self.fc_global = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Linear(256, 6))
        
        self.fc_local = nn.Sequential(
            nn.Linear(self.feature_dim + 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        # 8x8 Coarse Grid Head
        self.fc_coarse = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.coarse_grid_res * self.coarse_grid_res * 2)
        )
        
        self._init_heads()

    def _init_heads(self):
        nn.init.zeros_(self.fc_global[-1].weight)
        self.fc_global[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        nn.init.zeros_(self.fc_local[-1].weight)
        nn.init.zeros_(self.fc_local[-1].bias)
        nn.init.zeros_(self.fc_coarse[-1].weight)
        nn.init.zeros_(self.fc_coarse[-1].bias)

    def forward(self, x, test=False):
        """
        Args:
            x: Input images (B, C, H, W)
            test: 如果為 True，回傳包含特徵的字典；如果為 False (預設)，回傳訓練用的 Tuple。
        """
        B = x.shape[0]
        features = self.backbone(x)
        spatial_weights = self.spatial_attention(features)
        
        affine_flat = self.fc_global(features)
        affine_matrix = affine_flat.view(-1, 2, 3)
        
        # --- Fine Grid ---
        pos_enc = self.spatial_attention.position_encoding.unsqueeze(0).expand(B, -1, -1)
        global_feat_expanded = features.unsqueeze(1).expand(B, self.num_control_points, -1)
        point_features = torch.cat([global_feat_expanded, pos_enc], dim=2)
        
        # [Feature 1] Fine Grid Raw Output (B, 1024, 2)
        fine_offsets_raw = self.fc_local(point_features)
        
        # --- Coarse Grid ---
        coarse_flat = self.fc_coarse(features)
        
        # [Feature 2] Coarse Grid Raw Output (B, 8, 8, 2)
        # 這是你要的 8x8 骨架特徵
        coarse_offset_8x8 = coarse_flat.view(B, self.coarse_grid_res, self.coarse_grid_res, 2)
        
        # 準備插值：轉為 (B, 2, 8, 8)
        coarse_offset_for_interp = coarse_offset_8x8.permute(0, 3, 1, 2)
        
        # 雙三次插值放大到 32x32
        coarse_deformation = F.interpolate(
            coarse_offset_for_interp, 
            size=(self.cage_resolution, self.cage_resolution), 
            mode='bicubic', 
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(B, -1, 2)
        
        # 融合
        fine_offsets = fine_offsets_raw * spatial_weights * 0.5
        total_offsets = coarse_deformation + fine_offsets
        
        # === 分流邏輯 ===
        if test:
            # 下游任務模式：回傳特徵字典
            return {
                "affine": affine_matrix,
                
                # [你要的特徵 1] 8x8 骨架 (B, 8, 8, 2)
                "coarse_feat": coarse_offset_8x8, 
                
                # [你要的特徵 2] 32x32 細節 (B, 32, 32, 2)
                "fine_feat": fine_offsets_raw.view(B, self.cage_resolution, self.cage_resolution, 2),
                
                "final_grid": total_offsets,    # 最終結果 (畫圖用)
                "attn_map": spatial_weights     # Attention map
            }
        
        # 訓練模式：維持原狀 (B, 1024, 2)
        return affine_matrix, total_offsets, spatial_weights, coarse_deformation

