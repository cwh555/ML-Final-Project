"""
Shape Transformation Network - Hierarchical Cage-Residual Architecture
Based on:
- Neural Cages (CVPR 2020): Cage-based deformation with MVC
- Deforming Autoencoders (ECCV 2018): Global-Local decomposition

Three-stage deformation:
1. Global Affine (coarse alignment) -> Uses Global Pooling
2. Local Cage Offsets (skeleton warping) -> Uses Feature Sampling + Circular Conv
3. Residual Flow (fine detail) -> Per-point MLP for boundary refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class CircularConv1d(nn.Module):
    """Helper to perform convolution on circular topology (Cage)"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # x: (B, C, K)
        # Circular padding to maintain topology
        x_pad = F.pad(x, (self.pad, self.pad), mode='circular')
        return self.conv(x_pad)


class GaussianFourierFeatureTransform(nn.Module):
    """
    Gaussian Random Fourier Features for high-frequency function learning.
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn High 
    Frequency Functions in Low Dimensional Domains", NeurIPS 2020
    
    Key insight: Standard MLPs have spectral bias - they cannot learn
    high-frequency signals. Random Fourier features project inputs to
    a higher-dimensional space where high frequencies become learnable.
    
    Formula: γ(x) = [sin(2πBx), cos(2πBx)]
    where B is a (num_features, input_dim) matrix with entries ~ N(0, σ²)
    
    Scale σ controls frequency coverage:
    - σ = 1: standard frequencies
    - σ = 10: captures fine details (recommended for geometric shapes)
    - σ = 100: very high frequencies (may overfit)
    """
    def __init__(self, input_dim=2, num_features=64, scale=10.0):
        super().__init__()
        # Random frequency matrix B ~ N(0, scale²)
        # Fixed after initialization (not learned)
        B = torch.randn(num_features, input_dim) * scale
        self.register_buffer('B', B)
        self.output_dim = num_features * 2  # sin + cos
    
    def forward(self, x):
        """
        Args:
            x: (..., input_dim) - input coordinates
        Returns:
            features: (..., num_features * 2) - Fourier features
        """
        # x @ B.T: (..., num_features)
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MultiScaleGaussianFourierFeatures(nn.Module):
    """
    Multi-Scale Gaussian Fourier Features for balanced frequency learning.
    
    Reference: Tancik et al. (NeurIPS 2020) - suggests tunable bandwidth
    
    Key insight: Single-scale Fourier features may either:
    - Miss fine details (scale too low)
    - Cause high-frequency noise/bunching (scale too high)
    
    Multi-scale approach combines:
    - Low frequencies (scale=1.0): Global structure, smooth deformations
    - Mid frequencies (scale=5.0): Medium details, curves
    - High frequencies (scale=15.0): Fine details, sharp corners
    
    This prevents the "bunching" problem where points cluster at concavity
    edges due to over-emphasis on high-frequency signals.
    """
    def __init__(self, input_dim=2, num_features_per_scale=32, scales=[1.0, 5.0, 15.0]):
        super().__init__()
        self.scales = scales
        self.num_features_per_scale = num_features_per_scale
        
        # Create separate random frequency matrices for each scale
        # Each matrix is (num_features_per_scale, input_dim)
        for i, scale in enumerate(scales):
            B = torch.randn(num_features_per_scale, input_dim) * scale
            self.register_buffer(f'B_{i}', B)
        
        # Total output dimension: num_scales * num_features * 2 (sin + cos)
        self.output_dim = len(scales) * num_features_per_scale * 2
    
    def forward(self, x):
        """
        Args:
            x: (..., input_dim) - input coordinates
        Returns:
            features: (..., output_dim) - multi-scale Fourier features
        """
        features = []
        for i in range(len(self.scales)):
            B = getattr(self, f'B_{i}')
            x_proj = 2 * np.pi * torch.matmul(x, B.T)
            features.append(torch.sin(x_proj))
            features.append(torch.cos(x_proj))
        return torch.cat(features, dim=-1)


class ResidualFlowHead(nn.Module):
    """
    Residual Flow Head for boundary point refinement.
    Takes latent vector z and predicts per-point residual displacement.
    
    Design principle (Deforming Autoencoders + NeRF + Tancik et al.):
    - Uses HYBRID Fourier encoding: deterministic (NeRF-style) + random Gaussian
    - Deterministic: captures structured frequencies (2^i)
    - Random Gaussian: covers continuous spectrum for fine details
    - Initialized near-zero to let Cage learn first
    
    v9 Update: Multi-Scale Fourier Features
    - Prevents "bunching" by balancing low/mid/high frequencies
    - Low (1.0): global structure
    - Mid (5.0): medium details  
    - High (15.0): sharp corners (reduced from 30.0 to prevent noise)
    
    References:
    - NeRF (Mildenhall et al., ECCV 2020) - positional encoding
    - Tancik et al., NeurIPS 2020 - Gaussian Fourier features
    - Kass et al., IJCV 1988 - Active Contours (anti-bunching theory)
    """
    def __init__(self, latent_dim=256, feature_dim=512, hidden_dim=128, num_freq=6, 
                 use_gaussian_ff=True, gaussian_ff_features=64, gaussian_ff_scale=10.0,
                 use_multiscale_ff=True, multiscale_ff_scales=[1.0, 5.0, 15.0]):
        super().__init__()
        self.num_freq = num_freq
        self.use_gaussian_ff = use_gaussian_ff
        self.use_multiscale_ff = use_multiscale_ff
        
        # Deterministic positional encoding dimension: 2 (xy) + 2 * 2 * num_freq
        pos_dim_deterministic = 2 + 2 * 2 * num_freq  # = 26 for num_freq=6
        
        # Gaussian Fourier features (Tancik et al., NeurIPS 2020)
        if use_gaussian_ff:
            if use_multiscale_ff:
                # Multi-scale Fourier: prevents bunching by balanced frequency coverage
                # Total features = len(scales) * features_per_scale * 2
                features_per_scale = gaussian_ff_features // len(multiscale_ff_scales)
                self.gaussian_ff = MultiScaleGaussianFourierFeatures(
                    input_dim=2,
                    num_features_per_scale=features_per_scale,
                    scales=multiscale_ff_scales
                )
            else:
                # Single-scale Fourier (original)
                self.gaussian_ff = GaussianFourierFeatureTransform(
                    input_dim=2, 
                    num_features=gaussian_ff_features, 
                    scale=gaussian_ff_scale
                )
            pos_dim = pos_dim_deterministic + self.gaussian_ff.output_dim
        else:
            self.gaussian_ff = None
            pos_dim = pos_dim_deterministic
        
        # MLP that takes (latent_z, hybrid_encoded_coord)
        # Larger hidden dim to handle increased input dimension
        input_total_dim = latent_dim + pos_dim + feature_dim
        mlp_hidden = hidden_dim * 2 if use_gaussian_ff else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_total_dim, mlp_hidden),
            nn.ReLU(True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(True),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2)  # Output: (dx, dy)
        )
        self._init_weights()
    
    def _init_weights(self):
        # Initialize to near-zero output (Cage does most work initially)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.zeros_(m.bias)
    
    def positional_encoding(self, coords):
        """
        HYBRID Fourier positional encoding for coordinates.
        Combines deterministic (NeRF-style) and random Gaussian features.
        
        This enables learning high-frequency functions (deep concavities, 
        sharp corners) by covering both structured and continuous frequency spectra.
        
        Args:
            coords: (B, N, 2) - coordinates in [-1, 1]
        Returns:
            encoded: (B, N, D) where D = 2 + 4*num_freq [+ gaussian_ff_dim]
        """
        # Part 1: Deterministic positional encoding (NeRF-style)
        # Captures structured frequencies: 2^0, 2^1, ..., 2^(num_freq-1)
        encoded = [coords]
        for i in range(self.num_freq):
            freq = 2 ** i * np.pi
            encoded.append(torch.sin(freq * coords))
            encoded.append(torch.cos(freq * coords))
        deterministic = torch.cat(encoded, dim=-1)
        
        # Part 2: Gaussian Random Fourier Features (Tancik et al., NeurIPS 2020)
        # Covers continuous spectrum for fine geometric details
        if self.use_gaussian_ff and self.gaussian_ff is not None:
            gaussian_features = self.gaussian_ff(coords)
            return torch.cat([deterministic, gaussian_features], dim=-1)
        
        return deterministic
    
    def forward(self, latent_z, points, sampled_features):
        """
        Args:
            latent_z: (B, latent_dim) - Global shape latent
            points: (B, N, 2) - Points to compute residual for (after cage deformation)
        Returns:
            residual: (B, N, 2) - Per-point residual displacement
        """
        B, N, _ = points.shape
        
        # Positional encoding of coordinates
        pos_encoded = self.positional_encoding(points)  # (B, N, 2+4*num_freq)
        
        # Expand latent to each point
        z_expanded = latent_z.unsqueeze(1).expand(-1, N, -1)
        
        # Concatenate
        mlp_input = torch.cat([z_expanded, pos_encoded, sampled_features], dim=-1)
        
        # Predict residual
        residual = self.mlp(mlp_input)
        return residual

class ShapeTransformationNetwork(nn.Module):
    def __init__(self, input_channels=4, cage_num_vertices=96, latent_dim=256,
                 use_gaussian_ff=True, gaussian_ff_features=64, gaussian_ff_scale=10.0,
                 use_multiscale_ff=True, multiscale_ff_scales=[1.0, 5.0, 15.0]):
        """
        Args:
            input_channels: Number of input channels (RGB + Template Mask = 4)
            cage_num_vertices: Number of cage control points (K)
            latent_dim: Dimension of latent feature vector z (for downstream tasks)
            use_gaussian_ff: Whether to use Gaussian Fourier features (Tancik et al.)
            gaussian_ff_features: Number of random Fourier features
            gaussian_ff_scale: Scale of random frequencies (10.0 for fine details)
            use_multiscale_ff: Whether to use multi-scale Fourier (prevents bunching)
            multiscale_ff_scales: List of scales for multi-scale Fourier [1.0, 5.0, 15.0]
        """
        super().__init__()
        self.cage_num_vertices = cage_num_vertices
        self.latent_dim = latent_dim
        
        # Backbone: ResNet18
        # We need the spatial features, so we cut it before the global pool
        resnet = models.resnet18(weights=None)
        
        # Modify first conv layer to accept 4 channels
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            input_channels, 
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Keep layers up to layer4, discard avgpool and fc
        self.backbone_features = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 512  # ResNet18 layer4 output channels
        
        # Latent Encoder: Compress global features to latent_z
        # This is the KEY feature for downstream tasks (per README)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, latent_dim),
            nn.ReLU(True)
        )
        
        # Head 1: Affine Transformation (Global) - from latent_z
        self.fc_affine = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)  # 2x3 matrix elements
        )
        
        # Head 2: Local Cage Offsets (Local Feature Sampling)
        # Input: Features sampled at cage vertices (512) + Geometric coords (2)
        cage_feat_dim = self.feature_dim + 2 
        
        self.cage_net = nn.Sequential(
            CircularConv1d(cage_feat_dim, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            CircularConv1d(256, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            CircularConv1d(128, 2, kernel_size=3)  # Output: (dx, dy)
        )
        
        # Head 3: Residual Flow (Fine Details) - from latent_z
        # Only for boundary points refinement
        # Uses hybrid Fourier encoding for high-frequency detail fitting
        # v9: Multi-scale Fourier to prevent bunching (Kass et al. IJCV 1988)
        self.residual_head = ResidualFlowHead(
            latent_dim=latent_dim, 
            hidden_dim=128,
            feature_dim=self.feature_dim,
            use_gaussian_ff=use_gaussian_ff,
            gaussian_ff_features=gaussian_ff_features,
            gaussian_ff_scale=gaussian_ff_scale,
            use_multiscale_ff=use_multiscale_ff,
            multiscale_ff_scales=multiscale_ff_scales
        )
        
        self._init_weights()

    def _init_weights(self):
        # Latent encoder: default init
        for m in self.latent_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        
        # Affine head: Initialize to identity [1, 0, 0, 0, 1, 0]
        nn.init.zeros_(self.fc_affine[-1].weight)
        self.fc_affine[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
        
        # Cage head: Initialize to zero offsets (start with rigid circle)
        for m in self.cage_net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.001)  # Small noise
                nn.init.zeros_(m.bias)

    def forward(self, x, rest_cage):
        """
        Args:
            x: (B, 4, H, W) - Input images
            rest_cage: (B, K, 2) - Normalized coordinates [-1, 1] of the rest cage
                       Used to sample features from the image.
        Returns:
            dict with:
                - affine_matrix: (B, 2, 3)
                - cage_offsets: (B, K, 2)
                - latent_z: (B, latent_dim) - For downstream feature extraction
        """
        B = x.shape[0]
        
        # 1. Extract Spatial Features: (B, 512, H/32, W/32)
        features = self.backbone_features(x)
        
        # 2. Compute Latent Vector z (KEY for downstream)
        global_feat = self.global_pool(features).flatten(1)  # (B, 512)
        latent_z = self.latent_encoder(global_feat)  # (B, latent_dim)
        
        # 3. Predict Affine (from latent_z)
        affine_params = self.fc_affine(latent_z)
        affine_matrix = affine_params.view(B, 2, 3)
        
        # 4. Predict Cage Offsets (Local branch - from spatial features)
        # Sample features at the rest cage locations.
        # grid_sample expects (B, H, W, 2). Treating K vertices as a 1xK grid.
        sample_grid = rest_cage.unsqueeze(1)  # (B, 1, K, 2)
        
        # sampled_feats: (B, 512, 1, K) -> (B, 512, K)
        point_features = F.grid_sample(features, sample_grid, align_corners=True).squeeze(2)
        
        # Concatenate with geometric info (canonical coordinates)
        rest_cage_trans = rest_cage.transpose(1, 2)  # (B, 2, K)
        cage_input = torch.cat([point_features, rest_cage_trans], dim=1)  # (B, 514, K)
        
        # Predict offsets using Circular Convolution
        cage_offsets = self.cage_net(cage_input)  # (B, 2, K)
        cage_offsets = cage_offsets.transpose(1, 2)  # (B, K, 2)
        
        return {
            'affine_matrix': affine_matrix,
            'cage_offsets': cage_offsets,
            'latent_z': latent_z,
            'spatial_features': features
        }
    
    def compute_residual(self, latent_z, deformed_points, spatial_features, steps=3): # [MODIFIED] 接收特徵圖
        """
        Args:
            spatial_features: (B, C, H, W) - ResNet 輸出的特徵圖
        """
        # 1. 初始狀態
        current_points = deformed_points.clone()
        total_residual = torch.zeros_like(deformed_points)
        
        B, N, _ = current_points.shape
        
        # 2. 迭代推論 (The "Sliding" Process)
        for _ in range(steps):
            # [MODIFIED] A. 採樣特徵 (Feature Sampling)
            # 這是 "看" 的動作。看看現在 current_points 腳下的特徵是什麼。
            # grid_sample 需要 (B, C, 1, N) 的 grid，且座標要在 [-1, 1]
            
            # (B, N, 2) -> (B, 1, N, 2)
            sample_grid = current_points.unsqueeze(1) 
            
            # Sample: (B, 512, H, W) + (B, 1, N, 2) -> (B, 512, 1, N)
            sampled_feats = F.grid_sample(spatial_features, sample_grid, align_corners=True)
            
            # Reshape: (B, 512, 1, N) -> (B, N, 512)
            sampled_feats = sampled_feats.squeeze(2).transpose(1, 2)
            
            # B. 預測增量
            # 現在 Residual Head 擁有全知全能的視野了
            delta = self.residual_head(latent_z, current_points, sampled_feats)
            
            # C. 步長限制與更新 (保持不變)
            delta = torch.tanh(delta) * 0.05 
            current_points = current_points + delta
            total_residual = total_residual + delta
        return total_residual