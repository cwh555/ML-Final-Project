import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Advanced Building Block (Hybrid)
# ==========================================

class TemplateBlock(nn.Module):
    """
    Hybrid Block: Gating (Clean) -> Mixer (Fuse) -> Feature MLP (Process)
    """
    def __init__(self, num_templates, dim, dropout=0.1):
        super().__init__()
        
        # --- A. Gating (Template-wise Selection) ---
        # 負責過濾掉 STN 沒對準或變形壞掉的 Template
        # 邏輯: Squeeze(Mean) -> MLP(5->32->5) -> Sigmoid -> Reweight
        # 注意: 這裡不加 Dropout，以確保開關機制的穩定性
        self.gating_mlp = nn.Sequential(
            nn.Linear(num_templates, 32),
            nn.GELU(),
            nn.Linear(32, num_templates),
            nn.Sigmoid()
        )
        
        # --- B. Mixer (Template-wise Interaction) ---
        # 負責整合不同 Template 的資訊 (Fusion)
        # 邏輯: Norm -> Transpose -> Linear(5->5) -> Transpose -> Add
        self.mixer_norm = nn.LayerNorm(dim)
        self.mixer_linear = nn.Linear(num_templates, num_templates)
        # self.mixer_dropout = nn.Dropout(dropout)
        
        # --- C. Feature MLP (Channel-wise Processing) ---
        # 負責處理每個 Template 內部的幾何特徵
        # 邏輯: Norm -> Linear(D->2D) -> GELU -> Linear(2D->D) -> Add
        self.feat_norm = nn.LayerNorm(dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), # 升維做非線性變換
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), # 降維回原長度
        )

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        
        # 1. Gating (Cleaning)
        # Global pooling over feature dim -> (B, N)
        gate_in = x.mean(dim=-1)
        gate_weights = self.gating_mlp(gate_in) # (B, N)
        gate_weights = gate_weights.unsqueeze(-1) # (B, N, 1)
        
        # Reweight: 強制壓低雜訊 Template 的數值
        # 這裡不使用 Residual，直接乘上去，達到過濾效果
        x = x * gate_weights
        
        # 2. Mixer (Fusion)
        shortcut = x
        x = self.mixer_norm(x)
        x = x.transpose(1, 2)     # (B, D, N)
        x = self.mixer_linear(x)  # Mixing across templates
        x = x.transpose(1, 2)     # (B, N, D)
        # x = self.mixer_dropout(x)
        x = x + shortcut          # Residual Add
        
        # 3. Feature MLP (Processing)
        shortcut = x
        x = self.feat_norm(x)
        x = self.feat_mlp(x)
        x = x + shortcut          # Residual Add
        
        return x

class ResLinearBlock(nn.Module):
    """
    Simple Residual Block for the final Fusion Stage (Flattened).
    Linear -> BN -> GELU -> Dropout -> Add
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.downsample = None
        if in_dim != out_dim:
            self.downsample = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = x
        out = self.net(x)
        if self.downsample:
            residual = self.downsample(x)
        return out + residual

# ==========================================
# 2. Main Encoder Architecture
# ==========================================

class GeometricEncoder(nn.Module):
    def __init__(
        self, 
        input_dim_per_template=2182, 
        num_templates=5,
        shared_hidden_dim=128,
        shared_layers=2,        # TemplateBlock 的層數
        fusion_hidden_dim=256,
        fusion_layers=1,        # Flatten 後的 ResBlock 層數
        output_dim=128,
        dropout=0.1             # [新增] Dropout 機率
    ):
        super().__init__()
        
        # --- Stage 1: Initial Projection (Shared) ---
        # 先將龐大的參數壓縮到 hidden_dim
        self.input_proj = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(input_dim_per_template, shared_hidden_dim),
            nn.LayerNorm(shared_hidden_dim), # 使用 LayerNorm 比較適合 MLP 架構
            nn.GELU(),
            # nn.Dropout(dropout)
        )
        
        # --- Stage 2: Template-Aware Processing (Shared) ---
        # 使用 Hybrid TemplateBlock 進行 Gating & Mixing
        self.shared_blocks = nn.Sequential(*[
            TemplateBlock(num_templates, shared_hidden_dim, dropout)
            for _ in range(shared_layers)
        ])
        
        # --- Stage 3: Fusion (Flattened) ---
        flatten_dim = num_templates * shared_hidden_dim
        fusion_blocks = []
        
        # 3.1 First Linear Fusion
        fusion_blocks.append(
            nn.Sequential(
                nn.Linear(flatten_dim, fusion_hidden_dim),
                nn.BatchNorm1d(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        )
        
        # 3.2 Residual Fusion Blocks
        for _ in range(fusion_layers):
            fusion_blocks.append(
                ResLinearBlock(fusion_hidden_dim, fusion_hidden_dim, dropout)
            )
            
        # 3.3 Final Head (No Activation)
        fusion_blocks.append(
            nn.Linear(fusion_hidden_dim, output_dim)
        )
        
        self.fusion_mlp = nn.Sequential(*fusion_blocks)

    def forward(self, x):
        # x: (B, N, Input_Dim)
        B, N, _ = x.shape
        
        # 1. Projection: (B, N, Shared_Dim)
        x = self.input_proj(x)
        
        # 2. Shared Blocks (Gating -> Mixer -> Feat): (B, N, Shared_Dim)
        x = self.shared_blocks(x)
        
        # 3. Flatten: (B, N * Shared_Dim)
        x = x.view(B, -1)
        
        # 4. Fusion MLP: (B, Output_Dim)
        logits = self.fusion_mlp(x)
        
        # 5. Normalize
        embedding = F.normalize(logits, p=2, dim=-1)
        
        return embedding

# ==========================================
# 3. Full Model Wrapper
# ==========================================

class FullContrastiveModel(nn.Module):
    def __init__(self, 
                 stn_class, 
                 stn_ckpt_path, 
                 encoder_config=None, 
                 encoder_ckpt_path=None,
                 cfg=None,
                 device='cuda'):
        super().__init__()
        
        # 1. STN Loading (Mandatory)
        if stn_ckpt_path is None:
            raise ValueError("[Model] stn_ckpt_path is required!")
            
        # self.stn = stn_class(input_channels=4, cage_resolution=32)
        self.stn = stn_class(
            checkpoint_path=stn_ckpt_path,
            template_dir=cfg.data.template_dir,
            template_names=cfg.data.template_names,
            cage_num_vertices=128,
            cage_radius=1.2,
            device=device
        )
        
        # Load STN weights
        # try:
        #     ckpt = torch.load(stn_ckpt_path, map_location='cpu')
        #     state_dict = ckpt.get('model_state_dict', ckpt)
        #     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        #     self.stn.load_state_dict(new_state_dict)
        #     print(f"[Model] STN loaded from {stn_ckpt_path}")
        # except Exception as e:
        #     raise RuntimeError(f"[Model] Failed to load STN: {e}")
        
        self.stn.to(device)
        for param in self.stn.parameters():
            param.requires_grad = False
            
        # 2. Encoder Initialization
        self.params_per_template = 2182
        if encoder_config is None: encoder_config = {}
        
        self.encoder = GeometricEncoder(
            **encoder_config
        )
        
        # 3. Encoder Loading (Optional)
        if encoder_ckpt_path is not None:
            self.load_encoder_weights(encoder_ckpt_path)

    def load_encoder_weights(self, path):
        print(f"[Model] Loading Encoder from {path}...")
        try:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt.get('encoder_state_dict', ckpt.get('model_state_dict', ckpt))
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.encoder.load_state_dict(new_state_dict)
            print("[Model] Encoder loaded successfully.")
        except Exception as e:
            print(f"[Model Error] Failed to load Encoder: {e}")
            raise e

    def train(self, mode=True):
        super().train(mode)
        self.stn.eval() # Force STN eval
        return self

    def forward(self, x, weight_boundary=None, template_names=None):
        # x: (B, 5, 4, H, W) -> Stacked images
        B, N, C, H, W = x.shape
        x_reshaped = x.view(B * N, C, H, W)
        
        with torch.inference_mode():
            # stn_out = self.stn(x_reshaped, test=True)
            # template_names = template_names.flatten().tolist()
            weight_boundary = weight_boundary.flatten(start_dim=0, end_dim=1)
            stn_out = self.stn(x_reshaped, weight_boundary=weight_boundary, template_names=template_names)
        # print(B*N, stn_out['affine'].shape, stn_out['coarse_feat'].shape, stn_out['fine_feat'].shape)
        aff = stn_out['affine'].contiguous().view(B * N, -1)
        coarse = stn_out['coarse_feat'].contiguous().view(B * N, -1)
        fine = stn_out['fine_feat'].contiguous().view(B * N, -1)
        
        # raw_features = torch.cat([aff, coarse], dim=1) # (B*N, 262)
        raw_features = torch.cat([aff, coarse, fine], dim=1) # (B*N, 2182)
        encoder_input = raw_features.view(B, N, -1)         # (B, N, 2182)
        
        embedding = self.encoder(encoder_input)
        
        return embedding