import torch
import torch.nn.functional as F
import numpy as np

def generate_regular_grid(resolution, device='cuda'):
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).to(device)
    return grid

def apply_affine_transform(grid, affine_matrix):
    B, N, _ = grid.shape
    ones = torch.ones(B, N, 1, device=grid.device)
    grid_homo = torch.cat([grid, ones], dim=2)
    grid_transformed = torch.bmm(affine_matrix, grid_homo.transpose(1, 2)).transpose(1, 2)
    return grid_transformed

def precompute_bilinear_weights(points, resolution):
    r_minus_1 = float(resolution - 1)
    coords = (points + 1.0) * (r_minus_1 / 2.0)
    j = torch.floor(coords[:, 0]).long(); i = torch.floor(coords[:, 1]).long()
    i = torch.clamp(i, 0, resolution - 2); j = torch.clamp(j, 0, resolution - 2)
    alpha = coords[:, 0] - j.float(); beta = coords[:, 1] - i.float()
    w_00 = (1.0 - alpha) * (1.0 - beta); w_10 = alpha * (1.0 - beta)
    w_01 = (1.0 - alpha) * beta; w_11 = alpha * beta
    w = torch.stack([w_00, w_10, w_01, w_11], dim=1)
    idx_00 = i * resolution + j; idx_10 = i * resolution + (j + 1)
    idx_01 = (i + 1) * resolution + j; idx_11 = (i + 1) * resolution + (j + 1)
    idx = torch.stack([idx_00, idx_10, idx_01, idx_11], dim=1)
    return idx, w

def deform_points_with_grid(grid_deformed, idx, w):
    B, N, _ = idx.shape
    deformed_pts_list = []
    for b in range(B):
        grid_b = grid_deformed[b]
        idx_b = idx[b]
        w_b = w[b].unsqueeze(2)
        control_points = grid_b[idx_b]
        deformed_pts = torch.sum(control_points * w_b, dim=1)
        deformed_pts_list.append(deformed_pts)
    return torch.stack(deformed_pts_list)

# --- Loss Functions ---

def loss_chamfer(pred_pts, target_pts):
    x = pred_pts.unsqueeze(2)
    y = target_pts.unsqueeze(1)
    dist_sq = torch.sum((x - y)**2, dim=-1)
    min_dist_sq_1 = torch.min(dist_sq, dim=2)[0]
    min_dist_sq_2 = torch.min(dist_sq, dim=1)[0]
    return torch.mean(min_dist_sq_1) + torch.mean(min_dist_sq_2)

def loss_sdf_boundary_focused(pred_boundary, target_sdfs, focus_weight=10.0):
    if pred_boundary.dim() == 4 and pred_boundary.shape[1] == 1:
        pred_boundary = pred_boundary.squeeze(1)
    grid = pred_boundary.unsqueeze(2)
    sampled_sdf = F.grid_sample(target_sdfs, grid, align_corners=True, padding_mode='border').squeeze()
    abs_sdf = torch.abs(sampled_sdf)
    weights = torch.exp(-abs_sdf * focus_weight)
    return (abs_sdf * weights).mean()

def loss_interior_repulsion_conservative(pred_interior, target_sdfs, threshold=0.02):
    if pred_interior.dim() == 4 and pred_interior.shape[1] == 1:
        pred_interior = pred_interior.squeeze(1)
    grid = pred_interior.unsqueeze(2)
    sampled_sdf = F.grid_sample(target_sdfs, grid, align_corners=True, padding_mode='border').squeeze()
    deep_inside = F.relu(sampled_sdf - threshold)
    return deep_inside.mean()

def loss_physics_with_flow_consistency(grid_deformed, resolution, mu=0.01, lam=10.0, barrier_strength=1e4, flow_weight=2.0):
    B = grid_deformed.shape[0]
    grid = grid_deformed.view(B, resolution, resolution, 2)
    scale = (resolution - 1) / 2.0 
    diff_u = (grid[:, :, 1:, :] - grid[:, :, :-1, :]) * scale
    diff_v = (grid[:, 1:, :, :] - grid[:, :-1, :, :]) * scale
    du = diff_u[:, :-1, :, :]; dv = diff_v[:, :, :-1, :] 
    F11 = du[..., 0]; F12 = du[..., 1]; F21 = dv[..., 0]; F22 = dv[..., 1]
    J = F11 * F22 - F12 * F21
    I1 = F11**2 + F12**2 + F21**2 + F22**2
    J_safe = torch.clamp(J, min=1e-4)
    log_term = -torch.log(J_safe)
    energy = (mu / 2.0) * (I1 - 2.0) + (mu / 2.0) * log_term + (lam / 2.0) * ((J - 1.0) ** 2)
    flip_penalty = torch.relu(0.01 - J) * barrier_strength
    flip_ratio = (J <= 0).float().mean()
    
    # Flow Consistency
    x_rest = torch.linspace(-1, 1, resolution, device=grid.device)
    y_rest = torch.linspace(-1, 1, resolution, device=grid.device)
    grid_y_rest, grid_x_rest = torch.meshgrid(y_rest, x_rest, indexing='ij')
    grid_rest = torch.stack([grid_x_rest, grid_y_rest], dim=2).unsqueeze(0).expand(B, -1, -1, -1)
    displacement = grid - grid_rest
    disp_diff_h = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
    disp_diff_v = displacement[:, 1:, :, :] - displacement[:, :-1, :, :]
    flow_loss = (torch.mean(disp_diff_h ** 2) + torch.mean(disp_diff_v ** 2)) * flow_weight
    
    return torch.mean(energy) + torch.mean(flip_penalty) + flow_loss, flip_ratio

def loss_topology_preservation(template_masks, grid_deformed, target_sdfs, cage_resolution, num_samples=1024):
    """
    [修正版] 拓扑保持约束：采样Template的背景/孔洞，确保不进入Target实心区域
    
    核心逻辑：
    - 采样 Template 的「空洞/背景」(mask=0 的区域)
    - 这些点变形后应该落在 Target 的「空洞/背景」(SDF>threshold)
    - 如果进入 Target 的实心区域 (SDF≈0)，就惩罚
    
    典型应用：
    - 甜甜圈 Template → 甜甜圈 Target: 中心空洞必须对齐
    - 圆形 Template → 甜甜圈 Target: 圆心区域的mesh必须移开（进入Target空洞）
    
    SDF 语义:
    - SDF ≈ 0: Target 的实心区域
    - SDF > 0: Target 的空洞/背景
    
    Args:
        template_masks: (B, 1, H, W) - Template Mask (1=实心, 0=空洞)
        grid_deformed: (B, N, 2) - 变形后的控制网格
        target_sdfs: (B, 1, H, W) - Target SDF
        cage_resolution: int - 控制网格分辨率
        num_samples: int - 采样点数
    
    Returns:
        loss: Tensor - 拓扑违反惩罚
        violation_ratio: Tensor - 违反点比例
    """
    B, _, H, W = template_masks.shape
    device = template_masks.device
    
    # 1. 采样 Template 的空洞/背景区域 (mask=0)
    hole_points_list = []
    valid_sample_count = []
    
    for b in range(B):
        mask = template_masks[b, 0].cpu().numpy()
        if mask.max() <= 1.0:
            mask_u8 = (mask * 255).astype(np.uint8)
        else:
            mask_u8 = mask.astype(np.uint8)
        
        # 找到空洞/背景点 (mask < 127)
        ys, xs = np.where(mask_u8 < 127)
        
        if len(xs) == 0:
            # 如果没有空洞（实心Template），这个Loss不起作用
            hole_points_list.append(torch.zeros(num_samples, 2, device=device))
            valid_sample_count.append(0)
            continue
        
        # 随机采样
        if len(xs) > num_samples:
            indices = np.random.choice(len(xs), num_samples, replace=False)
        else:
            indices = np.random.choice(len(xs), num_samples, replace=True)
        
        sampled_xs = xs[indices]
        sampled_ys = ys[indices]
        
        # 转换到 [-1, 1] 归一化坐标
        points_norm = torch.tensor(
            np.stack([sampled_xs / (W - 1), sampled_ys / (H - 1)], axis=1),
            dtype=torch.float32,
            device=device
        ) * 2.0 - 1.0
        
        hole_points_list.append(points_norm)
        valid_sample_count.append(len(xs))
    
    hole_points = torch.stack(hole_points_list)  # (B, num_samples, 2)
    
    # 2. 变形这些空洞点
    deformed_holes_list = []
    for b in range(B):
        if valid_sample_count[b] == 0:
            deformed_holes_list.append(torch.zeros(num_samples, 2, device=device))
            continue
        
        idx, w = precompute_bilinear_weights(hole_points[b], cage_resolution)
        deformed_pts = deform_points_with_grid(
            grid_deformed[b:b+1],
            idx.unsqueeze(0),
            w.unsqueeze(0)
        ).squeeze(0)
        deformed_holes_list.append(deformed_pts)
    
    deformed_holes = torch.stack(deformed_holes_list)  # (B, num_samples, 2)
    
    # 3. 采样 Target SDF
    grid_sample_input = deformed_holes.unsqueeze(2)  # (B, num_samples, 1, 2)
    sampled_sdf = F.grid_sample(
        target_sdfs,
        grid_sample_input,
        align_corners=True,
        padding_mode='border'
    ).squeeze(3).squeeze(1)  # (B, num_samples)
    
    # 4. 惩罚：Template的空洞点如果进入Target的实心区域 (SDF<threshold)
    # 期望：空洞对齐空洞 (SDF应该>threshold)
    threshold = 0.05  # 约13像素安全距离
    violation_penalty = F.relu(threshold - sampled_sdf)
    
    # 5. 计算违反比例
    violation_ratio = (sampled_sdf < threshold).float().mean()
    
    # 只对有空洞的样本计算loss
    valid_losses = []
    for b in range(B):
        if valid_sample_count[b] > 0:
            valid_losses.append(violation_penalty[b].mean())
    
    if len(valid_losses) == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    return torch.stack(valid_losses).mean(), violation_ratio


def loss_spatial_attention_guidance(spatial_weights, target_pts, grid_global, cage_resolution):
    """
    [修正版] 引導 Attention 關注那些【在全域對齊後】靠近目標的點。
    Args:
        spatial_weights: (B, N, 1)
        target_pts: (B, M, 2)
        grid_global: (B, N, 2)  <-- 這裡是關鍵，必須是 Affine 之後的網格
    """
    B, N, _ = spatial_weights.shape
    M = target_pts.shape[1]
    
    # 計算 grid_global 到 target_pts 的距離
    grid_expanded = grid_global.unsqueeze(2).expand(B, N, M, 2)
    target_expanded = target_pts.unsqueeze(1).expand(B, N, M, 2)
    
    # 歐式距離
    dist = torch.norm(grid_expanded - target_expanded, dim=3)
    min_dist, _ = torch.min(dist, dim=2)  # (B, N) 每個控制點到最近目標點的距離
    
    # [修正] 使用自適應 Scale: 初期大範圍引導，後期聚焦邊界
    # normalized_dist: 0~2 歸一化到 0~1
    normalized_dist = min_dist / 2.0
    # 使用平滑的 Sigmoid 曲線而非尖銳的 Exponential
    ideal_weights = 1.0 / (1.0 + (normalized_dist * 5.0) ** 2)
    
    # [關鍵修正] Detach ideal_weights 以避免梯度回傳到 affine
    # 理由：Attention 監督不應影響主要變形參數的學習（參考 STN 論文）
    ideal_weights = ideal_weights.detach()
    
    # MSE Loss
    mse_loss = F.mse_loss(spatial_weights.squeeze(2), ideal_weights)
    
    return mse_loss

