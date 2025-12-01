"""Cage Deformation Loss Functions - Hierarchical Cage-Residual Architecture

v9 Update: Anti-Bunching Strategy (基於 Active Contours 理論)
==============================================================

Based on:
- IPC (Incremental Potential Contact, SIGGRAPH 2020) for barrier function
- Neural Cages (CVPR 2020) for MVC negative weight penalty  
- Deforming Autoencoders (ECCV 2018) for residual magnitude penalty
- Kass et al. (IJCV 1988) - Active Contours (Snakes) ★ KEY REFERENCE ★

Anti-Bunching Strategy (v9):
  核心問題: Chamfer/Suction 只有 Normal Force (法向力)，會導致點在凹陷處堆積
  
  解決方案 (Kass et al., IJCV 1988 - 被引用 27,634 次):
  1. External Energy (外力): SDF 梯度 - 提供「貼合推力」
  2. Internal Energy (內力): Uniform Edge Length - 提供「切向張力」防止堆積
  
  v9 Changes:
  - Suction (Chamfer): DISABLED (alpha=0) - 會導致 bunching
  - Edge Uniform: INCREASED (alpha=500) - 防止堆積的唯一數學解
  - Residual Smooth: INCREASED (alpha=1.0) - 確保線條平滑
  - Residual Mag: DISABLED (alpha=0) - 完全釋放 residual

References:
  - Kass, Witkin & Terzopoulos, "Snakes: Active Contour Models", IJCV 1988
  - Peng et al., "Deep Snake for Real-Time Instance Segmentation", CVPR 2020
  - Tancik et al., "Fourier Features Let Networks Learn High Freq", NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import src.utils.grid_utils as utils


class CageDeformationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w_sdf = cfg.train.alpha_sdf
        self.w_smooth = cfg.train.alpha_cage_smooth
        self.w_edge = cfg.train.alpha_cage_edge
        self.w_repul = cfg.train.alpha_cage_repul
        self.w_mvc = cfg.train.alpha_mvc_negative
        self.w_area = getattr(cfg.train, 'alpha_area', 10.0)
        self.w_res_mag = getattr(cfg.train, 'alpha_residual_mag', 0.1)
        self.w_res_smooth = getattr(cfg.train, 'alpha_residual_smooth', 0.01)
        self.barrier_radius = cfg.train.barrier_safe_radius
        
        #v11
        self.w_grad_flow = getattr(cfg.train, 'alpha_grad_flow', 0.0)
        # Deep Concavity: Suction and Sliding weights
        # Suction: Reverse Chamfer - Target pulls Template into concavities
        self.w_suction = getattr(cfg.train, 'alpha_suction', 100.0)
        # Sliding: Uniform Edge - Points redistribute along contour
        self.w_edge_uniform = getattr(cfg.train, 'alpha_edge_uniform', 50.0)
        
        # Spatially Adaptive Regularization parameters
        # Higher alpha = more sensitive to error (softer in concave regions)
        self.res_adaptive_alpha = getattr(cfg.train, 'residual_adaptive_alpha', 10.0)
        self.res_min_stiffness = getattr(cfg.train, 'residual_min_stiffness', 0.01)
        
        # Curriculum learning phases
        self.warmup_affine_epochs = getattr(cfg.train, 'warmup_affine_epochs', 5)
        self.warmup_cage_epochs = getattr(cfg.train, 'warmup_cage_epochs', 50)

    def forward(self, deformed_b, deformed_i, cage_deformed, cage_rest, target_sdf, 
                mvc_weights_b, mvc_weights_i, epoch=0, residual_b=None, cage_after_affine=None,
                target_sdf_map=None, pts_target_boundary=None, boundary_is_ordered=False):
        """
        Args:
            deformed_b: (B, Nb, 2) - Deformed boundary points
            deformed_i: (B, Ni, 2) - Deformed interior points
            cage_deformed: (B, K, 2) - Deformed cage vertices (after affine + offsets)
            cage_rest: (B, K, 2) - Original cage vertices
            target_sdf: Function to query SDF, returns (B, 1, N)
            mvc_weights_b: (B, Nb, K) - MVC weights for boundary
            mvc_weights_i: (B, Ni, K) - MVC weights for interior
            epoch: current epoch for curriculum learning
            residual_b: (B, Nb, 2) - Residual displacement for boundary (optional)
            cage_after_affine: (B, K, 2) - Cage after ONLY affine (before offsets)
            target_sdf_map: (B, 1, H, W) - SDF map for bbox extraction
            pts_target_boundary: (B, Mt, 2) - Target boundary points for Suction loss
            boundary_is_ordered: bool - Whether deformed_b points are in contour order
        """
        
        # 1. SDF Alignment Loss (Bidirectional - Neural Cages style)
        sdf_b = target_sdf(deformed_b).squeeze(-1).squeeze(1)  # (B, Nb)
        sdf_i = target_sdf(deformed_i).squeeze(-1).squeeze(1)  # (B, Ni)
        
        # Forward: Deformed boundary → Target boundary (SDF=0)
        loss_sdf_b = torch.abs(sdf_b).mean()
        
        # Forward: Deformed interior → inside Target (SDF<0)
        loss_sdf_i = torch.relu(sdf_i).mean()
        
        # Backward: Target coverage - deformed bbox should match target bbox
        # This prevents collapse: if deformed shrinks, bbox won't match target
        if target_sdf_map is not None:
            loss_coverage = self._loss_target_coverage(deformed_b, target_sdf_map)
        else:
            loss_coverage = torch.tensor(0.0, device=cage_deformed.device)
        
        loss_sdf = loss_sdf_b + loss_sdf_i + loss_coverage

        # 2. Geometry Regularization
        loss_smooth = self._loss_cage_smoothness(cage_deformed)
        loss_edge = self._loss_cage_edge_consistency(cage_deformed, cage_rest)
        
        # 3. MVC Validity (Neural Cages: prevent negative weights)
        if epoch >= self.warmup_affine_epochs:
            loss_mvc = self._loss_mvc_validity(mvc_weights_b, mvc_weights_i)
        else:
            loss_mvc = torch.tensor(0.0, device=cage_deformed.device)
        
        # 4. IPC Barrier (Prevent self-intersection)
        if epoch >= self.warmup_cage_epochs:
            loss_barrier = self._loss_ipc_barrier_vectorized(cage_deformed)
        else:
            loss_barrier = torch.tensor(0.0, device=cage_deformed.device)
        
        # 5. Removed: area loss is now handled by coverage loss in SDF
        loss_area = torch.tensor(0.0, device=cage_deformed.device)
        
        # 6. Spatially Adaptive Residual Regularization (針對深凹陷優化)
        # Reference: Niethammer et al., "Spatially-Varying Regularization"
        # 
        # Core idea: In concave regions (high |SDF|), reduce regularization
        # to allow large residual displacements. In flat regions (low |SDF|),
        # keep strong regularization to maintain stability.
        #
        # Formula: stiffness = exp(-alpha * |SDF|) + min_stiffness
        #   - High |SDF| (concave): stiffness → min_stiffness (soft, allow large residual)
        #   - Low |SDF| (flat): stiffness → 1.0 (hard, penalize residual)
        loss_res_mag = torch.tensor(0.0, device=cage_deformed.device)
        loss_res_smooth = torch.tensor(0.0, device=cage_deformed.device)
        loss_suction = torch.tensor(0.0, device=cage_deformed.device)
        loss_edge_uniform = torch.tensor(0.0, device=cage_deformed.device)
        
        if residual_b is not None:
            # 1. Compute local alignment error (|SDF| at each boundary point)
            # detach(): We only use this for weight computation, not for gradient
            local_error = torch.abs(sdf_b.detach())  # (B, Nb)
            
            # 2. Adaptive stiffness map
            # stiffness = exp(-alpha * error) + epsilon
            # - alpha: sensitivity (higher = softer in concave regions)
            # - epsilon: minimum stiffness (prevent numerical instability)
            stiffness_map = torch.exp(-self.res_adaptive_alpha * local_error) + self.res_min_stiffness
            # stiffness_map: (B, Nb), range [min_stiffness, 1.0 + min_stiffness]
            
            # 3. Weighted magnitude loss
            # In concave regions (high error), stiffness is low → less penalty on large residual
            # residual_b: (B, Nb, 2)
            res_sq = (residual_b ** 2).sum(dim=-1)  # (B, Nb)
            loss_res_mag = (stiffness_map * res_sq).mean()
            
            # 4. Residual smoothness loss (prevent noisy/jagged boundaries)
            # If points are ordered, use neighbor diff; otherwise use global variance
            if boundary_is_ordered:
                # Ordered: use sequential difference (Active Contours style)
                res_diff = residual_b[:, 1:] - residual_b[:, :-1]
                loss_res_smooth = (res_diff ** 2).sum(dim=-1).mean()
            else:
                # Random: use global variance penalty
                res_mean = residual_b.mean(dim=1, keepdim=True)  # (B, 1, 2)
                loss_res_smooth = ((residual_b - res_mean) ** 2).sum(dim=-1).mean()
            
            # ================================================================
            # 7. TARGET SUCTION LOSS (逆向吸力)
            # Reference: Fan et al. (CVPR 2017) - Bidirectional Chamfer Distance
            #
            # Core insight: SDF only provides gradients near the boundary.
            # For deep concavities, we need Target points to "pull" Template.
            #
            # Formula: For each target point, find nearest deformed point,
            #          minimize that distance → long-range attractive force
            # ================================================================
            if pts_target_boundary is not None and self.w_suction > 0:
                # Use torch.enable_grad() to ensure gradient computation works
                # even during validation (torch.no_grad() context)
                with torch.enable_grad():
                    # 1. 基礎距離矩陣
                    dist_mat = torch.cdist(pts_target_boundary, deformed_b)
                    min_dist_val, min_dist_idx = torch.min(dist_mat, dim=2) # (B, Mt)
                    
                    # 2. 準備法線資料
                    # A. Template 點的法線 (用 SDF 梯度估算)
                    # Need to ensure deformed_b has requires_grad for autograd
                    deformed_b_grad = deformed_b.detach().clone().requires_grad_(True)
                    sdf_at_template = target_sdf(deformed_b_grad).squeeze()
                    temp_grads = torch.autograd.grad(
                        outputs=sdf_at_template.sum(), inputs=deformed_b_grad,
                        create_graph=False, retain_graph=False, only_inputs=True)[0]
                    temp_normals = F.normalize(temp_grads, dim=-1)
                    
                    # B. 對應的 Template 法線
                    B, Mt = min_dist_idx.shape
                    batch_idx = torch.arange(B, device=deformed_b.device).unsqueeze(1).expand(-1, Mt)
                    matched_normals = temp_normals[batch_idx, min_dist_idx]
                    
                    # C. Target 點的法線 (若 Dataset 無提供，現場算)
                    # pts_target_boundary 位於 SDF=0 處，梯度即法線
                    # NOTE: pts_target_boundary comes from dataset without requires_grad
                    # We need to enable requires_grad to compute SDF gradients
                    pts_target_b_grad = pts_target_boundary.detach().clone().requires_grad_(True)
                    sdf_at_target = target_sdf(pts_target_b_grad).squeeze()
                    tgt_grads = torch.autograd.grad(
                        outputs=sdf_at_target.sum(), inputs=pts_target_b_grad,
                        create_graph=False, retain_graph=False, only_inputs=True)[0]
                    # 注意: SDF 梯度指向外部。
                    # Target 表面法線通常指向外部。 Template 表面法線也指向外部。
                    # 如果兩者 "面對面"，它們的法線應該是 "相反" 的嗎？
                    # 不，SDF 梯度都指向 "SDF 增加的方向" (遠離物體)。
                    # 如果 Template 在外部，Target 在表面。
                    # 兩者的梯度方向應該是 "一致" 的 (都指向外部)。
                    # 如果隔了一道牆 (在背面)，Template 的梯度會指向 "背面的外部"，與 Target 梯度 "相反" 或 "垂直"。
                    
                    tgt_normals = F.normalize(tgt_grads, dim=-1)
                    
                    # 3. 可見性過濾 (Visibility Filter)
                    # 判斷標準: 兩者的 SDF 梯度方向必須 "一致" (Dot > 0)
                    # 解釋: 
                    # - 正常情況: Template 在洞口外。Template 的梯度指向外，Target 梯度指向外 -> Dot > 0
                    # - 穿牆情況: Template 在 U 型管背面。Template 的梯度指向 "背面的外" (左)，Target 梯度指向 "洞口的外" (右) -> Dot < 0
                    
                    alignment = (matched_normals * tgt_normals).sum(dim=-1)
                    
                    # 嚴格過濾: 只吸方向一致的點 (Cosine Similarity > 0.5, 夾角小於 60 度)
                    valid_mask = alignment > 0.5 
                    
                    # 4. 只對 "可見" (合法) 的點產生吸力
                    valid_dist = min_dist_val[valid_mask]
                    
                    if valid_dist.numel() > 0:
                        # OHEM: 為了更強的吸力，我們可以只取最遠的那批點 (但必須是合法的)
                        loss_suction = (valid_dist ** 2).mean()            
            # ================================================================
            # 8. UNIFORM EDGE LENGTH LOSS (切向滑動)
            # Reference: Kass et al. (IJCV 1988) - Active Contours (Snakes)
            #
            # Core insight: When Affine shrinks the shape, points in concavities
            # become sparse. This loss forces points to "slide" along contour
            # from dense regions to sparse regions.
            #
            # Formula: Minimize variance of edge lengths → uniform distribution
            # ================================================================
            if boundary_is_ordered:
                # Compute edge lengths between consecutive points
                # deformed_b: (B, Nb, 2), points are in contour order
                p_curr = deformed_b
                p_next = torch.roll(deformed_b, shifts=-1, dims=1)  # Circular: last connects to first
                edge_lens = torch.norm(p_next - p_curr, dim=-1)  # (B, Nb)
                
                # Mean edge length per sample
                avg_len = edge_lens.mean(dim=1, keepdim=True)  # (B, 1)
                
                # Penalize variance: force all edges to have similar length
                # This causes points to slide from short-edge regions to long-edge regions
                loss_edge_uniform = ((edge_lens - avg_len) ** 2).mean()

        loss_grad_flow = torch.tensor(0.0, device=cage_deformed.device)
        if residual_b is not None and self.w_grad_flow > 0:
            # Use torch.enable_grad() to ensure gradient computation works
            # even during validation (torch.no_grad() context)
            with torch.enable_grad():
                # 1. 重新計算 SDF 值
                # NOTE: We detach deformed_b to avoid needing 2nd order derivatives
                # through grid_sampler (cudnn_grid_sampler doesn't support it)
                deformed_b_for_grad = deformed_b.detach().requires_grad_(True)
                sdf_val = target_sdf(deformed_b_for_grad).squeeze()
                
                # 2. 計算 Target 在當前點的法線 (SDF 梯度)
                # create_graph=False because we don't need 2nd order derivatives
                target_grad = torch.autograd.grad(
                    outputs=sdf_val.sum(), 
                    inputs=deformed_b_for_grad, 
                    create_graph=False,  # NO 2nd order derivatives needed
                    retain_graph=False,
                    only_inputs=True
                )[0]
                
                # 3. 計算方向一致性
                # 目標方向: -target_grad (下坡方向/指向內部)
                # Detach target_grad - we only want gradients to flow through residual_b
                target_dir = F.normalize(-target_grad.detach(), dim=-1)
                res_dir = F.normalize(residual_b, dim=-1)
                
                # Loss = 1 - CosineSimilarity (方向越一致 Loss 越小)
                dot_prod = (res_dir * target_dir).sum(dim=-1)
                loss_grad_flow = (1.0 - dot_prod).mean()


        total_loss = (self.w_sdf * loss_sdf +
                      self.w_smooth * loss_smooth +
                      self.w_edge * loss_edge +
                      self.w_mvc * loss_mvc +
                      self.w_repul * loss_barrier + 
                      self.w_res_mag * loss_res_mag +
                      self.w_res_smooth * loss_res_smooth +
                      self.w_suction * loss_suction +
                      self.w_edge_uniform * loss_edge_uniform +
                      self.w_grad_flow * loss_grad_flow)

        return {
            "total": total_loss,
            "sdf": loss_sdf,
            "sdf_b": loss_sdf_b,
            "sdf_i": loss_sdf_i,
            "coverage": loss_coverage,
            "smooth": loss_smooth,
            "edge": loss_edge,
            "mvc": loss_mvc,
            "barrier": loss_barrier,
            "area": loss_area,
            "res_mag": loss_res_mag,
            "res_smooth": loss_res_smooth,
            "suction": loss_suction,
            "edge_uniform": loss_edge_uniform
        }

    def _loss_target_coverage(self, deformed_b, target_sdf_map):
        """
        Bidirectional coverage loss - prevents collapse.
        
        Core insight: If deformed shrinks to a small blob, it can still have
        SDF=0 (on target boundary), but it won't COVER the target.
        
        Method: Compare deformed bbox to target bbox extracted from SDF map.
        Target boundary is where SDF ≈ 0, so we find those pixels' extent.
        
        Based on Chamfer Distance principle: need bidirectional matching.
        
        Coordinate system: deformed_b is (x, y) in [-1, 1], where:
        - x: horizontal (corresponds to W dimension)
        - y: vertical (corresponds to H dimension)
        """
        B, _, H, W = target_sdf_map.shape
        device = target_sdf_map.device
        
        # Extract target bbox from SDF map (where |SDF| < threshold = boundary)
        threshold = 0.05  # Small SDF = near boundary
        boundary_mask = (target_sdf_map.abs() < threshold).squeeze(1)  # (B, H, W)
        
        # Create coordinate grids in [-1, 1]
        # Note: grid_sample uses (x, y) where x is W and y is H
        y_coords = torch.linspace(-1, 1, H, device=device)  # H dimension = y
        x_coords = torch.linspace(-1, 1, W, device=device)  # W dimension = x
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # yy: (H,W), xx: (H,W)
        
        # Vectorized bbox computation for all batches
        target_bbox_losses = []
        for b in range(B):
            mask_b = boundary_mask[b]  # (H, W)
            if mask_b.sum() < 10:  # Skip if no boundary found
                target_bbox_losses.append(torch.tensor(0.0, device=device))
                continue
            
            # Target bbox (from SDF boundary pixels)
            # xx, yy are already in [-1, 1] range
            x_valid = xx[mask_b]  # x coordinates of boundary pixels
            y_valid = yy[mask_b]  # y coordinates of boundary pixels
            
            tgt_min_x, tgt_max_x = x_valid.min(), x_valid.max()
            tgt_min_y, tgt_max_y = y_valid.min(), y_valid.max()
            tgt_center = torch.stack([
                (tgt_min_x + tgt_max_x) / 2,  # center x
                (tgt_min_y + tgt_max_y) / 2   # center y
            ])
            tgt_size = torch.stack([
                tgt_max_x - tgt_min_x,  # width
                tgt_max_y - tgt_min_y   # height
            ])
            
            # Deformed bbox - deformed_b is (Nb, 2) with (x, y) coordinates
            def_pts = deformed_b[b]  # (Nb, 2)
            def_min = def_pts.min(dim=0)[0]  # (2,) = (min_x, min_y)
            def_max = def_pts.max(dim=0)[0]  # (2,) = (max_x, max_y)
            def_center = (def_min + def_max) / 2  # (2,)
            def_size = def_max - def_min  # (2,) = (width, height)
            
            # Loss components:
            # 1. Center should match
            center_loss = ((def_center - tgt_center) ** 2).sum()
            
            # 2. Size should be similar (allow 0.5x to 2.0x range)
            # This prevents both collapse (too small) and explosion (too large)
            size_ratio = def_size / (tgt_size + 1e-6)
            size_loss = (F.relu(0.5 - size_ratio) + F.relu(size_ratio - 2.0)).sum()
            
            target_bbox_losses.append(center_loss + size_loss)
        
        return torch.stack(target_bbox_losses).mean()

    def _loss_mvc_validity(self, weights_b, weights_i):
        """
        Neural Cages: Penalize negative MVC weights.
        Negative weights can cause fold-overs and artifacts.
        """
        neg_b = F.relu(-weights_b).mean()
        neg_i = F.relu(-weights_i).mean()
        return neg_b + neg_i

    def _loss_cage_smoothness(self, cage):
        """
        Laplacian smoothness: each vertex should be near the midpoint of its neighbors.
        """
        cage_prev = torch.roll(cage, shifts=1, dims=1)
        cage_next = torch.roll(cage, shifts=-1, dims=1)
        laplacian = cage - 0.5 * (cage_prev + cage_next)
        return (laplacian ** 2).sum(dim=-1).mean()

    def _loss_cage_edge_consistency(self, cage, cage_rest):
        """
        Edge length consistency: edge lengths should be similar.
        Also penalize drastic edge length changes from rest state.
        """
        cage_next = torch.roll(cage, shifts=-1, dims=1)
        edge_len = torch.norm(cage - cage_next, dim=-1)
        mean_len = torch.mean(edge_len, dim=1, keepdim=True)
        
        # Variance penalty
        var_loss = torch.mean((edge_len - mean_len) ** 2)
        
        # Also compare to rest edge lengths (prevent extreme stretching)
        cage_rest_next = torch.roll(cage_rest, shifts=-1, dims=1)
        rest_edge_len = torch.norm(cage_rest - cage_rest_next, dim=-1)
        
        # Penalize if edge changes by more than 2x or less than 0.5x
        ratio = edge_len / (rest_edge_len + 1e-8)
        stretch_loss = F.relu(ratio - 2.0).mean() + F.relu(0.5 - ratio).mean()
        
        return var_loss + 0.1 * stretch_loss

    def _loss_ipc_barrier_vectorized(self, cage):
        """
        Vectorized IPC barrier - GPU-efficient version.
        Computes all vertex-to-edge distances in parallel.
        
        Time complexity: O(1) GPU operations instead of O(K²) sequential loops.
        """
        B, K, _ = cage.shape
        d_hat = self.barrier_radius
        
        # Get all vertices and edges
        vertices = cage  # (B, K, 2)
        cage_next = torch.roll(cage, shifts=-1, dims=1)
        
        # Edge endpoints: edge j goes from cage[:, j] to cage_next[:, j]
        seg_starts = cage  # (B, K, 2)
        seg_ends = cage_next  # (B, K, 2)
        
        # Expand for broadcasting: all pairs of (vertex i, edge j)
        # vertices: (B, K, 1, 2) - vertex i
        # seg_starts: (B, 1, K, 2) - edge j start
        # seg_ends: (B, 1, K, 2) - edge j end
        v = vertices.unsqueeze(2)  # (B, K, 1, 2)
        s = seg_starts.unsqueeze(1)  # (B, 1, K, 2)
        e = seg_ends.unsqueeze(1)  # (B, 1, K, 2)
        
        # Compute point-to-segment distance for all pairs
        # Using vectorized formula
        seg_vec = e - s  # (B, 1, K, 2)
        point_vec = v - s  # (B, K, K, 2)
        
        seg_len_sq = (seg_vec ** 2).sum(dim=-1, keepdim=True)  # (B, 1, K, 1)
        
        # Project point onto line
        t = (point_vec * seg_vec).sum(dim=-1, keepdim=True) / (seg_len_sq + 1e-8)  # (B, K, K, 1)
        t = torch.clamp(t, 0.0, 1.0)
        
        # Closest point on segment
        closest = s + t * seg_vec  # (B, K, K, 2)
        
        # Distance from vertex to closest point
        dist_sq = ((v - closest) ** 2).sum(dim=-1)  # (B, K, K)
        dist = torch.sqrt(dist_sq + 1e-8)  # (B, K, K)
        
        # Create adjacency mask: vertex i is adjacent to edges (i-1) and (i)
        # So we mask out those pairs
        idx = torch.arange(K, device=cage.device)
        adj_mask = torch.zeros(K, K, device=cage.device, dtype=torch.bool)
        adj_mask[idx, idx] = True  # vertex i, edge i (same index)
        adj_mask[idx, (idx - 1) % K] = True  # vertex i, edge i-1
        
        # Apply mask - set adjacent distances to large value (no barrier)
        adj_mask = adj_mask.unsqueeze(0)  # (1, K, K)
        dist_masked = dist.masked_fill(adj_mask, float('inf'))
        
        # Log barrier for non-adjacent pairs
        barrier_active = (dist_masked < 2 * d_hat).float()
        barrier_val = barrier_active * (-torch.log(torch.clamp(dist_masked - d_hat, min=1e-8)))
        
        # Replace inf values with 0 for mean calculation
        barrier_val = barrier_val.masked_fill(adj_mask, 0.0)
        
        # Normalize by number of non-adjacent pairs: K * (K - 3) for closed polygon
        num_pairs = K * (K - 3)
        return barrier_val.sum() / (B * num_pairs + 1e-8)

    def _loss_ipc_barrier(self, cage):
        """
        IPC-style barrier function to prevent cage self-intersection.
        Check distance between non-adjacent cage edges and vertices.
        Uses a log-barrier formulation for smooth gradients.
        NOTE: This is the O(K²) loop version, kept for reference. Use vectorized version.
        """
        B, K, _ = cage.shape
        d_hat = self.barrier_radius  # Minimum safe distance
        
        # Get cage edges
        cage_next = torch.roll(cage, shifts=-1, dims=1)
        
        # For each vertex, compute distance to non-adjacent edges
        # Skip adjacent edges (i-1, i), (i, i+1)
        total_barrier = torch.tensor(0.0, device=cage.device)
        
        for i in range(K):
            # Get vertex i
            v_i = cage[:, i:i+1, :]  # (B, 1, 2)
            
            # Get all non-adjacent edges
            # Edge j is from cage[:, j] to cage[:, j+1]
            # Adjacent to vertex i are edges (i-1, i) and (i, i+1)
            adjacent = [(i - 1) % K, i]
            
            for j in range(K):
                if j in adjacent:
                    continue
                
                seg_start = cage[:, j:j+1, :]  # (B, 1, 2)
                seg_end = cage_next[:, j:j+1, :]  # (B, 1, 2)
                
                # Point-to-segment distance
                dist_sq = utils.point_to_segment_distance_sq(
                    v_i.unsqueeze(2),  # (B, 1, 1, 2)
                    seg_start.unsqueeze(1),  # (B, 1, 1, 2)
                    seg_end.unsqueeze(1)  # (B, 1, 1, 2)
                )  # (B, 1, 1)
                
                dist = torch.sqrt(dist_sq.squeeze() + 1e-8)  # (B,)
                
                # Log barrier: -log(d - d_hat) if d < 2*d_hat, else 0
                # This creates a soft barrier that increases as distance decreases
                barrier_active = (dist < 2 * d_hat).float()
                barrier_val = barrier_active * (-torch.log(torch.clamp(dist - d_hat, min=1e-8)))
                total_barrier = total_barrier + barrier_val.mean()
        
        return total_barrier / (K * (K - 3))  # Normalize by number of pairs