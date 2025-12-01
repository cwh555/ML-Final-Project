"""
Geometric Utilities for Cage-based Deformation
Based on:
- Mean Value Coordinates (Floater et al., 2003)
- IPC Barrier Function (Li et al., SIGGRAPH 2020)
"""

import torch
import torch.nn.functional as F
import numpy as np

def generate_circular_cage(num_vertices, radius=1.2, device='cpu'):
    """
    Generate a standard circular cage centered at origin.
    
    Args:
        num_vertices: Number of cage vertices (K)
        radius: Cage radius (should be > 1.0 to enclose unit square)
        device: 'cpu' or 'cuda'
    
    Returns:
        cage: (K, 2) tensor
    """
    theta = torch.linspace(0, 2 * np.pi, num_vertices + 1, device=device)[:-1]
    # Counter-clockwise for positive MVC weights
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    return torch.stack([x, y], dim=1)

def compute_mvc_weights(points, cage):
    """
    Compute Mean Value Coordinates for points w.r.t. cage.
    Reference: "Mean Value Coordinates" (Floater et al., 2003)
    
    Standard MVC formula:
        w_i = (tan(alpha_{i-1}/2) + tan(alpha_i/2)) / r_i
    where:
        alpha_i = angle at point p between v_i and v_{i+1}
        r_i = ||v_i|| = distance from point to cage vertex i
        tan(alpha/2) = (r_i * r_{i+1} - dot(v_i, v_{i+1})) / cross(v_i, v_{i+1})
    
    Args:
        points: (B, N, 2) - Query points (can be boundary or interior)
        cage:   (B, K, 2) - Cage vertices (counter-clockwise order)
    
    Returns:
        weights: (B, N, K) - Normalized MVC weights (sum to 1)
    """
    B, N, _ = points.shape
    K = cage.shape[1]
    
    # Vector from point to each cage vertex: v_i = cage_i - point
    # Shape: (B, N, K, 2)
    v = cage.unsqueeze(1) - points.unsqueeze(2)
    
    # Distance from point to each cage vertex
    # Shape: (B, N, K)
    r = torch.norm(v, dim=-1, p=2).clamp(min=1e-8)
    
    # Get v_{i+1} (next vertex)
    v_next = torch.roll(v, shifts=-1, dims=2)  # (B, N, K, 2)
    r_next = torch.roll(r, shifts=-1, dims=2)  # (B, N, K)
    
    # Cross product (2D): v_i x v_{i+1} = v_i.x * v_{i+1}.y - v_i.y * v_{i+1}.x
    # This gives signed area of parallelogram, positive for CCW
    cross = v[..., 0] * v_next[..., 1] - v[..., 1] * v_next[..., 0]  # (B, N, K)
    
    # Dot product: v_i · v_{i+1}
    dot = torch.sum(v * v_next, dim=-1)  # (B, N, K)
    
    # tan(alpha_i / 2) where alpha_i is angle between v_i and v_{i+1}
    # Using half-angle formula: tan(alpha/2) = (r_i * r_{i+1} - dot) / cross
    # Add small epsilon to cross to avoid division by zero
    tan_half_alpha = (r * r_next - dot) / (cross + 1e-8 * torch.sign(cross + 1e-10))
    
    # MVC weight for vertex i: w_i = (tan(alpha_{i-1}/2) + tan(alpha_i/2)) / r_i
    # tan_half_alpha[i] corresponds to angle between v_i and v_{i+1}
    # So we need tan_half_alpha[i-1] + tan_half_alpha[i]
    tan_half_alpha_prev = torch.roll(tan_half_alpha, shifts=1, dims=2)
    
    # Unnormalized weights
    weights = (tan_half_alpha_prev + tan_half_alpha) / r
    
    # Handle numerical issues: clamp very negative weights
    # For points inside a convex cage, all weights should be positive
    # For points outside or near boundary, some weights may be negative
    
    # Normalize to sum to 1
    w_sum = torch.sum(weights, dim=-1, keepdim=True)
    weights = weights / (w_sum + 1e-8)
    
    return weights

def apply_affine(points, matrix):
    """
    Apply affine transformation to points.
    
    Args:
        points: (B, N, 2)
        matrix: (B, 2, 3) - Affine transformation matrix
    
    Returns:
        points_transformed: (B, N, 2)
    """
    B, N, _ = points.shape
    ones = torch.ones(B, N, 1, device=points.device)
    points_homo = torch.cat([points, ones], dim=2)  # (B, N, 3)
    return torch.bmm(points_homo, matrix.transpose(1, 2))

def point_to_segment_distance_sq(points, seg_start, seg_end):
    """
    Compute squared distance from points to line segments.
    Used for IPC barrier function.
    
    Args:
        points: (B, N_p, 1, 2)
        seg_start: (B, 1, N_s, 2)
        seg_end: (B, 1, N_s, 2)
    
    Returns:
        dist_sq: (B, N_p, N_s) - Squared distances
    """
    # Edge vector
    edge_vec = seg_end - seg_start  # (B, 1, N_s, 2)
    
    # Vector from segment start to point
    point_vec = points - seg_start  # (B, N_p, N_s, 2)
    
    # Projection parameter t = dot(point_vec, edge_vec) / dot(edge_vec, edge_vec)
    edge_len_sq = torch.sum(edge_vec ** 2, dim=-1, keepdim=True) + 1e-8
    t = torch.sum(point_vec * edge_vec, dim=-1, keepdim=True) / edge_len_sq
    
    # Clamp t to segment [0, 1]
    t = torch.clamp(t, 0.0, 1.0)
    
    # Closest point on segment
    closest = seg_start + t * edge_vec
    
    # Distance
    dist_sq = torch.sum((points - closest) ** 2, dim=-1)
    return dist_sq

def compute_polygon_area(points):
    """
    Compute the area of a polygon using the Shoelace formula.
    Useful for preventing the cage from collapsing to zero area.
    
    Args:
        points: (B, K, 2) - Closed polygon vertices (ordered)
    Returns:
        area: (B,)
    """
    x = points[..., 0]
    y = points[..., 1]
    
    # Shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    # Roll to get the next vertex
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    
    area = 0.5 * torch.abs(torch.sum(x * y_next - x_next * y, dim=1))
    return area