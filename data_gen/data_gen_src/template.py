import os
import argparse
import numpy as np
import cv2
import math

# --- Base Utility Functions ---

def get_base_canvas(size=256):
    """Creates a black 256x256 float32 canvas."""
    return np.zeros((size, size), dtype=np.float32)

def save_template(mask, name, output_dir):
    """Saves the mask as a compressed .npz file if it doesn't exist."""
    filename = f"{name}.npz"
    path = os.path.join(output_dir, filename)
    
    if os.path.exists(path):
        print(f"Skipping {name}: File already exists at {path}")
        return
    
    # Save as .npz with key 'mask' to match dataset loader expectations
    np.savez_compressed(path, mask=mask)
    print(f"Generated {name}: Saved to {path}")

# --- Original Template Functions ---

def create_circle(size=256, margin=10):
    """Generates a large filled circle."""
    mask = get_base_canvas(size)
    center = (size // 2, size // 2)
    # Radius is half size minus margin
    radius = (size // 2) - margin
    cv2.circle(mask, center, radius, 1.0, -1)
    return mask

def create_rectangle(size=256, margin=10):
    """Generates a large filled rectangle."""
    mask = get_base_canvas(size)
    # Top-Left and Bottom-Right coordinates
    pt1 = (margin, margin)
    pt2 = (size - margin, size - margin)
    cv2.rectangle(mask, pt1, pt2, 1.0, -1)
    return mask

def create_star(size=256, margin=10, points=5, inner_ratio=0.4):
    """
    Generates a large filled star (5 tips).
    [修正]: 角度計算，確保第一個尖角朝上。
    """
    mask = get_base_canvas(size)
    center = (size // 2, size // 2)
    outer_radius = (size // 2) - margin
    inner_radius = int(outer_radius * inner_ratio)
    
    vertices = []
    angle_step = np.pi / points # Half step for inner/outer alternation
    
    # [修正]: Start angle should be -np.pi / 2 (or 270 deg) for the tip to point up.
    current_angle = -np.pi / 2 
    
    for i in range(points * 2):
        # Even indices are outer points, Odd indices are inner points
        r = outer_radius if i % 2 == 0 else inner_radius
        
        x = center[0] + int(r * np.cos(current_angle))
        y = center[1] + int(r * np.sin(current_angle))
        vertices.append((x, y))
        
        current_angle += angle_step
        
    vertices = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 1.0)
    return mask

# --- New Template Functions (For Stability Test) ---

def create_triangle(size=256, margin=10):
    """Generates an equilateral triangle with a flat base."""
    mask = get_base_canvas(size)
    center_x = size // 2
    height = size - 2 * margin
    
    # Calculate vertices of an equilateral triangle
    # Top vertex (pointing up): (x, y)
    y_top = margin
    
    # Base vertices
    side_length = height / (math.sqrt(3) / 2)
    half_base = side_length / 2
    
    y_base = margin + height
    x_left = int(center_x - half_base)
    x_right = int(center_x + half_base)

    vertices = [
        (center_x, y_top),
        (x_left, y_base),
        (x_right, y_base)
    ]
    
    vertices = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 1.0)
    return mask

def create_rounded_rect(size=256, margin=10, corner_radius_ratio=0.1):
    """Generates a filled rectangle with rounded corners (Capsule shape)."""
    mask = get_base_canvas(size)
    rect_side = size - 2 * margin
    
    # Determine the corner radius based on the shorter side or a ratio
    corner_radius = int(rect_side * corner_radius_ratio)
    
    # Use cv2.rectangle combined with cv2.circle for rounded corners
    # Draw the main rectangular body (two vertical rects and one horizontal rect)
    
    # The center rectangular area
    pt1_center = (margin + corner_radius, margin)
    pt2_center = (size - margin - corner_radius, size - margin)
    cv2.rectangle(mask, pt1_center, pt2_center, 1.0, -1)
    
    # The two end rectangular areas (to cover the remaining space)
    pt1_end = (margin, margin + corner_radius)
    pt2_end = (size - margin, size - margin - corner_radius)
    cv2.rectangle(mask, pt1_end, pt2_end, 1.0, -1)
    
    # Draw circles at the four corners
    radius = corner_radius
    
    # Top-Left corner
    cv2.circle(mask, (margin + radius, margin + radius), radius, 1.0, -1)
    # Top-Right corner
    cv2.circle(mask, (size - margin - radius, margin + radius), radius, 1.0, -1)
    # Bottom-Left corner
    cv2.circle(mask, (margin + radius, size - margin - radius), radius, 1.0, -1)
    # Bottom-Right corner
    cv2.circle(mask, (size - margin - radius, size - margin - radius), radius, 1.0, -1)
    
    return mask

def create_annulus(size=256, margin=10, inner_ratio=0.5):
    """Generates a filled ring/annulus (Doughnut)."""
    mask = get_base_canvas(size)
    center = (size // 2, size // 2)
    
    outer_radius = (size // 2) - margin
    inner_radius = int(outer_radius * inner_ratio)
    
    # Draw the outer filled circle (1.0)
    cv2.circle(mask, center, outer_radius, 1.0, -1)
    
    # Draw the inner black circle (0.0)
    cv2.circle(mask, center, inner_radius, 0.0, -1)
    
    return mask
    
def create_ellipse(size=256, margin=10, ratio=0.7, orientation='vertical'):
    """
    Generates a filled ellipse (oval shape).
    
    Args:
        size (int): Canvas size.
        margin (int): Margin from canvas edge.
        ratio (float): Ratio of minor axis to major axis (e.g., 0.7 means 70% width/height).
        orientation (str): 'vertical' (taller) or 'horizontal' (wider).
    """
    mask = get_base_canvas(size)
    center = (size // 2, size // 2)
    
    # Calculate major axis radius based on size and margin
    major_radius = (size // 2) - margin
    minor_radius = int(major_radius * ratio)
    
    # Determine axes based on orientation
    if orientation == 'vertical':
        # Vertical ellipse: height (y-axis) is major, width (x-axis) is minor
        axes = (minor_radius, major_radius)
    elif orientation == 'horizontal':
        # Horizontal ellipse: width (x-axis) is major, height (y-axis) is minor
        axes = (major_radius, minor_radius)
    else:
        # Fallback to horizontal if invalid input
        print(f"Warning: Invalid orientation '{orientation}'. Defaulting to 'horizontal'.")
        axes = (major_radius, minor_radius)
        
    # Draw the filled ellipse (angle=0, startAngle=0, endAngle=360, thickness=-1)
    # axes: (half_width, half_height)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    
    return mask

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Generate binary shape templates for Cage Training.")
    parser.add_argument("--template_dir", type=str, default="./template", help="Directory to store templates")
    parser.add_argument("--size", type=int, default=256, help="Image resolution")
    args = parser.parse_args()
    
    # Create directory if not exists
    os.makedirs(args.template_dir, exist_ok=True)
    
    print(f"Generating templates in: {args.template_dir}...")
    
    # 1. Base Shapes
    save_template(create_circle(size=args.size), "circle", args.template_dir)
    save_template(create_rectangle(size=args.size), "rectangle", args.template_dir)
    save_template(create_star(size=args.size), "star", args.template_dir)
    
    # 2. Stability Test Shapes (New)
    print("-" * 20)
    # Simple Sharp Corner Test
    save_template(create_triangle(size=args.size), "triangle", args.template_dir)
    
    # Smooth Transition Test
    save_template(create_rounded_rect(size=args.size), "rounded_rect", args.template_dir)
    
    # Topology/Hole Test
    save_template(create_annulus(size=args.size), "annulus", args.template_dir)
    
    # --- Ellipse Test ---
    # Vertical Ellipse (Taller)
    save_template(create_ellipse(size=args.size, orientation='vertical'), "ellipse_v", args.template_dir)
    # Horizontal Ellipse (Wider)
    save_template(create_ellipse(size=args.size, orientation='horizontal'), "ellipse_h", args.template_dir)
    
    print("Done. Please ensure your dataset loading script is updated to include these new shapes.")

if __name__ == "__main__":
    main()