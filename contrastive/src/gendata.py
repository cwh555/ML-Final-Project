from PIL import Image, ImageDraw, ImageFilter
import random
import string
import numpy as np
import argparse
import os
from collections import deque

def get_largest_component(mask_img, threshold):
    """
    Finds the largest connected component of white pixels in the mask.
    Returns a set of (x, y) coordinates belonging to that largest component.
    """
    width, height = mask_img.size
    pixels = mask_img.load()
    visited = set()
    largest_component = set()
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for y in range(height):
        for x in range(width):
            if pixels[x, y] > threshold and (x, y) not in visited:
                current_component = set()
                q = deque([(x, y)])
                visited.add((x, y))
                current_component.add((x, y))
                
                while q:
                    cx, cy = q.popleft()
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if (nx, ny) not in visited and pixels[nx, ny] > threshold:
                                visited.add((nx, ny))
                                current_component.add((nx, ny))
                                q.append((nx, ny))
                
                if len(current_component) > len(largest_component):
                    largest_component = current_component
                    
    return largest_component

def generate_gradient_image(width, height):
    """
    Generates a random smooth linear gradient image (H, W, 3).
    """
    # Pick two random colors
    c1 = np.array([random.randint(50, 255) for _ in range(3)])
    c2 = np.array([random.randint(50, 255) for _ in range(3)])
    
    # Create coordinate grid
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Random angle for the gradient
    angle = random.uniform(0, 2 * np.pi)
    
    # Calculate blend factor based on angle
    blend = X * np.cos(angle) + Y * np.sin(angle)
    
    # Normalize blend to 0.0 - 1.0 range
    blend = (blend - blend.min()) / (blend.max() - blend.min())
    
    # Expand dims for broadcasting: (H, W, 1)
    blend = blend[:, :, np.newaxis]
    
    # Linear interpolation between c1 and c2
    gradient = c1 * (1 - blend) + c2 * blend
    
    return gradient.astype(np.uint8)

def generate_base_shape_data():
    """
    Generates a base shape and returns the PIL Image (Color) and PIL Mask.
    """
    WIDTH, HEIGHT = 256, 256
    CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
    MIN_PIXELS = 4000
    
    print("Generating base shape...")

    while True:
        # 1. Create a grayscale mask
        mask = Image.new('L', (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(mask)
        
        # 2. Draw Positive Blobs
        con = random.choice([(10,30),(20,50)])
        num_blobs = random.randint(15, 25)
        for _ in range(num_blobs):
            radius = random.randint(con[0],con[1])
            offset_range = 50 
            x = CENTER_X + random.randint(-offset_range, offset_range)
            y = CENTER_Y + random.randint(-offset_range, offset_range)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)

        # 3. Draw Negative Blobs (Holes)
        num_holes = random.randint(20, 30)
        con2 = random.choice([(5,15),(10,30)])
        for _ in range(num_holes):
            radius = random.randint(con2[0],con2[1])
            offset_range = 40 
            x = CENTER_X + random.randint(-offset_range, offset_range)
            y = CENTER_Y + random.randint(-offset_range, offset_range)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=0)

        # 4. Blur
        blur_radius = random.choice([5,30])
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 5. Analyze Components
        threshold = 100 
        largest_component_pixels = get_largest_component(mask, threshold)
        
        # 6. Check constraints
        if len(largest_component_pixels) >= MIN_PIXELS:
            print(f"Valid shape found! Size: {len(largest_component_pixels)} pixels.")
            
            # --- APPLY COLOR GRADIENT ---
            
            # 1. Create Binary Mask Array (H, W)
            mask_array = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            for (x, y) in largest_component_pixels:
                mask_array[y, x] = 1
                
            # 2. Create Continuous Color Gradient (H, W, 3)
            gradient_layer = generate_gradient_image(WIDTH, HEIGHT)
            
            # 3. Apply Mask to Gradient (Black background)
            final_image_hwc = gradient_layer * mask_array[:, :, np.newaxis]
            
            # 4. Convert to PIL for the upcoming rotation step
            base_img = Image.fromarray(final_image_hwc, 'RGB')
            base_mask = Image.fromarray(mask_array * 255, 'L')
            
            return base_img, base_mask

def generate_dataset_batch(output_dir):
    # 1. Generate the original shape once (with color)
    base_img, base_mask = generate_base_shape_data()
    
    # 2. Generate Random Identifiers for this batch
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    batch_label = random.randint(0, 10000000000) 
    
    print(f"Processing batch: ID={random_str}, Label={batch_label}")

    # 3. Generate 10 variations
    for i in range(1):
        # Random rotation between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        
        # Rotate Image (Bicubic for smoothness)
        # expand=False keeps it 256x256 
        img_rot = base_img.rotate(angle, resample=Image.BICUBIC)
        
        # Rotate Mask (Nearest to keep edges sharp/binary)
        mask_rot = base_mask.rotate(angle, resample=Image.NEAREST)
        
        # --- Convert to Numpy & Format ---
        
        # Image: (3, 256, 256), float 0-1
        img_arr = np.array(img_rot).astype(np.float32) / 255.0
        img_arr = img_arr.transpose(2, 0, 1) # HWC to CHW
        
        # Mask: (256, 256), int 0 or 1
        mask_arr = np.array(mask_rot)
        mask_arr = (mask_arr > 128).astype(np.uint8) 
        
        # --- Save ---
        filename = f"{random_str}_{i}.npy"
        filepath = filepath = os.path.join(output_dir, filename); 
        # Create the pair (image, mask, label)
        data_to_save = (img_arr, mask_arr, batch_label)
        
        np.save(filepath, np.array(data_to_save, dtype=object))
        #print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()
    for _ in range(100):
        generate_dataset_batch(args.output_dir)

