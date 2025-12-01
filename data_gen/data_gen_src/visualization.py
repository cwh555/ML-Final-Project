import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

def visualize_npy_files():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize generated .npy shape files.")
    parser.add_argument("filename", nargs="?", default="*.npy", help="Specific filename or pattern (e.g., 'data_0.npy' or '*.npy')")
    args = parser.parse_args()

    # 2. Find files based on the argument
    # Use glob to handle wildcards or exact filenames
    files = sorted(glob.glob(args.filename))
    
    if not files:
        print(f"No files found matching pattern: '{args.filename}'")
        return

    # Take up to 10 files to visualize
    files_to_show = files[:10]
    count = len(files_to_show)
    
    print(f"Found {len(files)} files matching '{args.filename}'. Visualizing {count}...")

    # 3. Setup Matplotlib Figure
    # 2 Rows: Top for Image, Bottom for Mask
    # 'count' Columns
    # Adjust figsize based on number of images to keep aspect ratio reasonable
    fig, axes = plt.subplots(2, count, figsize=(max(count * 2.5, 5), 5))
    
    # Handle edge case if only 1 file exists (axes wouldn't be 2D array)
    if count == 1:
        # Reshape to (2, 1) so standard indexing axes[row, col] works
        axes = np.array([[axes[0]], [axes[1]]])

    for i, filename in enumerate(files_to_show):
        try:
            # 4. Load Data
            # allow_pickle=True is required because the npy contains an object array (tuple)
            data = np.load(filename, allow_pickle=True)
            
            # The saved structure is a numpy array of objects with shape (3,) 
            # representing (image, mask, label)
            img_arr, mask_arr, label = data
            
            # 5. Process Image for Display
            # Saved format: (3, 256, 256) -> CHW
            # Matplotlib format: (256, 256, 3) -> HWC
            img_display = img_arr.transpose(1, 2, 0)
            
            # 6. Plot Image (Top Row)
            ax_img = axes[0, i]
            ax_img.imshow(img_display)
            # Parse file index from name like 'abc123_0.npy'
            idx = os.path.basename(filename).replace('.npy', '')
            ax_img.set_title(f"Label: {label}\n{idx}", fontsize=9)
            ax_img.axis('off')
            
            # 7. Plot Mask (Bottom Row)
            ax_mask = axes[1, i]
            ax_mask.imshow(mask_arr, cmap='gray')
            ax_mask.set_title("Mask", fontsize=8)
            ax_mask.axis('off')
            
            # Print info to console
            if i == 0:
                print(f"Data loaded example ({filename}):")
                print(f" - Image Shape: {img_arr.shape}, Range: [{img_arr.min():.2f}, {img_arr.max():.2f}]")
                print(f" - Mask Shape:  {mask_arr.shape}, Unique Values: {np.unique(mask_arr)}")
                print(f" - Label:       {label}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_npy_files()
