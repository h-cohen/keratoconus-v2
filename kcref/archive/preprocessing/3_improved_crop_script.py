#!/usr/bin/env python3
"""
Improved batch cropping script for keratoconus tomography images.
Fixes issues with the original cropping:
1. Tighter crop to exclude tick marks and labels
2. Smaller circular mask to remove edge artifacts
3. Option to mask out the OD/OS text marker
4. Better handling of the actual map data region
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path


def create_output_directory(output_dir="data/cropped_output_v2"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir


def get_base_filename(filepath):
    """Extract base filename without extension."""
    return Path(filepath).stem


def apply_tight_circular_mask(image, radius_ratio=0.88):
    """
    Apply a tighter circular mask to exclude edge artifacts.
    
    Args:
        image: Input image
        radius_ratio: Ratio of mask radius to image half-width (0.88 excludes tick marks)
    
    Returns:
        Masked image with black background outside the circle
    """
    height, width = image.shape[:2]
    
    # Use a smaller radius to exclude tick marks
    diameter = min(width, height)
    radius = int((diameter // 2) * radius_ratio)
    
    # Create circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask
    result = image.copy()
    if len(result.shape) == 3:
        result[mask == 0] = [0, 0, 0]
    else:
        result[mask == 0] = 0
    
    return result, mask


def mask_od_os_text(image, eye_indicator='OD'):
    """
    Mask out the OD/OS text that appears on the map.
    The text typically appears in the upper-right quadrant of each map.
    
    Args:
        image: Input image
        eye_indicator: 'OD' or 'OS'
    
    Returns:
        Image with text region masked (filled with interpolated values)
    """
    height, width = image.shape[:2]
    result = image.copy()
    
    # The OD/OS text is typically in the upper right area
    # Approximate text region (may need adjustment)
    text_region = {
        'x': int(width * 0.55),
        'y': int(height * 0.08),
        'w': int(width * 0.25),
        'h': int(height * 0.12)
    }
    
    # Instead of just blacking out, we can try to interpolate from surroundings
    # For now, just mask with the average color of surrounding region
    x, y, w, h = text_region['x'], text_region['y'], text_region['w'], text_region['h']
    
    # Get surrounding pixels (expand region slightly)
    margin = 5
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(width, x + w + margin), min(height, y + h + margin)
    
    surrounding = result[y1:y2, x1:x2].copy()
    
    # Create mask for the text area within the surrounding region
    text_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    text_mask[margin:margin+h, margin:margin+w] = 255
    
    # Use inpainting to fill the text region
    if len(result.shape) == 3:
        inpainted = cv2.inpaint(surrounding, text_mask, 3, cv2.INPAINT_TELEA)
        result[y1:y2, x1:x2] = inpainted
    
    return result


def extract_inner_region(image, inner_ratio=0.85):
    """
    Extract only the inner region of the map, excluding the edge tick marks.
    
    Args:
        image: Input image (should be square)
        inner_ratio: Ratio of inner region to keep (0.85 = 85% of the diameter)
    
    Returns:
        Cropped and resized image containing only the inner map data
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Calculate inner region bounds
    inner_radius = int(min(width, height) // 2 * inner_ratio)
    
    x1 = center_x - inner_radius
    x2 = center_x + inner_radius
    y1 = center_y - inner_radius
    y2 = center_y + inner_radius
    
    # Ensure bounds are within image
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    # Crop
    inner = image[y1:y2, x1:x2]
    
    return inner


def crop_single_image_improved(image_path, output_dir, apply_text_mask=True, 
                                extract_inner=True, inner_ratio=0.85):
    """
    Improved cropping with better handling of artifacts.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        apply_text_mask: Whether to mask the OD/OS text
        extract_inner: Whether to extract only the inner region
        inner_ratio: Ratio for inner region extraction
    
    Returns:
        List of created files
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return []
    
    base_name = get_base_filename(image_path)
    
    # Detect eye indicator from filename
    eye_indicator = 'OD' if '_OD_' in image_path.upper() else 'OS'
    
    # Improved crop coordinates - slightly adjusted to better center on the maps
    # These coordinates are for 1200x910 images from Pentacam
    crops = {
        'corneal_thickness': (448, 98, 290, 290),
        'curvature_front': (822, 98, 290, 290),
        'elevation_front': (448, 496, 290, 290),
        'elevation_back': (822, 496, 290, 290)
    }
    
    created_files = []
    
    for crop_name, (x, y, w, h) in crops.items():
        if y + h > image.shape[0] or x + w > image.shape[1]:
            print(f"Warning: Crop {crop_name} exceeds image bounds")
            continue
        
        # Initial crop
        cropped = image[y:y+h, x:x+w].copy()
        
        # Step 1: Extract inner region (removes tick marks)
        if extract_inner:
            cropped = extract_inner_region(cropped, inner_ratio)
        
        # Step 2: Mask OD/OS text
        if apply_text_mask:
            cropped = mask_od_os_text(cropped, eye_indicator)
        
        # Step 3: Apply circular mask (tighter than before)
        cropped, _ = apply_tight_circular_mask(cropped, radius_ratio=0.95)
        
        # Step 4: Resize to standard size
        cropped = cv2.resize(cropped, (256, 256))
        
        # Save
        output_filename = f"{base_name}_{crop_name}.jpg".lower()
        output_path = os.path.join(output_dir, output_filename)
        
        if cv2.imwrite(output_path, cropped):
            created_files.append(output_path)
            print(f"✓ {output_filename}")
        else:
            print(f"✗ Failed: {output_filename}")
    
    return created_files


def analyze_crop_quality(image_path):
    """
    Analyze a cropped image to check for artifacts.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Count black pixels (masked area)
    black_pixels = np.sum(img.sum(axis=2) == 0)
    total_pixels = img.shape[0] * img.shape[1]
    
    # Check for very dark pixels (potential tick marks)
    very_dark = np.sum(np.all(img < 20, axis=2))
    
    # Check variance (should be high for actual map data)
    center = gray[gray.shape[0]//4:3*gray.shape[0]//4, 
                  gray.shape[1]//4:3*gray.shape[1]//4]
    center_variance = np.var(center)
    
    return {
        'black_pixel_ratio': black_pixels / total_pixels,
        'very_dark_ratio': very_dark / total_pixels,
        'center_variance': center_variance,
        'mean_intensity': gray.mean()
    }


def batch_crop_improved(input_pattern="data/image_scans/*.JPG", 
                        output_dir="data/cropped_output_v2"):
    """
    Process all images with improved cropping.
    """
    output_dir = create_output_directory(output_dir)
    image_files = glob.glob(input_pattern)
    
    if not image_files:
        print(f"No files found: {input_pattern}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Output: {output_dir}")
    print("-" * 50)
    
    processed = 0
    for image_path in image_files:
        print(f"\n{os.path.basename(image_path)}")
        try:
            files = crop_single_image_improved(image_path, output_dir)
            if files:
                processed += 1
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Processed: {processed}/{len(image_files)} images")


def compare_old_vs_new(old_dir="cropped_output", new_dir="data/cropped_output_v2"):
    """
    Compare quality metrics between old and new cropping.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Find matching files
    old_files = glob.glob(f"{old_dir}/*corneal_thickness*.jpg")[:5]
    
    fig, axes = plt.subplots(len(old_files), 3, figsize=(12, 4*len(old_files)))
    
    for i, old_path in enumerate(old_files):
        base = os.path.basename(old_path)
        new_path = os.path.join(new_dir, base)
        
        old_img = cv2.imread(old_path)
        new_img = cv2.imread(new_path) if os.path.exists(new_path) else None
        
        # Show old
        if old_img is not None:
            axes[i, 0].imshow(cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f'Old crop\nBlack: {100*(old_img.sum(axis=2)==0).mean():.1f}%')
        axes[i, 0].axis('off')
        
        # Show new
        if new_img is not None:
            axes[i, 1].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title(f'New crop\nBlack: {100*(new_img.sum(axis=2)==0).mean():.1f}%')
        axes[i, 1].axis('off')
        
        # Show difference
        if old_img is not None and new_img is not None:
            # Resize old to match new if needed
            if old_img.shape != new_img.shape:
                old_resized = cv2.resize(old_img, (new_img.shape[1], new_img.shape[0]))
            else:
                old_resized = old_img
            diff = cv2.absdiff(old_resized, new_img)
            axes[i, 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title('Difference')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('crop_comparison.png', dpi=150)
    print("Saved crop_comparison.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test on a single file
        test_file = "data/image_scans/Aasi_Radi_OD_08052014_085912_4 Maps Select.JPG"
        output_dir = create_output_directory("test_crop_v2")
        crop_single_image_improved(test_file, output_dir)
        
        # Analyze quality
        print("\nQuality analysis:")
        for f in glob.glob(f"{output_dir}/*.jpg"):
            metrics = analyze_crop_quality(f)
            print(f"  {os.path.basename(f)}: {metrics}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_old_vs_new()
    
    else:
        batch_crop_improved()





















