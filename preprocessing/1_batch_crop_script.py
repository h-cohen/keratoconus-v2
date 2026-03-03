#!/usr/bin/env python3
"""
Batch cropping script for keratoconus tomography images.
Processes all JPG files in data/image_scans/ and creates 4 circular crops per image.
Applies circular masking with black background to remove unwanted text.
Output format: JPG
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

def create_output_directory(output_dir="cropped_output"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir

def get_base_filename(filepath):
    """Extract base filename without extension."""
    return Path(filepath).stem

def apply_circular_mask(image, transparent=True):
    """
    Apply a circular mask to the image, keeping only the circular area.
    
    Args:
        image (numpy.ndarray): Input image
        transparent (bool): If True, use transparent background; if False, use black
    
    Returns:
        numpy.ndarray: Image with circular mask applied (BGRA if transparent)
    """
    height, width = image.shape[:2]
    
    # Use the smaller dimension to ensure the circle fits
    diameter = min(width, height)
    radius = diameter // 2
    
    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(mask, center, radius, 255, -1)
    
    if transparent:
        # Convert to BGRA (with alpha channel for transparency)
        if len(image.shape) == 3:
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            result = image.copy()
        
        # Set alpha channel based on mask (0 = transparent, 255 = opaque)
        result[:, :, 3] = mask
    else:
        # Use black background
        result = image.copy()
        result[mask == 0] = (0, 0, 0)
    
    return result

def crop_single_image(image_path, output_dir):
    """
    Crop a single image into 4 maps and save them.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save cropped images
    
    Returns:
        list: List of created output files
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return []
    
    # Get base filename for output naming
    base_name = get_base_filename(image_path)
    
    # Define crop coordinates for each map (x, y, width, height)
    crops = {
        'corneal_thickness': (448, 98, 290, 290),
        'curvature_front': (822, 98, 290, 290),
        'elevation_front': (448, 496, 290, 290),
        'elevation_back': (822, 496, 290, 290)
    }
    
    created_files = []
    
    # Crop and save each map
    for crop_name, (x, y, w, h) in crops.items():
        # Check if coordinates are within image bounds
        if y + h > image.shape[0] or x + w > image.shape[1]:
            print(f"Warning: Crop {crop_name} exceeds image bounds for {image_path}")
            continue
            
        # Initial rectangular crop
        cropped = image[y:y+h, x:x+w]
        
        # Apply circular mask with black background
        circular_cropped = apply_circular_mask(cropped, transparent=False)
        
        output_filename = f"{base_name}_{crop_name}.jpg".lower()
        output_path = os.path.join(output_dir, output_filename)
        
        success = cv2.imwrite(output_path, circular_cropped)
        if success:
            created_files.append(output_path)
            print(f"✓ Saved: {output_filename} (with black circular mask)")
        else:
            print(f"✗ Failed to save: {output_filename}")
    
    return created_files

def batch_crop_images(input_pattern="data/image_scans/*.JPG", output_dir="cropped_output"):
    """
    Process all JPG files matching the input pattern.
    
    Args:
        input_pattern (str): Glob pattern for input files
        output_dir (str): Directory to save cropped images
    
    Returns:
        dict: Summary of processing results
    """
    # Create output directory
    output_dir = create_output_directory(output_dir)
    
    # Find all matching files
    image_files = glob.glob(input_pattern)
    
    if not image_files:
        print(f"No files found matching pattern: {input_pattern}")
        return {"processed": 0, "total_crops": 0, "failed": 0}
    
    print(f"Found {len(image_files)} images to process...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    processed = 0
    total_crops = 0
    failed = 0
    
    for image_path in image_files:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        try:
            created_files = crop_single_image(image_path, output_dir)
            if created_files:
                processed += 1
                total_crops += len(created_files)
                print(f"  → Created {len(created_files)} crops")
            else:
                failed += 1
                print(f"  → Failed to process")
        except Exception as e:
            failed += 1
            print(f"  → Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Images processed: {processed}")
    print(f"Total crops created: {total_crops}")
    print(f"Failed images: {failed}")
    print(f"Output directory: {output_dir}")
    
    return {
        "processed": processed,
        "total_crops": total_crops,
        "failed": failed,
        "output_dir": output_dir
    }

def test_single_file(test_file="data/image_scans/Aasi_Radi_OD_08052014_085912_4 Maps Select.JPG"):
    """
    Test cropping on a single file for verification.
    
    Args:
        test_file (str): Path to test file
    """
    print("TESTING SINGLE FILE")
    print("=" * 30)
    print(f"Test file: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return
    
    output_dir = create_output_directory("test_crop_output")
    created_files = crop_single_image(test_file, output_dir)
    
    print(f"\nTest completed!")
    print(f"Created {len(created_files)} crop files in {output_dir}/")
    
    if created_files:
        print("\nCreated files:")
        for file_path in created_files:
            print(f"  - {os.path.basename(file_path)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - process single file
        test_single_file()
    else:
        # Full batch processing
        batch_crop_images()
