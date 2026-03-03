#!/usr/bin/env python3
"""
Extract Belin/Ambrósio Enhanced Ectasia Display (BAD) values from Pentacam DICOM files.

The BAD-D and related indices are critical for keratoconus detection/progression
but are often not included in standard Pentacam data exports.

This script extracts them from the encapsulated PDFs in DICOM files.
"""

import pydicom
import numpy as np
import pandas as pd
import re
import os
import glob
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Try to import PDF processing libraries
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pdf2image or pytesseract not available. Install with:")
    print("  pip install pdf2image pytesseract")
    print("  Also install Tesseract OCR: apt-get install tesseract-ocr")


def extract_pdf_from_dicom(dcm_path):
    """Extract encapsulated PDF from DICOM file."""
    ds = pydicom.dcmread(dcm_path)
    
    if hasattr(ds, 'EncapsulatedDocument'):
        return ds.EncapsulatedDocument
    return None


def extract_bad_values_from_image(image):
    """
    Use OCR to extract BAD values from the Belin/Ambrósio page.
    
    Target values:
    - D: (BAD-D total score)
    - Df, Db, Dp, Dt, Da (deviation components)
    - ARTmax
    - Progression Index
    """
    if not TESSERACT_AVAILABLE:
        return {}
    
    # Convert PIL image to grayscale for better OCR
    gray = image.convert('L')
    
    # OCR the image
    text = pytesseract.image_to_string(gray)
    
    # Parse the text for BAD values
    values = {}
    
    # Look for D: value (BAD-D score) - format "D: X.XX"
    d_match = re.search(r'D:\s*(\d+\.?\d*)', text)
    if d_match:
        values['BAD_D'] = float(d_match.group(1))
    
    # Look for individual deviation values
    for idx_name in ['Df', 'Db', 'Dp', 'Dt', 'Da']:
        # Pattern: "Df: X.XX" or "Df 1.22"
        pattern = rf'{idx_name}[:\s]+(\d+\.?\d*)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            values[f'BAD_{idx_name}'] = float(match.group(1))
    
    # Look for ARTmax
    art_match = re.search(r'ART\s*max[:\s]*(\d+)', text, re.IGNORECASE)
    if art_match:
        values['ARTmax'] = float(art_match.group(1))
    
    # Look for Progression Index values
    prog_match = re.search(r'Progression\s*Index[:\s]*\n?\s*Min[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    if prog_match:
        values['ProgIdx_Min'] = float(prog_match.group(1))
    
    prog_max_match = re.search(r'Max[:\s]*(\d+\.?\d*)', text)
    if prog_max_match:
        values['ProgIdx_Max'] = float(prog_max_match.group(1))
    
    return values


def find_bad_page(images):
    """Find the page containing BAD data (usually has 'Belin' or 'Ectasia' in it)."""
    if not TESSERACT_AVAILABLE:
        return None, None
    
    for i, img in enumerate(images):
        # Quick OCR to identify the page
        text = pytesseract.image_to_string(img)
        if 'Belin' in text or 'Ectasia' in text or 'BAD' in text:
            return i, img
    return None, None


def process_single_dicom(dcm_path, verbose=False):
    """Extract BAD values from a single DICOM file."""
    if verbose:
        print(f"Processing: {os.path.basename(dcm_path)}")
    
    # Extract PDF
    pdf_bytes = extract_pdf_from_dicom(dcm_path)
    if pdf_bytes is None:
        if verbose:
            print("  No encapsulated PDF found")
        return None
    
    if not TESSERACT_AVAILABLE:
        return None
    
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, dpi=150)
        
        if verbose:
            print(f"  PDF has {len(images)} pages")
        
        # Find and process the BAD page
        bad_page_idx, bad_page = find_bad_page(images)
        
        if bad_page is None:
            # Try page 4 directly (often the BAD page)
            if len(images) >= 5:
                bad_page = images[4]
            elif len(images) >= 4:
                bad_page = images[3]
            else:
                if verbose:
                    print("  Could not find BAD page")
                return None
        
        # Extract values
        values = extract_bad_values_from_image(bad_page)
        
        if verbose and values:
            print(f"  Extracted: {values}")
        
        return values
        
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def batch_extract_bad_values(dicom_pattern="data/image_scans/*.DCM", output_csv="data/bad_values.csv"):
    """
    Extract BAD values from all DICOM files and save to CSV.
    """
    dcm_files = glob.glob(dicom_pattern)
    
    if not dcm_files:
        print(f"No DICOM files found: {dicom_pattern}")
        return
    
    print(f"Found {len(dcm_files)} DICOM files")
    print(f"Tesseract available: {TESSERACT_AVAILABLE}")
    
    if not TESSERACT_AVAILABLE:
        print("Cannot proceed without Tesseract OCR")
        return
    
    results = []
    
    for i, dcm_path in enumerate(dcm_files):
        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(dcm_files)}...")
        
        # Extract filename info
        basename = os.path.basename(dcm_path)
        
        # Parse filename for patient/exam info
        parts = basename.replace('.DCM', '').replace('.dcm', '').split('_')
        
        # Extract values
        values = process_single_dicom(dcm_path, verbose=False)
        
        if values:
            values['filename'] = basename.lower().replace('.dcm', '.jpg')
            results.append(values)
    
    print(f"\nSuccessfully extracted BAD values from {len(results)} files")
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Saved to: {output_csv}")
        
        # Show summary
        print("\nExtracted features:")
        for col in df.columns:
            if col != 'filename':
                valid = df[col].notna().sum()
                print(f"  {col}: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
        
        return df
    
    return None


def test_single_file():
    """Test extraction on a single file."""
    test_dcm = "data/image_scans/Aasi_Radi_OD_08052014_085912_4 Maps Select.DCM"
    
    if not os.path.exists(test_dcm):
        print(f"Test file not found: {test_dcm}")
        return
    
    print("Testing BAD extraction on single file...")
    print("="*50)
    
    values = process_single_dicom(test_dcm, verbose=True)
    
    if values:
        print("\nExtracted values:")
        for k, v in values.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo values extracted")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_file()
    else:
        batch_extract_bad_values()





















