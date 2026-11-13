#!/usr/bin/env python3
"""
Image Scan Preprocessor
Extracts metadata from JPG image scan filenames and outputs to CSV.

File naming pattern: [LASTNAME]_[FIRSTNAME]_[OD/OS]_[DDMMYYYY]_[HHMMSS]_4 Maps Select.JPG
"""

import os
import re
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path

def canonize_id(id):
    """
    Canonize patient ID by removing dashes and standardizing format.
    
    Args:
        id: Patient ID to canonize (can be string, int, or float)
        
    Returns:
        str: Canonized patient ID
    """
    # Convert to string and handle NaN/None values
    if pd.isna(id) or id is None:
        return ''
    
    # Convert to string and remove decimal point if it exists
    id_str = str(id)
    if '.' in id_str:
        id_str = id_str.split('.')[0]  # Remove decimal part
    
    # Remove dashes
    id_str = id_str.replace('-','')
    
    # Pad with zeros if less than 8 characters
    while (len(id_str) < 8):
        id_str = '0' + id_str
    
    # If 9 characters, remove last digit
    if len(id_str) == 9:
        id_str = id_str[:-1]
    
    return id_str

def load_patient_id_lookup(kc_csv_path):
    """
    Load patient ID lookup dictionary from KC CSV file.
    
    Args:
        kc_csv_path (str): Path to KC_filtered_by_date_labeled.csv file
        
    Returns:
        dict: Dictionary mapping patient names to IDs
    """
    try:
        # Read the KC CSV file
        df = pd.read_csv(kc_csv_path)
        
        # Create lookup dictionary
        # Key: "LASTNAME, FIRSTNAME" format to match image metadata
        # Value: patient ID
        lookup = {}
        
        for _, row in df.iterrows():
            # Skip rows with missing name data
            if pd.isna(row['Last Name:']) or pd.isna(row['First Name:']):
                continue
                
            last_name = str(row['Last Name:']).strip().upper()
            first_name = str(row['First Name:']).strip().upper()
            
            # Create the full name in the same format as image metadata
            full_name = f"{last_name} {first_name}"
            
            # Canonize the patient ID (handles NaN/missing values internally)
            canonized_id = canonize_id(row['id'])
            
            # Only store if we have a valid canonized ID
            if canonized_id:
                lookup[full_name] = canonized_id
            
        print(f"Loaded {len(lookup)} unique patient name-ID mappings from KC CSV")
        return lookup
        
    except Exception as e:
        print(f"Error loading patient ID lookup: {e}")
        return {}

def parse_filename(filename):
    """
    Parse the filename to extract patient name, eye (OD/OS), date, and time.
    
    Args:
        filename (str): The filename to parse
        
    Returns:
        dict: Dictionary containing extracted metadata or None if parsing fails
    """
    # Remove the file extension
    base_name = filename.replace('.JPG', '').replace('.jpg', '')
    
    # Pattern: LASTNAME_FIRSTNAME_[OD/OS]_DDMMYYYY_HHMMSS_4 Maps Select
    # We need to handle cases where names might have spaces or special characters
    
    # Split by underscore, but we need to be careful about names with underscores
    parts = base_name.split('_')
    
    if len(parts) < 5:
        print(f"Warning: Could not parse filename: {filename}")
        return None
    
    try:
        # Find the eye indicator (OD or OS) - it should be one of the parts
        eye_index = None
        for i, part in enumerate(parts):
            if part in ['OD', 'OS']:
                eye_index = i
                break
        
        if eye_index is None:
            print(f"Warning: Could not find eye indicator (OD/OS) in: {filename}")
            return None
        
        # Extract components based on eye_index position
        # Everything before eye_index is part of the name
        name_parts = parts[:eye_index]
        eye = parts[eye_index]
        
        # After eye should be date, time, and "4 Maps Select"
        if len(parts) <= eye_index + 2:
            print(f"Warning: Missing date/time information in: {filename}")
            return None
            
        date_str = parts[eye_index + 1]
        time_str = parts[eye_index + 2]
        
        # Validate date format (DDMMYYYY)
        if not re.match(r'^\d{8}$', date_str):
            print(f"Warning: Invalid date format in: {filename}")
            return None
            
        # Validate time format (HHMMSS)
        if not re.match(r'^\d{6}$', time_str):
            print(f"Warning: Invalid time format in: {filename}")
            return None
        
        # Parse date and time
        day = date_str[:2]
        month = date_str[2:4]
        year = date_str[4:8]
        
        hour = time_str[:2]
        minute = time_str[2:4]
        second = time_str[4:6]
        
        # Create datetime object for validation
        try:
            scan_datetime = datetime(int(year), int(month), int(day), 
                                   int(hour), int(minute), int(second))
        except ValueError as e:
            print(f"Warning: Invalid date/time values in {filename}: {e}")
            return None
        
        # Construct full name
        if len(name_parts) >= 2:
            last_name = name_parts[0]
            first_name = ' '.join(name_parts[1:])  # Join remaining parts as first name
        elif len(name_parts) == 1:
            last_name = name_parts[0]
            first_name = ""
        else:
            last_name = ""
            first_name = ""
        
        return {
            'filename': filename.lower(),
            'last_name': last_name,
            'first_name': first_name,
            'full_name': f"{last_name} {first_name}".strip(', '),
            'eye': eye,
            'scan_date': scan_datetime.strftime('%Y-%m-%d'),
            'scan_time': scan_datetime.strftime('%H:%M:%S'),
            'scan_datetime': scan_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'day': day,
            'month': month,
            'year': year,
            'hour': hour,
            'minute': minute,
            'second': second
        }
        
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def process_image_scans(input_dir, output_csv, kc_csv_path=None):
    """
    Process all JPG files in the input directory and extract metadata.
    
    Args:
        input_dir (str): Path to directory containing JPG files
        output_csv (str): Path to output CSV file
        kc_csv_path (str, optional): Path to KC CSV file for patient ID lookup
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Load patient ID lookup if KC CSV path is provided
    patient_id_lookup = {}
    if kc_csv_path:
        patient_id_lookup = load_patient_id_lookup(kc_csv_path)
    
    # Find all JPG files
    jpg_files = list(input_path.glob('*.JPG')) + list(input_path.glob('*.jpg'))
    
    print(f"Found {len(jpg_files)} JPG files to process...")
    
    # Process files and collect metadata
    metadata_list = []
    processed_count = 0
    failed_count = 0
    
    for jpg_file in jpg_files:
        metadata = parse_filename(jpg_file.name)
        if metadata:
            # Add patient ID lookup
            full_name = metadata['full_name']
            patient_id = patient_id_lookup.get(full_name, '')
            metadata['patient_id'] = patient_id
            
            # Add ideye concatenation column
            if patient_id:
                eye_name = 'right' if metadata['eye'] == 'OD' else 'left'
                metadata['ideye'] = f"{patient_id}-{eye_name}"
            else:
                metadata['ideye'] = ''
            
            metadata_list.append(metadata)
            processed_count += 1
        else:
            failed_count += 1
        
        # Progress indicator
        if (processed_count + failed_count) % 100 == 0:
            print(f"Processed {processed_count + failed_count} files...")
    
    print(f"Processing complete: {processed_count} successful, {failed_count} failed")
    
    if not metadata_list:
        print("No valid metadata extracted. Exiting.")
        return
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(metadata_list)
    
    # Ensure patient_id and ideye columns are stored as string type
    if 'patient_id' in df.columns:
        df['patient_id'] = df['patient_id'].astype(str)
        # Replace 'nan' strings with empty strings
        df['patient_id'] = df['patient_id'].replace('nan', '')
    
    if 'ideye' in df.columns:
        df['ideye'] = df['ideye'].astype(str)
        # Replace 'nan' strings with empty strings
        df['ideye'] = df['ideye'].replace('nan', '')
        # Also handle cases where patient_id was nan, resulting in "nan-right" or "nan-left"
        df['ideye'] = df['ideye'].replace(['nan-right', 'nan-left'], '')
    
    # Sort by patient name and scan datetime
    df = df.sort_values(['last_name', 'first_name', 'scan_datetime'])
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to: {output_csv}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total files processed: {len(df)}")
    print(f"Unique patients: {df['full_name'].nunique()}")
    print(f"OD scans: {len(df[df['eye'] == 'OD'])}")
    print(f"OS scans: {len(df[df['eye'] == 'OS'])}")
    print(f"Date range: {df['scan_date'].min()} to {df['scan_date'].max()}")
    
    # Show patient ID match statistics if KC CSV was provided
    if kc_csv_path and 'patient_id' in df.columns:
        matched_count = len(df[df['patient_id'] != ''])
        unmatched_count = len(df[df['patient_id'] == ''])
        print(f"Patients with matched IDs: {matched_count}")
        print(f"Patients without matched IDs: {unmatched_count}")
        print(f"ID match rate: {matched_count/len(df)*100:.1f}%")
    
    # Show first few rows
    print(f"\nFirst 5 records:")
    columns_to_show = ['full_name', 'eye', 'scan_date', 'scan_time']
    if 'patient_id' in df.columns:
        columns_to_show.append('patient_id')
    if 'ideye' in df.columns:
        columns_to_show.append('ideye')
    print(df[columns_to_show].head())

def main():
    """Main function to run the image scan preprocessor."""
    # Define paths
    input_dir = "/home/hcohen/Cloud/Work/projects/keratoconus-v2/data/image_scans"
    output_csv = "/home/hcohen/Cloud/Work/projects/keratoconus-v2/data/image_scan_metadata.csv"
    kc_csv_path = "/home/hcohen/Cloud/Work/projects/keratoconus-v2/data/221204_v2/KC_filtered_by_date_labeled.csv"
    
    # Ensure output directory exists
    output_path = Path(output_csv).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process the files
    process_image_scans(input_dir, output_csv, kc_csv_path)

if __name__ == "__main__":
    main()
