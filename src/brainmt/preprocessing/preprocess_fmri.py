"""
fMRI Data Preprocessing Script

This script preprocesses fMRI data by loading NIfTI files, applying normalization,
and saving them as PyTorch tensors in fp16 format for efficient storage and loading.

The preprocessing pipeline is adapted from:
- SwiFT: https://arxiv.org/abs/2307.05916
- TFF: https://arxiv.org/abs/2112.05761
"""

from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue


def preprocess_subject(filename, load_root, save_root, subj_name, scaling_method=None, fill_zeroback=False):
    """
    Preprocess a single subject's fMRI data.
    
    Args:
        filename (str): Name of the subject's file
        load_root (str): Root directory containing raw fMRI data
        save_root (str): Root directory to save processed data
        subj_name (str): Subject identifier
        scaling_method (str): Normalization method ('z-norm' or 'minmax')
        fill_zeroback (bool): Whether to fill background with zeros
    """
    print(f"Processing: {filename}", flush=True)
    
    # Construct path to fMRI file (modify this path based on your data structure)
    fmri_path = os.path.join(load_root, filename, "fMRI/filtered_func_data_clean_MNI.nii.gz")
    
    try:
        # Load NIfTI file
        data = LoadImage()(fmri_path)
    except Exception as e:
        print(f"Error loading {filename}: {e}", flush=True)
        return None

    # Validate data dimensions (modify expected timepoints as needed)
    expected_timepoints = 490
    if data.shape[-1] != expected_timepoints:
        print(f"Skipping {subj_name} due to incorrect shape: {data.shape} (expected timepoints: {expected_timepoints})", flush=True)
        return
    
    # Create output directory
    save_dir = os.path.join(save_root, subj_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Data dimensions: [width, height, depth, time]
    # Note: Crop dimensions to be under 96 voxels if needed
    # Each dimension in MNI space (2mm) should be around 100
    data = data[:, :, :, :]
    
    # Identify background voxels (zero values)
    background = data == 0
    
    # Apply normalization
    if scaling_method == 'z-norm':
        # Z-score normalization using non-background voxels
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_normalized = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        # Min-max normalization using non-background voxels
        data_min = data[~background].min()
        data_max = data[~background].max()
        data_normalized = (data - data_min) / (data_max - data_min)
    else:
        # No normalization
        data_normalized = data.clone()

    # Handle background values
    data_final = torch.empty(data.shape)
    if fill_zeroback:
        data_final[background] = 0
    else:
        # Set background to minimum normalized value
        data_final[background] = data_normalized[~background].min()
    data_final[~background] = data_normalized[~background]

    # Convert to fp16 for storage efficiency
    data_final = data_final.type(torch.float16)
    
    # Save processed data
    output_path = os.path.join(save_dir, "func_data_MNI_fp16.pt")
    torch.save(data_final, output_path)
    
    print(f"Processing of {subj_name} completed.", flush=True)


def main():
    """
    Main preprocessing function.
    
    Configure the paths and parameters below according to your dataset:
    """
    
    # =============================================================================
    # CONFIGURATION - Modify these paths and parameters for your dataset
    # =============================================================================
    
    # Dataset configuration
    dataset_name = 'UKBiobank'  # Change to your dataset name
    
    # Input directory containing raw fMRI NIfTI files
    # Structure should be: load_root/subject_folder/fMRI/filtered_func_data_clean_MNI.nii.gz
    load_root = '/path/to/your/raw/fmri/data'
    
    # Output directory for processed PyTorch tensors
    save_root = f'/path/to/processed/{dataset_name}_MNI_to_torch'
    
    # Normalization method: 'z-norm' (recommended) or 'minmax'
    scaling_method = 'z-norm'
    
    # Directory to check for already processed subjects (optional)
    finished_root = None  # Set to existing processed data directory if resuming
    
    # Expected number of timepoints (modify based on your data)
    expected_timepoints = 490
    
    # Number of parallel processes (adjust based on your CPU cores)
    max_parallel_processes = 2
    
    # =============================================================================
    # END CONFIGURATION
    # =============================================================================
    
    # Validate input directory
    if not os.path.exists(load_root):
        raise FileNotFoundError(f"Input directory not found: {load_root}")
    
    # Create output directories
    os.makedirs(os.path.join(save_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok=True)
    save_root_img = os.path.join(save_root, 'img')
    
    # Get list of subjects to process
    filenames = os.listdir(load_root)
    total_subjects = len(filenames)
    print(f"Found {total_subjects} subjects to process")
    
    # Check for already processed subjects
    finished_samples = []
    if finished_root and os.path.exists(finished_root):
        finished_samples = os.listdir(finished_root)
        print(f"Found {len(finished_samples)} already processed subjects")
    
    # Process subjects
    processed_count = 0
    active_processes = []
    
    for filename in sorted(filenames):
        # Extract subject name (modify based on your naming convention)
        subj_name = filename.replace('.nii.gz', '')  # Remove file extension
        
        # Skip if already processed
        if subj_name in finished_samples:
            print(f"Skipping {subj_name} (already processed)", flush=True)
            continue
        
        try:
            processed_count += 1
            print(f"Starting processing for subject {processed_count}/{total_subjects}: {subj_name}")
            
            # Create process for this subject
            process = Process(
                target=preprocess_subject,
                args=(filename, load_root, save_root_img, subj_name, scaling_method, False)
            )
            process.start()
            active_processes.append(process)
            
            # Limit number of parallel processes
            if len(active_processes) >= max_parallel_processes:
                # Wait for processes to complete
                for p in active_processes:
                    p.join()
                active_processes = []
                
        except Exception as e:
            print(f'Error processing {filename}: {e}', flush=True)
    
    # Wait for remaining processes
    for p in active_processes:
        p.join()
    
    print(f"\nProcessing completed! Processed {processed_count} subjects.")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_minutes = round((end_time - start_time) / 60)
    print(f'\nTotal processing time: {elapsed_minutes} minutes')
