import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

log = logging.getLogger(__name__)

try:
    from utils.distributed import get_rank
except:
    def get_rank():
        return 0

class fMRIDataset(Dataset):
    def __init__(self, img_path, target_path, target_col, id_col, num_frames_slice=200):
        self.img_path = img_path
        self.target_col = target_col
        self.id_col = id_col
        self.num_frames_slice = num_frames_slice
        if get_rank() == 0:
            log.info(f"Loading target data from: {target_path}")
        if target_path.endswith('.pkl'):
            self.target_df = pd.read_pickle(target_path)
        elif target_path.endswith('.csv'):
            self.target_df = pd.read_csv(target_path)
        else:
            raise ValueError("Target file must be a .pkl or .csv file.")

        if get_rank() == 0:
            log.info("Filtering subjects based on available image data.")
        all_subject_dirs = {d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))}
        
        # Ensure subject IDs in dataframe are strings to match directory names
        self.target_df[self.id_col] = self.target_df[self.id_col].astype(str)
        
        # Filter dataframe for subjects that have a directory
        self.target_df = self.target_df[self.target_df[self.id_col].isin(all_subject_dirs)]
        
        # Create a dictionary for quick lookup
        self.target_dict = self.target_df.set_index(self.id_col)[self.target_col].to_dict()
        
        # Final filter: ensure the required .pt file exists
        self.subject_dirs = [
            d for d in self.target_df[self.id_col].tolist()
            if os.path.isfile(os.path.join(img_path, d, 'func_data_MNI_fp16.pt'))
        ]
        
        if get_rank() == 0:
            log.info(f"Found {len(self.subject_dirs)} subjects with complete data.")

    def __len__(self):
        return len(self.subject_dirs)
    
    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]
        img_file = os.path.join(self.img_path, subject_dir, 'func_data_MNI_fp16.pt')
        
        try:
            data = torch.load(img_file)
        except Exception as e:
            log.error(f"Error loading {img_file}: {e}")
            raise RuntimeError(f"Failed to load {img_file}")
            
        # Randomly select a contiguous segment of frames along the time dimension
        total_frames = data.shape[-1]
        num_frames = self.num_frames_slice
        if total_frames > num_frames:
            start_index = torch.randint(0, total_frames - num_frames + 1, (1,)).item()
            end_index = start_index + num_frames
            data_sliced = data[:, :, :, start_index:end_index]
        else:
            data_sliced = data[:, :, :, :num_frames]
        
        # Reshape and permute
        data_global = data_sliced.unsqueeze(0).permute(4, 0, 2, 1, 3)
        
        target = self.target_dict[subject_dir]
        
        return data_global, torch.tensor(target, dtype=torch.float16)
