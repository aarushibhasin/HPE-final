import gc
import json
from tqdm import tqdm
import concurrent.futures
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from abc import ABC, abstractmethod
import random
import pickle
import sys
import os
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
import matplotlib.pyplot as plt

import multiprocessing as mp
#mp.set_start_method("fork", force=True)


DATA_ROOT_STRING = "C:\\Users\\GSBME\\SmartCupStudy\\Unified_network\\data_sets\\UNSW-PANOPTES"
#DATA_ROOT_STRING = Path("/media/rokny") / "DATA1" / "Jonathan" / "UNSW-PANOPTES"

#BODY_8_KPT_NAMES = 

BODY_8_KPT_COLS_X = ['Nose_x', 'L_Eye_x', 'R_Eye_x', 'L_Ear_x', 'R_Ear_x', 'L_Shoulder_x', 'R_Shoulder_x', 'L_Elbow_x', 'R_Elbow_x', 'L_Wrist_x', 'R_Wrist_x', 'L_Hip_x', 'R_Hip_x', 'L_Knee_x', 'R_Knee_x', 'L_Ankle_x', 'R_Ankle_x', 'Head_Apex_x', 'Neck_x', 'Hip_Center_x', 'L_BigToe_x', 'R_BigToe_x', 'L_SmallToe_x', 'R_SmallToe_x', 'L_Heel_x', 'R_Heel_x']
BODY_8_KPT_COLS_Y = ['Nose_y', 'L_Eye_y', 'R_Eye_y', 'L_Ear_y', 'R_Ear_y', 'L_Shoulder_y', 'R_Shoulder_y', 'L_Elbow_y', 'R_Elbow_y', 'L_Wrist_y', 'R_Wrist_y', 'L_Hip_y', 'R_Hip_y', 'L_Knee_y', 'R_Knee_y', 'L_Ankle_y', 'R_Ankle_y', 'Head_Apex_y', 'Neck_y', 'Hip_Center_y', 'L_BigToe_y', 'R_BigToe_y', 'L_SmallToe_y', 'R_SmallToe_y', 'L_Heel_y', 'R_Heel_y']


def pc_array_to_grid(pc, max_pcs, pc_dims, pc_grid_dim):
    unsorted_pc_grid = np.zeros((max_pcs, pc_dims), dtype=pc.dtype)
    num_points = pc.shape[0]
    if num_points > max_pcs:
        #randomly remove points
        #TODO: maybe remove points with lowest SNR instead
        indices = np.arange(num_points)
        np.random.shuffle(indices)
        indices = indices[:max_pcs]
        unsorted_pc_grid = pc[indices]
    if num_points < max_pcs:
        #0 pad
        padded_arr = np.zeros((max_pcs, pc_dims), dtype=pc.dtype)
        padded_arr[:num_points] = pc
        unsorted_pc_grid = padded_arr
    
    unsorted_pc_grid = unsorted_pc_grid.reshape((max_pcs, pc_dims))
    pc_grid = np.array(unsorted_pc_grid[np.lexsort((unsorted_pc_grid[:, 2], unsorted_pc_grid[:, 1], unsorted_pc_grid[:, 0]))])
    pc_grid = pc_grid.reshape(pc_grid_dim, pc_grid_dim, pc_dims)
    return pc_grid

def transform_pc_frame(pc_np: np.ndarray, max_pcs, pc_dims, pc_grid_dim) -> torch.Tensor:
    """
    Transform a NumPy array of point cloud data into a tensor.
    """
    # Reshape to a sorted grid for mpMARS
    pc_grid = pc_array_to_grid(pc_np, max_pcs, pc_dims, pc_grid_dim)
    return torch.Tensor(pc_grid)

def pose_array_to_posepresence(poses, max_poses, num_kpts, kpt_dims):
    '''
    Process pose estimate labels into padded poses and presence arrays.

    Args:
        poses (np.ndarray): Array of shape (n, self.num_kpts, self.kpt_dims)
    Returns:
        pposes (np.ndarray): Array of (self.max_poses, self.num_kpts, self.kpt_dims) with the poses padded to size self.max_poses.
    '''
    # Number of poses in the current frame
    n = poses.shape[0]
    pposes = np.zeros((max_poses, num_kpts, kpt_dims), dtype=poses.dtype)
    pposes[:n] = poses  # Copy the poses into the padded array
    # Create a one-hot encoded presence array
    ppresence = np.zeros((max_poses+1,), dtype=np.int32)
    ppresence[n] = 1 #one hot encoding
    return pposes, ppresence

def transform_pose_frame_np(poses_np: np.ndarray, max_poses, num_kpts, kpt_dims) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform a NumPy array of pose data into tensors.
    """
    if np.isnan(poses_np).all():
        poses = np.zeros((max_poses, num_kpts, kpt_dims))
        presence = np.zeros((max_poses + 1,), dtype=np.int32)
        presence[0] = 1
        return torch.Tensor(poses), torch.Tensor(presence)

    # Extract x and y coordinates assuming specific columns
    poses_x_np = poses_np[:, ::2]
    poses_y_np = poses_np[:, 1::2]

    # Stack x and y coordinates along the last dimension
    poses_stacked = np.stack((poses_x_np, poses_y_np), axis=-1)

    # Process poses and presence
    poses, presence = pose_array_to_posepresence(poses_stacked, max_poses, num_kpts, kpt_dims)
    return torch.Tensor(poses), torch.Tensor(presence)

TEMP_PATH = Path("temp_little_pickles")

def save_temp_little_pickles(uuid, chunk):
    save_path = TEMP_PATH / f"chunk_{uuid}.pickle"
    with open(save_path, 'wb') as handle:
        pickle.dump(chunk, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    

def preprocess_record_batch(args):
    
    batch, tempf_uuid, shm_name_pose, all_pose_shape, all_pose_dtype, shm_name_pc, all_pc_shape, all_pc_dtype, max_pcs, pc_dims, pc_grid_dim, max_poses, num_kpts, kpt_dims = args
    results = {}
    record_id_col = -1  # Assume 'record_id' is column 0 in all_pose
    frame_number_col = 0  # Assume 'frame_number_rgb' is column 1 in all_pose
    path_col = -2  # Assume 'path' is the last column in all_pose
    pc_path_col = -1
    xyzds_cols = [2,3,4,5,6]
    kpts_cols_slice = [1,-2]
    #print(f"Supposed to be record_id: {all_pose[:,0]}")
    #print(f"Supposed to be record_id: {all_pose[:,-1]}")
    
    shm_pose = SharedMemory(name=shm_name_pose)
    arr_view_pose = np.ndarray(all_pose_shape, dtype=all_pose_dtype, buffer=shm_pose.buf)
    
    shm_pc = SharedMemory(name=shm_name_pc)
    arr_view_pc = np.ndarray(all_pc_shape, dtype=all_pc_dtype, buffer=shm_pc.buf)
    
    for record_id in batch:
        pose_mask = arr_view_pose[:, record_id_col] == record_id
        #print(f"arr_view_pose: {arr_view_pose.shape}")
        #print(f"pose_mask: {pose_mask.shape}")
        pose_records = arr_view_pose[pose_mask].copy()
        #print(f"pose_records: {pose_records.shape}")
        if pose_records.size == 0:
            #print("skipped because no pose records")
            continue  # Skip if no pose records found
        
        frame_number_rgb = pose_records[0, frame_number_col]
        session_path = pose_records[0, path_col]
        
        pc_mask = (arr_view_pc[:, frame_number_col] == frame_number_rgb) & (arr_view_pc[:, pc_path_col] == session_path)
        #print(f"all_pc: {arr_view_pc.shape}")
        #print(f"pc_mask: {pc_mask.shape}")
        pc_records = arr_view_pc[pc_mask].copy()
        #print(f"pc_records: {pc_records.shape}")
        if pc_records.size == 0:
            #print("skipped because no pc records")
            continue  # Skip if no point cloud records found
        
        # Transform using NumPy-compatible functions
        pc_records_xyzds = pc_records[:,xyzds_cols]
        pc_records_xyzds = pc_records_xyzds.astype('float64')
        pose_records_kpts = pose_records[:,kpts_cols_slice[0]:kpts_cols_slice[1]]
        pose_records_kpts = pose_records_kpts.astype('float64')

        pcs = transform_pc_frame(pc_records_xyzds, max_pcs, pc_dims, pc_grid_dim)  # Pass only relevant columns
        poses, presence = transform_pose_frame_np(pose_records_kpts, max_poses, num_kpts, kpt_dims)  # Pass only relevant columns

        results[record_id] = {
            "kpts": poses,
            "presence": presence,
            "pc": pcs,
            "path": session_path,
            "frame_number_rgb": frame_number_rgb
        }
    save_temp_little_pickles(tempf_uuid, results)    

def preprocess_record_batch_not_shared(args):
    batch, tempf_uuid, all_pose, all_pc, max_pcs, pc_dims, pc_grid_dim, max_poses, num_kpts, kpt_dims = args
    results = {}
    record_id_col = -1  # Assume 'record_id' is column 0 in all_pose
    frame_number_col = 0  # Assume 'frame_number_rgb' is column 1 in all_pose
    path_col = -2  # Assume 'path' is the last column in all_pose
    pc_path_col = -1
    xyzds_cols = [2,3,4,5,6]
    kpts_cols_slice = [1,-2]
    #print(f"Supposed to be record_id: {all_pose[:,0]}")
    #print(f"Supposed to be record_id: {all_pose[:,-1]}")
    for record_id in batch:
        pose_mask = all_pose[:, record_id_col] == record_id
        #print(f"all_pose: {all_pose.shape}")
        #print(f"pose_mask: {pose_mask.shape}")
        pose_records = all_pose[pose_mask].copy()
        #print(f"pose_records: {pose_records.shape}")
        if pose_records.size == 0:
            #print("skipped because no pose records")
            continue  # Skip if no pose records found
        
        frame_number_rgb = pose_records[0, frame_number_col]
        session_path = pose_records[0, path_col]
        
        pc_mask = (all_pc[:, frame_number_col] == frame_number_rgb) & (all_pc[:, pc_path_col] == session_path)
        #print(f"all_pc: {all_pc.shape}")
        #print(f"pc_mask: {pc_mask.shape}")
        pc_records = all_pc[pc_mask].copy()
        #print(f"pc_records: {pc_records.shape}")
        if pc_records.size == 0:
            #print("skipped because no pc records")
            continue  # Skip if no point cloud records found
        
        # Transform using NumPy-compatible functions
        pc_records_xyzds = pc_records[:,xyzds_cols]
        pc_records_xyzds = pc_records_xyzds.astype('float64')
        pose_records_kpts = pose_records[:,kpts_cols_slice[0]:kpts_cols_slice[1]]
        pose_records_kpts = pose_records_kpts.astype('float64')

        pcs = transform_pc_frame(pc_records_xyzds, max_pcs, pc_dims, pc_grid_dim)  # Pass only relevant columns
        poses, presence = transform_pose_frame_np(pose_records_kpts, max_poses, num_kpts, kpt_dims)  # Pass only relevant columns

        results[record_id] = {
            "kpts": poses,
            "presence": presence,
            "pc": pcs,
            "path": session_path,
            "frame_number_rgb": frame_number_rgb
        }
        #results[record_id] = {}
    #print("done")
    save_temp_little_pickles(tempf_uuid, results)    
    #return results #{}


class poseidonDataLoader(Dataset):
    def __init__(self, subjects: list, 
                 dynamic_loading: bool = True,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 pc_grid_dim: int = 8,
                 data_build: str = "POSEIDON_build_w0.083"):
        self.subjects = subjects
        self.dynamic_loading = dynamic_loading
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_build = data_build
        self.pc_grid_dim = pc_grid_dim
        self.max_pcs = self.pc_grid_dim * self.pc_grid_dim
        self.pc_dims = 5
        self.num_kpts = 26
        self.kpt_dims = 2
        self.max_poses = 4
        self.processed_data = {}
        # Initialize file paths
        self.init_paths()
        print("Paths initialized")

        self.sanity_check()
        print("Sanity Checked")

        #self.compile_all_pose_and_pc()
        #print("All pose and pc compiled")

        #self.define_record_list()
        #print("Defined record list")
    
    def init_paths(self):
        self.data_root = Path(DATA_ROOT_STRING)
        self.data_build_path = self.data_root / self.data_build
        self.synchronized_data_path = self.data_build_path / "sync_data"
    
    @abstractmethod
    def sanity_check(self):
        pass

    def compile_all_pose_and_pc(self):
        #TODO: build a frame_to_origin_path.csv instead
        all_pose_dfs = []
        all_pc_dfs = []
        for cohort_path in self.synchronized_data_path.iterdir():
            if cohort_path.is_file(): continue
            for subject_path in cohort_path.iterdir():
                if subject_path.is_file() or subject_path.name not in self.subjects: continue
                for session_path in subject_path.iterdir():
                    if session_path.is_file(): continue
                    pose_file_path = session_path / "pose.csv"
                    pc_file_path = session_path / "pc.csv"
                    pose_df = pd.read_csv(pose_file_path, na_filter=True)
                    pc_df = pd.read_csv(pc_file_path, na_filter=True)
                    relative_path = Path(*session_path.parts[-3:])
                    print(relative_path)
                    pose_df['path'] = relative_path
                    pc_df['path'] = relative_path
                    all_pose_dfs.append(pose_df)
                    all_pc_dfs.append(pc_df)
        self.all_pose = pd.concat(all_pose_dfs)
        self.all_pc = pd.concat(all_pc_dfs)
        #print(f"all_pose.dtypes: {self.all_pose.dtypes}")
        #print(f"all_pc.dtypes: {self.all_pc.dtypes}")

        return
    
    @abstractmethod
    def define_record_list(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def get_dataloader(self) -> DataLoader:
        return DataLoader(self,
                          batch_size = self.batch_size,
                          shuffle = self.shuffle,
                          collate_fn = None)

class poseidonDataLoaderMars(poseidonDataLoader):
    def __init__(self, subjects: list, 
                 dynamic_loading: bool = True,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 pc_grid_dim: int = 8,
                 data_build: str = "POSEIDON_build_w0.083"):
        super().__init__(subjects, dynamic_loading, batch_size, shuffle, pc_grid_dim, data_build)
        #self.preprocess_data_multithreaded()
        #print("Data preprocessing complete")
        #UNCOMMENT WHEN DONE TESTING...
        '''
        self.all_pc_np = self.all_pc.to_numpy()
        self.all_pose_np = self.all_pose.to_numpy()
        
        self.shm_all_pose = SharedMemory(create=True, size=self.all_pose_np.nbytes)
        arr_view_all_pose = np.ndarray(self.all_pose_np.shape, dtype=self.all_pose_np.dtype, buffer=self.shm_all_pose.buf)
        arr_view_all_pose[:] = self.all_pose_np[:]  # copy data into shared memory
        
        self.shm_all_pc = SharedMemory(create=True, size=self.all_pc_np.nbytes)
        arr_view_all_pc = np.ndarray(self.all_pc_np.shape, dtype=self.all_pc_np.dtype, buffer=self.shm_all_pc.buf)
        arr_view_all_pc[:] = self.all_pc_np[:]  # copy data into shared memory
        '''

    def compile_all_pose_and_pc(self):
        all_pose_dfs = []
        all_pc_dfs = []
        for cohort_path in self.synchronized_data_path.iterdir():
            if cohort_path.is_file(): continue
            for subject_path in cohort_path.iterdir():
                if subject_path.is_file() or subject_path.name not in self.subjects: continue
                for session_path in subject_path.iterdir():
                    if session_path.is_file(): continue
                    pose_file_path = session_path / "pose.csv"
                    pc_file_path = session_path / "pc.csv"
                    pose_df = pd.read_csv(pose_file_path, na_filter=True)
                    pc_df = pd.read_csv(pc_file_path, na_filter=True)
                    relative_path = Path(*session_path.parts[-3:])
                    pose_df['path'] = str(relative_path)
                    pc_df['path'] = str(relative_path)
                    all_pose_dfs.append(pose_df)
                    all_pc_dfs.append(pc_df)

        self.all_pose = pd.concat(all_pose_dfs)  # Convert to NumPy
        self.all_pc = pd.concat(all_pc_dfs)      # Convert to NumPy
        return

    def define_record_list(self):
        #TODO: make this dependent on the RGB frame numbers specified in the ground truth and then move it to abstract
        #undersample = random.sample(range(len(self.all_pose)), 1000)  # Generate 100 random integer indices
        #self.all_pose = self.all_pose.iloc[undersample]
        self.all_pose['record_id'] = pd.factorize(self.all_pose[['frame_number_rgb', 'path']].apply(tuple,axis=1))[0]
        return
    
    def combine_pickle_chunks(self):
        combined_dict = {}
        
        for file_path in TEMP_PATH.iterdir():
            with open(file_path, "rb") as f:
                chunk_dict = pickle.load(f)
            combined_dict.update(chunk_dict)
        return combined_dict
            
    
    def preprocess_data_batched(self, chunk_size=200):
        """
        Preprocess data in chunks using multiprocessing and NumPy for efficiency.
        """
        # Extract unique record IDs
        unique_record_ids = self.all_pose['record_id'].unique()

        # Split record IDs into chunks
        record_batches = [
            unique_record_ids[i:i + chunk_size] for i in range(0, len(unique_record_ids), chunk_size)
        ]

        pbar = tqdm(total=len(record_batches), desc="Processing Records", file=sys.stdout, ncols=80)

        # Use multiprocessing Pool for efficient parallel execution
        #with Pool(processes=min(os.cpu_count(), 90), maxtasksperchild=5) as pool:
        with Pool(processes=min(os.cpu_count(), 110)) as pool:
            try:
                args_list = [
                    #(batch, tuuid, self.all_pose_np, self.all_pc_np, self.max_pcs, self.pc_dims, self.pc_grid_dim, self.max_poses, self.num_kpts, self.kpt_dims) for tuuid, batch in enumerate(record_batches)
                    (batch, tuuid, self.shm_all_pose.name, self.all_pose_np.shape, self.all_pose_np.dtype, self.shm_all_pc.name, self.all_pc_np.shape, self.all_pc_np.dtype, self.max_pcs, self.pc_dims, self.pc_grid_dim, self.max_poses, self.num_kpts, self.kpt_dims) for tuuid, batch in enumerate(record_batches)
                ]
                results = pool.imap_unordered(
                    preprocess_record_batch,
                    args_list
                )

                for chunk_result in results:
                    pbar.update(1)
                    gc.collect()
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Terminating workers...")
                pool.terminate()
                raise
            finally:
                #pool.close()
                #pool.join()
                pool.terminate()  # ✅ Ensure all processes exit
                pool.join()
                del pool  # ✅ Remove lingering reference
                gc.collect()  # ✅ Cleanup memory

        pbar.close()
        
        print("Combining temp files")
        self.processed_data = self.combine_pickle_chunks()
        
        return self.processed_data
    
    
    def preprocess_data_threadpool(self):
        """
        Preprocess all pose and point cloud records during initialization
        using multi-threading for faster execution.
        """
        
        unique_record_ids = self.all_pose['record_id'].unique()
        pbar_groups = tqdm(total=len(unique_record_ids), leave=True, desc="Processing Records", file=sys.stdout, ncols=80)

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                # Submit tasks for each record ID
                future_to_record = {
                    executor.submit(preprocess_record, record_id, self.all_pose, self.all_pc, self.transform_pc_frame, self.transform_pose_frame): record_id
                    for record_id in unique_record_ids
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_record):
                    record_id, processed_item = future.result()
                    self.processed_data[record_id] = processed_item
                    pbar_groups.update(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Shutting down...")
                executor.shutdown(wait=False)  # Stop threads immediately
                raise
        pbar_groups.close()
        return self.processed_data
    
    def preprocess_data_processpool(self):
        """
        Preprocess all pose and point cloud records during initialization
        using multi-threading for faster execution.
        """
        
        unique_record_ids = self.all_pose['record_id'].unique()
        pbar_groups = tqdm(total=len(unique_record_ids), leave=True, desc="Processing Records", file=sys.stdout, ncols=80)

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            try:
                # Submit tasks for each record ID
                future_to_record = {
                    executor.submit(preprocess_record, record_id, self.all_pose, self.all_pc, self.transform_pc_frame, self.transform_pose_frame): record_id
                    for record_id in unique_record_ids
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_record):
                    record_id, processed_item = future.result()
                    self.processed_data[record_id] = processed_item
                    pbar_groups.update(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Shutting down...")
                executor.shutdown(wait=False)  # Stop threads immediately
                raise
        pbar_groups.close()
        return self.processed_data

    def preprocess_data_batched_old(self):
        """
        Preprocess data in batches using multiprocessing for efficiency.
        """
        unique_record_ids = self.all_pose['record_id'].unique()
        batch_size = 50  # Adjust this based on your system and workload
        record_batches = [
            unique_record_ids[i:i + batch_size] for i in range(0, len(unique_record_ids), batch_size)
        ]
    
        # Use tqdm to track progress
        pbar = tqdm(total=len(record_batches), desc="Processing Records", file=sys.stdout, ncols=80)
    
        # Use multiprocessing Pool for parallel execution
        #with Pool(processes=min(os.cpu_count() // 2, 16)) as pool:
        with Pool(processes=50) as pool:
            try:
                # Pass arguments as tuples
                args_list = [
                    (batch, self.all_pose, self.all_pc, self.transform_pc_frame, self.transform_pose_frame)
                    for batch in record_batches
                ]
                # Use pool.imap_unordered for better load balancing
                results = pool.imap_unordered(
                    preprocess_record_batch,
                    args_list
                )
    
                # Collect and merge results
                for batch_results in results:
                    self.processed_data.update(batch_results)
                    pbar.update(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Terminating workers...")
                pool.terminate()
                raise
            finally:
                pool.close()
                pool.join()
    
        pbar.close()
        return self.processed_data

    def save_processed_data(self, save_path=None):
        if not save_path:
            save_path = Path("processed_data") / f"{len(self.subjects)} subjects__{self.subjects}"
        save_path.mkdir(exist_ok=True, parents=True)
        pick_file = save_path / "processed_data.pickle"
        subjects_file = save_path / "subjects.json"
        with open(subjects_file, "w") as json_file:
            json.dump(self.subjects, json_file, indent=4)
        with open(pick_file, 'wb') as handle:
            pickle.dump(self.processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return save_path

    def clean_data(self):
        print("Unclean processed_data, length:", len(self.processed_data))
        clean_samples = []
        for sample_id, sample in self.processed_data.items():
            if (torch.isnan(sample['kpts']).any() or torch.isinf(sample['kpts']).any() or
                torch.isnan(sample['presence']).any() or torch.isinf(sample['presence']).any() or
                torch.isnan(sample['pc']).any() or torch.isinf(sample['pc']).any()):
                continue  
            clean_samples.append(sample)
        # Reassign keys to be consecutive integers.
        self.processed_data = {i: sample for i, sample in enumerate(clean_samples)}
        print("Cleaned processed_data, new length:", len(self.processed_data))
    
    def plot_data_distributions(self, root_load_path):
        kpts_x, kpts_y = [], []
        presence_list = []
        pc_channels = [[] for _ in range(5)]

        for sample_id, sample in self.processed_data.items():
            kpts_x.append(sample['kpts'][:, :, 0].flatten())
            kpts_y.append(sample['kpts'][:, :, 1].flatten())
            presence_list.append(sample['presence'].argmax())
            for i in range(5):
                pc_channels[i].append(sample['pc'][:, :, i].flatten())
        kpts_x = torch.cat(kpts_x).cpu().numpy()
        kpts_y = torch.cat(kpts_y).cpu().numpy()
        mask = (kpts_x != 0) & (kpts_y != 0)
        kpts_x = kpts_x[mask]
        kpts_y = kpts_y[mask]
        pc_channels = [torch.cat(pc_channel).cpu().numpy() for pc_channel in pc_channels]
        presence_arr = np.array(presence_list)

        def plot_distribution(data, title, file_path, bins=100):
            plt.figure(figsize=(12, 6))
            plt.hist(data, bins=bins, alpha=0.75, color='b', edgecolor='black')
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(file_path)
            plt.close()
        
        plot_distribution(kpts_x, "Distribution of kpts[:,:,0]", root_load_path / "kpts_x_distribution.png")
        plot_distribution(kpts_y, "Distribution of kpts[:,:,1]", root_load_path / "kpts_y_distribution.png")
        plot_distribution(presence_arr, f"Distribution of presence.argmax()", root_load_path / f"presence_distribution.png", bins=len(sample['presence']))
        for i in range(5):
            plot_distribution(pc_channels[i], f"Distribution of pc[:,:,{i}]", root_load_path / f"pc_channel_{i}_distribution.png")

    def load_processed_data(self, load_path, clean=True, plot=True, pose_norm=True, save_cleaned=False, single_person_only=False):
        print("Loading pickle")
        root_load_path = load_path.parent
        with open(load_path, 'rb') as handle:
            self.processed_data = pickle.load(handle)
        print("Cleaning data")
        if clean:
            self.clean_data()
        #print(self.processed_data[0]['presence'].shape)
        #print(self.processed_data[0]['pc'].shape)
        if pose_norm:
            print("Normalizing kpts")
            kpts_list = [sample['kpts'] for sample in self.processed_data.values() if 'kpts' in sample]

            kpts_tensor = torch.stack(kpts_list)
            # Compute min and max for x-coordinates ([:,:,0]) and y-coordinates ([:,:,1])
            min_x, max_x = kpts_tensor[..., 0].min(), kpts_tensor[..., 0].max()
            min_y, max_y = kpts_tensor[..., 1].min(), kpts_tensor[..., 1].max()

            # Normalize the x and y coordinates separately
            kpts_tensor[..., 0] = (kpts_tensor[..., 0] - min_x) / (max_x - min_x)
            kpts_tensor[..., 1] = (kpts_tensor[..., 1] - min_y) / (max_y - min_y)

            # Assign normalized tensors back to the dictionary
            new_processed = {}
            new_keys = 0
            for i, key in enumerate(self.processed_data.keys()):
                if 'kpts' in self.processed_data[key]:
                    self.processed_data[key]['kpts'] = kpts_tensor[i]
                    if single_person_only and self.processed_data[key]['presence'][1]==1:
                      new_processed[new_keys] = self.processed_data[key]
                      new_keys+=1
            if single_person_only:
                self.processed_data = new_processed
                
        if plot:
            self.plot_data_distributions(root_load_path)
        
        if save_cleaned:
            root_processed_path = root_load_path.parent
            root_processed_clean_path = root_processed_path.parent / f"{str(root_processed_path.name)}_cleaned"
            processed_clean_split_path = root_processed_clean_path / str(root_load_path.name)
            processed_clean_split_path.mkdir(exist_ok=True, parents=True)
            load_subjects_file = root_load_path / "subjects.json"
            with open(load_subjects_file) as f:
                subjects = json.load(f)
            save_subjects_file = processed_clean_split_path / "subjects.json"
            with open(save_subjects_file, "w") as json_file:
                json.dump(subjects, json_file, indent=4)
            pickle_file = processed_clean_split_path / "processed_data.pickle"
            print("Saving cleaned pickle")
            with open(pickle_file, 'wb') as handle:
                pickle.dump(self.processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
            
    
    def transform_pose_frame(self, poses_df: pd.core.frame.DataFrame) -> torch.Tensor:
        if poses_df.isna().all().all():
            poses_np = poses_df.to_numpy()
            poses = np.zeros((self.max_poses, self.num_kpts, self.kpt_dims))
            presence = np.zeros((self.max_poses+1,), dtype=np.int32)
            presence[0] = 1
            return torch.Tensor(poses), torch.Tensor(presence)
        poses_x_df = poses_df[BODY_8_KPT_COLS_X]
        poses_y_df = poses_df[BODY_8_KPT_COLS_Y]
        poses_x_np = poses_x_df.to_numpy()
        poses_y_np = poses_y_df.to_numpy()
        #print(f"poses_x_np: {poses_x_np.shape}")
        #print(f"poses_y_np: {poses_y_np.shape}")
        poses_np = np.stack((poses_x_np,poses_y_np),axis=-1)
        #print(poses_np.shape)
        poses, presence = pose_array_to_posepresence(poses_np, self.max_poses, self.num_kpts, self.kpt_dims)
        return torch.Tensor(poses), torch.Tensor(presence)
    
    def __len__(self):
        if len(self.processed_data)!=0:
            return len(self.processed_data)
        else:
            return len(self.all_pose['record_id'].unique())
    
    def __getitem__(self, idx):
        if len(self.processed_data)!=0:
            return self.processed_data[idx]
        pose_record = self.all_pose[self.all_pose['record_id']==idx]
        frame_number_rgb = pose_record['frame_number_rgb'].iloc[0]
        session_path = pose_record['path'].iloc[0]
        pc_records = self.all_pc[(self.all_pc['frame_number_rgb']==frame_number_rgb) & (self.all_pc['path']==session_path)]
        
        pcf = pc_records.drop(['frame_number_rgb', 'frame_number_pc', 'path'],axis=1)
        posef = pose_record.drop(['frame_number_rgb','path', 'record_id'], axis=1)  
        pcs = transform_pc_frame(pcf.to_numpy(), self.max_pcs, self.pc_dims, self.pc_grid_dim)
        poses, presence = self.transform_pose_frame(posef)
        item = {"kpts": poses,
                "presence": presence,
                "pc": pcs,
                "path": session_path,
                "frame_number_rgb": frame_number_rgb}
        return item
    
class poseidonDataLoaderMri(Dataset):
    def __init__(self, root_path: Path, batch_size: int = 8, shuffle: bool = True):
        self.root_path = root_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pc_path = root_path / "featuremap.npy"
        self.pose_path = root_path / "kpt_labels.npy"
        self.num_kpts = 17
        self.kpt_dims = 2
        self.pc_grid_dim = 8
        self.max_poses = 4
        self.all_pc = None
        self.all_pose = None
        self.all_kpts = None
        self.all_presence = None
        self.load_data()
    
    def load_data(self):
        self.all_pc = np.load(self.pc_path, )
        self.all_pose = np.load(self.pose_path)
        #convert pose to kpt and presence arrays
        kpts = self.all_pose[:, :-self.max_poses]
        kpts = kpts.reshape(-1,self.kpt_dims,self.max_poses,self.num_kpts)
        self.all_kpts = np.transpose(kpts,(0,2,3,1))
        self.all_kpts[:,:,:,1] = -self.all_kpts[:,:,:,1]
        #make all_kpts be n, max_poses, num_kpts, kpt_dims
        self.all_presence = np.zeros((self.all_kpts.shape[0],self.max_poses+1), dtype=np.int32)
        for i in range(self.all_kpts.shape[0]):
            # Count how many poses have non-zero keypoints
            num_poses = 0
            for pose_idx in range(self.max_poses):
                pose = self.all_kpts[i, pose_idx]  # (num_kpts, kpt_dims)
                if np.any(pose != 0):  # If any keypoint is non-zero
                    num_poses += 1
            
            # Set the presence for the correct number of people
            self.all_presence[i, num_poses] = 1  # one hot encoding
        self.all_pc = torch.tensor(self.all_pc, dtype=torch.float32)
        self.all_kpts = torch.tensor(self.all_kpts, dtype=torch.float32)
        self.all_presence = torch.tensor(self.all_presence, dtype=torch.float32)

    def __len__(self):
        return len(self.all_pose)
    
    def __getitem__(self, idx):
        item = {
            "kpts": self.all_kpts[idx], #max_poses, num_kpts, kpt_dims
            "presence": self.all_presence[idx], #max_poses + 1
            "pc": self.all_pc[idx],
        }
        return item
    
    def get_dataloader(self) -> DataLoader:
        return DataLoader(self,
                          batch_size = self.batch_size,
                          shuffle = self.shuffle,
                          collate_fn = None)

class oldposeidonDataLoaderMars(poseidonDataLoader):
    def __init__(self, subjects: list, 
                 dynamic_loading: bool = True,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 pc_grid_dim: int = 8,
                 data_build: str = "POSEIDON_build_w0.083"):
        super().__init__(subjects, dynamic_loading, batch_size, shuffle, pc_grid_dim, data_build)
        #self.preprocess_data_multithreaded()
        #print("Data preprocessing complete")

    def compile_all_pose_and_pc(self):
        #TODO: build a frame_to_origin_path.csv instead
        all_pose_dfs = []
        all_pc_dfs = []
        for cohort_path in self.synchronized_data_path.iterdir():
            if cohort_path.is_file(): continue
            for subject_path in cohort_path.iterdir():
                if subject_path.is_file() or subject_path.name not in self.subjects: continue
                for session_path in subject_path.iterdir():
                    if session_path.is_file(): continue
                    pose_file_path = session_path / "pose.csv"
                    pc_file_path = session_path / "pc.csv"
                    pose_df = pd.read_csv(pose_file_path, na_filter=True)
                    pc_df = pd.read_csv(pc_file_path, na_filter=True)
                    relative_path = Path(*session_path.parts[-3:])
                    print(relative_path)
                    pose_df['path'] = relative_path
                    pc_df['path'] = relative_path
                    all_pose_dfs.append(pose_df)
                    all_pc_dfs.append(pc_df)
        self.all_pose = pd.concat(all_pose_dfs)
        self.all_pc = pd.concat(all_pc_dfs)
        return

    def define_record_list(self):
        #TODO: make this dependent on the RGB frame numbers specified in the ground truth and then move it to abstract

        undersample = random.sample(range(len(self.all_pose)), 100)  # Generate 100 random integer indices
        self.all_pose = self.all_pose.iloc[undersample]
        self.all_pose['record_id'] = pd.factorize(self.all_pose[['frame_number_rgb', 'path']].apply(tuple,axis=1))[0]
        return
    
    
    def preprocess_data_threadpool(self):
        """
        Preprocess all pose and point cloud records during initialization
        using multi-threading for faster execution.
        """
        
        unique_record_ids = self.all_pose['record_id'].unique()
        pbar_groups = tqdm(total=len(unique_record_ids), leave=True, desc="Processing Records", file=sys.stdout, ncols=80)

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                # Submit tasks for each record ID
                future_to_record = {
                    executor.submit(preprocess_record, record_id, self.all_pose, self.all_pc, self.transform_pc_frame, self.transform_pose_frame): record_id
                    for record_id in unique_record_ids
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_record):
                    record_id, processed_item = future.result()
                    self.processed_data[record_id] = processed_item
                    pbar_groups.update(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Shutting down...")
                executor.shutdown(wait=False)  # Stop threads immediately
                raise
        pbar_groups.close()
        return self.processed_data
    
    def preprocess_data_processpool(self):
        """
        Preprocess all pose and point cloud records during initialization
        using multi-threading for faster execution.
        """
        
        unique_record_ids = self.all_pose['record_id'].unique()
        pbar_groups = tqdm(total=len(unique_record_ids), leave=True, desc="Processing Records", file=sys.stdout, ncols=80)

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            try:
                # Submit tasks for each record ID
                future_to_record = {
                    executor.submit(preprocess_record, record_id, self.all_pose, self.all_pc, self.transform_pc_frame, self.transform_pose_frame): record_id
                    for record_id in unique_record_ids
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_record):
                    record_id, processed_item = future.result()
                    self.processed_data[record_id] = processed_item
                    pbar_groups.update(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Shutting down...")
                executor.shutdown(wait=False)  # Stop threads immediately
                raise
        pbar_groups.close()
        return self.processed_data

    def preprocess_data_batched(self):
        """
        Preprocess data in batches using multiprocessing for efficiency.
        """
        unique_record_ids = self.all_pose['record_id'].unique()
        batch_size = 50  # Adjust this based on your system and workload
        record_batches = [
            unique_record_ids[i:i + batch_size] for i in range(0, len(unique_record_ids), batch_size)
        ]
    
        # Use tqdm to track progress
        pbar = tqdm(total=len(record_batches), desc="Processing Records", file=sys.stdout, ncols=80)
    
        # Use multiprocessing Pool for parallel execution
        #with Pool(processes=min(os.cpu_count() // 2, 16)) as pool:
        with Pool(processes=50) as pool:
            try:
                # Pass arguments as tuples
                args_list = [
                    (batch, self.all_pose, self.all_pc, self.transform_pc_frame, self.transform_pose_frame)
                    for batch in record_batches
                ]
                # Use pool.imap_unordered for better load balancing
                results = pool.imap_unordered(
                    preprocess_record_batch,
                    args_list
                )
    
                # Collect and merge results
                for batch_results in results:
                    self.processed_data.update(batch_results)
                    pbar.update(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Terminating workers...")
                pool.terminate()
                raise
            finally:
                pool.close()
                pool.join()
    
        pbar.close()
        return self.processed_data


    def save_processed_data(self, save_path):
        with open(save_path, 'wb') as handle:
            pickle.dump(self.processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_processed_data(self, load_path):
        with open(load_path, 'rb') as handle:
            self.processed_data = pickle.load(handle)

    def pc_array_to_grid(self, pc):
        unsorted_pc_grid = np.zeros((self.max_pcs, self.pc_dims), dtype=pc.dtype)
        num_points = pc.shape[0]
        if num_points > self.max_pcs:
            #randomly remove points
            #TODO: maybe remove points with lowest SNR instead
            indices = np.arange(num_points)
            np.random.shuffle(indices)
            indices = indices[:self.max_pcs]
            unsorted_pc_grid = pc[indices]
        if num_points < self.max_pcs:
            #0 pad
            padded_arr = np.zeros((self.max_pcs, self.pc_dims), dtype=pc.dtype)
            padded_arr[:num_points] = pc
            unsorted_pc_grid = padded_arr
        
        unsorted_pc_grid = unsorted_pc_grid.reshape((self.max_pcs, self.pc_dims))
        pc_grid = np.array(unsorted_pc_grid[np.lexsort((unsorted_pc_grid[:, 2], unsorted_pc_grid[:, 1], unsorted_pc_grid[:, 0]))])
        pc_grid = pc_grid.reshape(self.pc_grid_dim, self.pc_grid_dim, self.pc_dims)
        return pc_grid

    def transform_pc_frame(self, pc_df: pd.core.frame.DataFrame) -> torch.Tensor: 
        pc_np = pc_df.to_numpy()
        #reshape to a sorted grid for mpMARS
        pc_grid = self.pc_array_to_grid(pc_np)
        return torch.Tensor(pc_grid)
    
    def pose_array_to_posepresence(self, poses):
        '''
        Process pose estimate labels into padded poses and presence arrays.

        Args:
            poses (np.ndarray): Array of shape (n, self.num_kpts, self.kpt_dims)
        Returns:
            pposes (np.ndarray): Array of (self.max_poses, self.num_kpts, self.kpt_dims) with the poses padded to size self.max_poses.
        '''
        # Number of poses in the current frame
        n = poses.shape[0]
        pposes = np.zeros((self.max_poses, self.num_kpts, self.kpt_dims), dtype=poses.dtype)
        pposes[:n] = poses  # Copy the poses into the padded array
        # Create a one-hot encoded presence array
        ppresence = np.zeros((self.max_poses+1,), dtype=np.int32)
        ppresence[n] = 1 #one hot encoding
        return pposes, ppresence
    
    def transform_pose_frame(self, poses_df: pd.core.frame.DataFrame) -> torch.Tensor:
        if poses_df.isna().all().all():
            poses_np = poses_df.to_numpy()
            poses = np.zeros((self.max_poses, self.num_kpts, self.kpt_dims))
            presence = np.zeros((self.max_poses+1,), dtype=np.int32)
            presence[0] = 1
            return torch.Tensor(poses), torch.Tensor(presence)
        poses_x_df = poses_df[BODY_8_KPT_COLS_X]
        poses_y_df = poses_df[BODY_8_KPT_COLS_Y]
        poses_x_np = poses_x_df.to_numpy()
        poses_y_np = poses_y_df.to_numpy()
        #print(f"poses_x_np: {poses_x_np.shape}")
        #print(f"poses_y_np: {poses_y_np.shape}")
        poses_np = np.stack((poses_x_np,poses_y_np),axis=-1)
        #print(poses_np.shape)
        poses, presence = self.pose_array_to_posepresence(poses_np)
        return torch.Tensor(poses), torch.Tensor(presence)
    
    def __len__(self):
        if self.processed_data:
            return len(self.processed_data)
        else:
            return len(self.all_pose['record_id'].unique())
    
    def __getitem__(self, idx):
        if self.processed_data:
            print("Doing this")
            return self.processed_data[idx]
        pose_record = self.all_pose[self.all_pose['record_id']==idx]
        frame_number_rgb = pose_record['frame_number_rgb'].iloc[0]
        session_path = pose_record['path'].iloc[0]
        pc_records = self.all_pc[(self.all_pc['frame_number_rgb']==frame_number_rgb) & (self.all_pc['path']==session_path)]
        
        pcf = pc_records.drop(['frame_number_rgb', 'frame_number_pc', 'path'],axis=1)
        posef = pose_record.drop(['frame_number_rgb','path', 'record_id'], axis=1)  
        pcs = self.transform_pc_frame(pcf)
        poses, presence = self.transform_pose_frame(posef)
        item = {"kpts": poses,
                "presence": presence,
                "pc": pcs}
                #"path": session_path,
                #"frame_number_rgb": frame_number_rgb}
        return item


class poseidonDataLoaderMarsRgb(poseidonDataLoaderMars):
    def __init__(self, subjects: list, 
                 dynamic_loading: bool = True,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 pc_grid_dim: int = 8,
                 data_build: str = "POSEIDON_build_w0.083"):
        super().__init__(subjects, dynamic_loading, batch_size, shuffle, pc_grid_dim, data_build)

        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
    
    def sanity_check(self):
        return    
    
    def __getitem__(self, idx):
        pose_record = self.all_pose[self.all_pose['record_id']==idx]
        frame_number_rgb = pose_record['frame_number_rgb'].iloc[0]
        session_path = pose_record['path'].iloc[0]
        pc_records = self.all_pc[(self.all_pc['frame_number_rgb']==frame_number_rgb) & (self.all_pc['path']==session_path)]
        image_file_name = f"frame_{frame_number_rgb}.jpg"
        image_file_path = self.synchronized_data_path / session_path / "rgb_frames" /image_file_name
        image = Image.open(image_file_path)
        pcf = pc_records.drop(['frame_number_rgb', 'frame_number_pc', 'path'],axis=1)
        posef = pose_record.drop(['frame_number_rgb','path', 'record_id'], axis=1)  
        pcs = self.transform_pc_frame(pcf)
        poses, presence = self.transform_pose_frame(posef)
        item = {"kpts": poses,
                "presence": presence,
                "pc": pcs, 
                "rgb": self.transform(image)}
                #"path": session_path,
                #"frame_number_rgb": frame_number_rgb}
        return item
    

if __name__ == "__main__":
    tr_subjects = ["Catherine Lemech"]
    
    pdl = poseidonDataLoaderMars(subjects=["Catherine Lemech"])
    dl = pdl.get_dataloader()
    max_i = 1
    for i, batch in enumerate(dl):
        print(f"batch['pose']: {batch['kpts'].shape}")
        print(f"batch['presence']: {batch['presence'].shape}")
        if i > max_i:
            break
    
