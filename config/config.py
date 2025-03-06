from pydantic import BaseModel
from typing import Literal

class Config(BaseModel):
    # Directory configurations
    frame_dir: str  # Directory where frames are stored
    save_dir: str  # Directory to save dataset information
    store_dir: str  # Directory to save model checkpoints, predictions, etc.
    store_mode: Literal["store", "load"]  # 'store' if it's the first time running the script to prepare and store dataset information, or 'load' to load previously stored information
    
    # Training parameters
    batch_size: int  # Batch size for training
    clip_len: int  # Length of the clips in number of frames
    crop_dim: int | None  # Dimension to crop the frames (if needed)
    dataset: str  # Name of the dataset ('finediving', 'fs_comp', 'fs_perf', or 'soccernet')
    radi_displacement: int  # Radius of displacement used
    epoch_num_frames: int  # Number of frames used per epoch
    feature_arch: Literal["rny002_gsf", "rny008_gsf"]  # Feature extractor architecture
    learning_rate: float  # Learning rate for training
    mixup: bool  # Whether to use mixup augmentation
    modality: Literal["rgb"]  # Input modality used
    num_classes: int  # Number of classes for the current dataset
    num_epochs: int  # Number of epochs for training
    warm_up_epochs: int  # Number of warm-up epochs
    start_val_epoch: int  # Epoch where validation evaluation starts
    
    # Model architecture parameters
    temporal_arch: Literal["ed_sgp_mixer"]  # Temporal architecture used
    n_layers: int  # Number of blocks/layers used for the temporal architecture
    sgp_ks: int  # Kernel size of the SGP and SGP-Mixer layers
    sgp_r: int  # r factor in SGP and SGP-Mixer layers
    
    # Evaluation parameters
    only_test: bool  # Whether to perform only inference or training + inference
    criterion: Literal["map", "loss"]  # Criterion used for validation evaluation
    num_workers: int  # Number of workers for data loading

